# Copyright (c) 2025 Resemble AI
# MIT License
import logging
import time
from typing import Callable, Iterator, Optional, Sequence

from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from transformers import LlamaModel, LlamaConfig, GPT2Config, GPT2Model, StaticCache
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    MinPLogitsWarper,
)
from .modules.learned_pos_emb import LearnedPositionEmbeddings

from .modules.cond_enc import T3CondEnc, T3Cond
from .modules.t3_config import T3Config
from .llama_configs import LLAMA_CONFIGS
from .inference.t3_hf_backend import T3HuggingfaceBackend
from ..utils import AttrDict


logger = logging.getLogger(__name__)


def _ensure_BOT_EOT(text_tokens: Tensor, hp):
    B = text_tokens.size(0)
    assert (text_tokens == hp.start_text_token).int().sum() >= B, (
        "missing start_text_token"
    )
    assert (text_tokens == hp.stop_text_token).int().sum() >= B, (
        "missing stop_text_token"
    )


class T3(nn.Module):
    """
    Token-To-Token (T3) TTS model using huggingface transformer models as backbones,
        * tokenization, including start / stop tokens are always added externally to this class
        * conditioning data like CLAP, emotion, etc are all in a separate file for more modularity
        * careful! this class assumes relative positional encoding -- with absolute PE, we would at
            least want to reset the position to 0 when speech tokens begin, and optionally use a
            different PE embedding space for speech.
    """

    def __init__(self, hp=None):
        if hp is None:
            hp = T3Config.english_only()
        super().__init__()
        self.hp = hp

        config_dict = LLAMA_CONFIGS[hp.llama_config_name]
        self.is_gpt = config_dict.get("model_type") == "gpt2"

        if self.is_gpt:
            self.cfg = GPT2Config(**config_dict)
            self.tfmr = GPT2Model(self.cfg)
        else:
            self.cfg = LlamaConfig(**config_dict)
            self.tfmr = LlamaModel(self.cfg)

        self.dim = self.cfg.hidden_size
        self.deepspeed_patch_applied = False

        # conditioning / embedding
        self.cond_enc = T3CondEnc(hp)
        self.text_emb = nn.Embedding(hp.text_tokens_dict_size, self.dim)
        self.speech_emb = nn.Embedding(hp.speech_tokens_dict_size, self.dim)

        # custom position embedding
        self.text_pos_emb = None
        self.speech_pos_emb = None
        if hp.input_pos_emb == "learned":
            max_text_seq_len = hp.max_text_tokens + 2
            self.text_pos_emb = LearnedPositionEmbeddings(max_text_seq_len, self.dim)

            max_mel_seq_len = hp.max_speech_tokens + 2 + 2
            self.speech_pos_emb = LearnedPositionEmbeddings(max_mel_seq_len, self.dim)

        # logit projection
        self.text_head = nn.Linear(
            self.cfg.hidden_size, hp.text_tokens_dict_size, bias=False
        )
        self.speech_head = nn.Linear(
            self.cfg.hidden_size, hp.speech_tokens_dict_size, bias=self.is_gpt
        )
        self.compiled = False

    @property
    def device(self):
        return self.speech_head.weight.device

    def prepare_conditioning(self, t3_cond: T3Cond):
        """
        Token cond data needs to be embedded, so that needs to be here instead of in `T3CondEnc`.
        """
        if (
            t3_cond.cond_prompt_speech_tokens is not None
            and t3_cond.cond_prompt_speech_emb is None
        ):
            t3_cond.cond_prompt_speech_emb = self.speech_emb(
                t3_cond.cond_prompt_speech_tokens
            )
            if not self.is_gpt:
                t3_cond.cond_prompt_speech_emb += self.speech_pos_emb(
                    t3_cond.cond_prompt_speech_tokens
                )
        return self.cond_enc(t3_cond)  # (B, len_cond, dim)

    def prepare_input_embeds(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: torch.LongTensor,
        speech_tokens: torch.LongTensor,
        cfg_weight: float = 0.0,
    ):
        # prepare input embeddings (skip backbone tranformer embeddings)
        cond_emb = self.prepare_conditioning(t3_cond)  # (B, len_cond, dim)
        text_emb = self.text_emb(text_tokens)  # (B, len_text, dim)
        if cfg_weight > 0.0 and not self.is_gpt:
            text_emb[1].zero_()  # CFG uncond

        speech_emb = self.speech_emb(speech_tokens)  # (B, len_speech, dim)
        if self.hp.input_pos_emb == "learned":
            text_emb = text_emb + self.text_pos_emb(text_tokens)
            speech_emb = speech_emb + self.speech_pos_emb(speech_tokens)
        len_cond = cond_emb.size(1)

        if cond_emb.size(0) != text_emb.size(0):
            cond_emb = cond_emb.expand(text_emb.size(0), -1, -1)

        # concat
        embeds = torch.stack(
            [
                torch.cat((ce, te, se))
                for ce, te, se in zip(cond_emb, text_emb, speech_emb)
            ]
        )  # (B, length, dim)
        return embeds, len_cond

    def forward(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: torch.LongTensor,
        text_token_lens: torch.LongTensor,
        speech_tokens: torch.LongTensor,
        speech_token_lens: torch.LongTensor,
        training=False,
    ):
        _ensure_BOT_EOT(text_tokens, self.hp)

        # prepare custom input embeds
        embeds, len_cond = self.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=speech_tokens,
        )

        # backbone tranformer forward
        tfmr_out = self.tfmr.forward(
            input_ids=None,
            # position_ids=position_ids, # TODO? ROPE should be fine?
            inputs_embeds=embeds,
            output_hidden_states=True,
            return_dict=True,
            use_cache=(not training),
        )
        hidden_states = tfmr_out.hidden_states[
            -1
        ]  # final tfmr layer output, (B, seq, dim)

        # post-processing: splice out text and speech parts of hidden states
        len_text = text_tokens.size(1)
        len_speech = speech_tokens.size(1)
        B, _, dim = hidden_states.shape
        device, dtype = hidden_states.device, hidden_states.dtype
        text_latents = torch.zeros(B, len_text, dim, dtype=dtype, device=device)
        speech_latents = torch.zeros(B, len_speech, dim, dtype=dtype, device=device)
        ttl, stl = text_token_lens, speech_token_lens
        for i in range(B):
            text_end = len_cond + ttl[i].item()
            speech_start = len_cond + text_tokens.size(1)
            speech_end = speech_start + stl[i].item()
            text_latents[i, : ttl[i]] = hidden_states[i, len_cond:text_end]
            speech_latents[i, : stl[i]] = hidden_states[i, speech_start:speech_end]

        # logit projection
        text_logits = self.text_head(text_latents)
        speech_logits = self.speech_head(speech_latents)

        return AttrDict(
            text_logits=text_logits,
            text_latents=text_latents,
            speech_logits=speech_logits,
            speech_latents=speech_latents,
            hidden_states=hidden_states,
        )

    def loss(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: torch.LongTensor,
        text_token_lens: torch.LongTensor,
        speech_tokens: torch.LongTensor,
        speech_token_lens: torch.LongTensor,
    ):
        "training method"
        len_text = text_tokens.size(1)
        len_speech = speech_tokens.size(1)
        assert len_text == text_token_lens.max()
        assert len_speech == speech_token_lens.max()

        out = self.forward(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            text_token_lens=text_token_lens,
            speech_tokens=speech_tokens,
            speech_token_lens=speech_token_lens,
            training=True,
        )  # (B, seq, vocab_size)

        # Calc CCE losses
        IGNORE_ID = -100
        device = out.text_logits.device
        mask_text = (
            torch.arange(len_text, device=device)[None] >= text_token_lens[:, None]
        )  # (B, len_text)
        mask_speech = (
            torch.arange(len_speech, device=device)[None] >= speech_token_lens[:, None]
        )  # (B, len_speech)
        masked_text = text_tokens.masked_fill(mask_text, IGNORE_ID)
        masked_speech = speech_tokens.masked_fill(mask_speech, IGNORE_ID)
        loss_text = F.cross_entropy(
            out.text_logits, masked_text, ignore_index=IGNORE_ID
        )
        loss_speech = F.cross_entropy(
            out.speech_logits, masked_speech, ignore_index=IGNORE_ID
        )

        return loss_text, loss_speech

    @torch.inference_mode()
    def inference(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: Tensor,
        initial_speech_tokens: Optional[Tensor] = None,
        # misc conditioning
        prepend_prompt_speech_tokens: Optional[Tensor] = None,
        # HF generate args
        num_return_sequences=1,
        max_new_tokens=None,
        stop_on_eos=True,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        min_p=0.05,
        length_penalty=1.0,
        repetition_penalty=1.2,
        cfg_weight=0.5,
    ):
        """
        Args:
            text_tokens: a 1D (unbatched) or 2D (batched) tensor.
        """
        # Validate / sanitize inputs
        assert prepend_prompt_speech_tokens is None, "not implemented"
        _ensure_BOT_EOT(text_tokens, self.hp)
        text_tokens = torch.atleast_2d(text_tokens).to(
            dtype=torch.long, device=self.device
        )

        # Default initial speech to a single start-of-speech token
        if initial_speech_tokens is None:
            initial_speech_tokens = self.hp.start_speech_token * torch.ones_like(
                text_tokens[:, :1]
            )

        # Prepare custom input embeds
        embeds, len_cond = self.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=initial_speech_tokens,
            cfg_weight=cfg_weight,
        )

        # In order to use the standard HF generate method, we need to extend some methods to inject our custom logic
        # Note the llama-specific logic. Other tfmr types can be added later.

        self.compiled = False

        # TODO? synchronize the expensive compile function
        # with self.compile_lock:
        if not self.compiled:
            patched_model = T3HuggingfaceBackend(
                config=self.cfg,
                llama=self.tfmr,
                speech_enc=self.speech_emb,
                speech_head=self.speech_head,
            )
            self.patched_model = patched_model
            self.compiled = True

        # # Run normal generate method, which calls our custom extended methods
        # return self.patched_model.generate(
        #     inputs=initial_speech_tokens,
        #     decoder_cond=embeds,
        #     bos_token_id=self.hp.start_speech_token,
        #     eos_token_id=(self.hp.stop_speech_token if stop_on_eos else -1),
        #     pad_token_id=self.hp.stop_speech_token,
        #     max_new_tokens=max_new_tokens or self.hp.max_speech_tokens,
        #     num_return_sequences=num_return_sequences,
        #     temperature=temperature,
        #     min_p=min_p,
        #     length_penalty=length_penalty,
        #     repetition_penalty=repetition_penalty,
        #     do_sample=do_sample,
        #     # cache_implementation=None if not self.compiled else "static",
        # )

        device = embeds.device

        bos_token = torch.tensor(
            [[self.hp.start_speech_token]], dtype=torch.long, device=device
        )
        bos_embed = self.speech_emb(bos_token)  # shape: (B, 1, embed_dim)
        bos_embed = bos_embed + self.speech_pos_emb.get_fixed_embedding(0)

        # batch_size=2 for CFG
        bos_embed = torch.cat([bos_embed, bos_embed])

        # Combine condition and BOS token for the initial input
        inputs_embeds = torch.cat([embeds, bos_embed], dim=1)

        # Track generated token ids; start with the BOS token.
        generated_ids = bos_token.clone()
        predicted = []  # To store the predicted tokens

        # Instantiate the logits processors.
        top_p_warper = TopPLogitsWarper(top_p=top_p)
        min_p_warper = MinPLogitsWarper(min_p=min_p)
        top_p_warper = TopPLogitsWarper(top_p=top_p)
        repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(
            penalty=float(repetition_penalty)
        )

        # ---- Initial Forward Pass (no kv_cache yet) ----
        output = self.patched_model(
            inputs_embeds=inputs_embeds,
            past_key_values=None,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        # Initialize kv_cache with the full context.
        past = output.past_key_values

        # ---- Generation Loop using kv_cache ----
        for i in tqdm(range(max_new_tokens), desc="Sampling", dynamic_ncols=True):
            logits_step = output.logits[:, -1, :]
            # CFG combine  → (1, V)
            cond = logits_step[0:1, :]
            uncond = logits_step[1:2, :]
            cfg = torch.as_tensor(cfg_weight, device=cond.device, dtype=cond.dtype)
            logits = cond + cfg * (cond - uncond)

            # Apply repetition penalty
            ids_for_proc = generated_ids[:1, ...]  # batch = 1
            logits = repetition_penalty_processor(ids_for_proc, logits)  # expects (B,V)

            # Apply temperature scaling.
            if temperature != 1.0:
                logits = logits / temperature

            # Apply min_p and top_p filtering
            logits = min_p_warper(ids_for_proc, logits)
            logits = top_p_warper(ids_for_proc, logits)

            # Convert logits to probabilities and sample the next token.
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # shape: (B, 1)

            predicted.append(next_token)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # Check for EOS token.
            if next_token.view(-1) == self.hp.stop_speech_token:
                logger.info(
                    f"✅ EOS token detected! Stopping generation at step {i + 1}"
                )
                break

            # Get embedding for the new token.
            next_token_embed = self.speech_emb(next_token)
            next_token_embed = (
                next_token_embed + self.speech_pos_emb.get_fixed_embedding(i + 1)
            )

            #  For CFG
            next_token_embed = torch.cat([next_token_embed, next_token_embed])

            # Forward pass with only the new token and the cached past.
            output = self.patched_model(
                inputs_embeds=next_token_embed,
                past_key_values=past,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # Update the kv_cache.
            past = output.past_key_values

        # Concatenate all predicted tokens along the sequence dimension.
        predicted_tokens = torch.cat(predicted, dim=1)  # shape: (B, num_tokens)
        return predicted_tokens

    @torch.inference_mode()
    def iter_inference_turbo(
        self,
        t3_cond,
        text_tokens,
        temperature=0.8,
        top_k=1000,
        top_p=0.95,
        repetition_penalty=1.2,
        max_gen_len=1000,
        show_progress=False,
    ) -> Iterator[torch.Tensor]:
        """Yield Turbo speech tokens incrementally using the KV-cache path."""

        logits_processors = LogitsProcessorList()
        if temperature > 0 and temperature != 1.0:
            logits_processors.append(TemperatureLogitsWarper(temperature))
        if top_k > 0:
            logits_processors.append(TopKLogitsWarper(top_k))
        if top_p < 1.0:
            logits_processors.append(TopPLogitsWarper(top_p))
        if repetition_penalty != 1.0:
            logits_processors.append(
                RepetitionPenaltyLogitsProcessor(repetition_penalty)
            )

        speech_start_token = self.hp.start_speech_token * torch.ones_like(
            text_tokens[:, :1]
        )
        embeds, _ = self.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=speech_start_token,
            cfg_weight=0.0,
        )

        generated_speech_tokens = []

        llm_outputs = self.tfmr(inputs_embeds=embeds, use_cache=True)

        hidden_states = llm_outputs[0]
        past_key_values = llm_outputs.past_key_values

        speech_hidden = hidden_states[:, -1:]
        speech_logits = self.speech_head(speech_hidden)

        processed_logits = logits_processors(
            speech_start_token, speech_logits[:, -1, :]
        )
        probs = F.softmax(processed_logits, dim=-1)
        next_speech_token = torch.multinomial(probs, num_samples=1)

        generated_speech_tokens.append(next_speech_token)
        current_speech_token = next_speech_token
        if torch.all(current_speech_token == self.hp.stop_speech_token):
            return
        yield current_speech_token

        steps = range(max_gen_len)
        if show_progress:
            steps = tqdm(steps)

        for _ in steps:
            current_speech_embed = self.speech_emb(current_speech_token)

            llm_outputs = self.tfmr(
                inputs_embeds=current_speech_embed,
                past_key_values=past_key_values,
                use_cache=True,
            )

            hidden_states = llm_outputs[0]
            past_key_values = llm_outputs.past_key_values
            speech_logits = self.speech_head(hidden_states)

            input_ids = torch.cat(generated_speech_tokens, dim=1)
            processed_logits = logits_processors(input_ids, speech_logits[:, -1, :])
            if torch.all(processed_logits == -float("inf")):
                print("Warning: All logits are -inf")
                break

            probs = F.softmax(processed_logits, dim=-1)
            next_speech_token = torch.multinomial(probs, num_samples=1)

            generated_speech_tokens.append(next_speech_token)
            current_speech_token = next_speech_token
            if torch.all(next_speech_token == self.hp.stop_speech_token):
                break
            yield current_speech_token

    @torch.inference_mode()
    def iter_inference_turbo_batch(
        self,
        t3_cond,
        text_tokens,
        text_attention_mask=None,
        temperature=0.8,
        top_k=1000,
        top_p=0.95,
        repetition_penalty=1.2,
        max_gen_len=1000,
    ) -> Iterator[tuple[Tensor, Tensor, Tensor]]:
        """Yield one speech-token step for a padded batch.

        Each item is ``(tokens, valid, finished)`` with shape ``[batch]``.
        ``valid`` excludes EOS tokens and rows that had already completed;
        ``finished`` is cumulative. Finished rows remain in the microbatch with
        a masked query so other rows can keep decoding without cache reshaping.
        """

        text_tokens = torch.atleast_2d(text_tokens).to(
            dtype=torch.long,
            device=self.device,
        )
        if text_attention_mask is None:
            text_attention_mask = torch.ones_like(text_tokens)
        else:
            text_attention_mask = torch.atleast_2d(text_attention_mask).to(
                dtype=torch.long,
                device=self.device,
            )
        if text_attention_mask.shape != text_tokens.shape:
            raise ValueError("text_attention_mask must match text_tokens")

        logits_processors = LogitsProcessorList()
        if temperature > 0 and temperature != 1.0:
            logits_processors.append(TemperatureLogitsWarper(temperature))
        if top_k > 0:
            logits_processors.append(TopKLogitsWarper(top_k))
        if top_p < 1.0:
            logits_processors.append(TopPLogitsWarper(top_p))
        if repetition_penalty != 1.0:
            logits_processors.append(
                RepetitionPenaltyLogitsProcessor(repetition_penalty),
            )

        batch_size = text_tokens.shape[0]
        speech_start_token = torch.full(
            (batch_size, 1),
            self.hp.start_speech_token,
            dtype=torch.long,
            device=self.device,
        )
        embeds, len_cond = self.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=speech_start_token,
            cfg_weight=0.0,
        )
        attention_mask = torch.cat(
            [
                torch.ones(
                    batch_size,
                    len_cond,
                    dtype=torch.long,
                    device=self.device,
                ),
                text_attention_mask,
                torch.ones(
                    batch_size,
                    1,
                    dtype=torch.long,
                    device=self.device,
                ),
            ],
            dim=1,
        )
        position_ids = attention_mask.cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)

        outputs = self.tfmr(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        generated_ids = speech_start_token
        finished = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=self.device,
        )

        for _ in range(max_gen_len):
            speech_logits = self.speech_head(outputs[0][:, -1, :])
            processed_logits = logits_processors(generated_ids, speech_logits)
            if torch.any(torch.all(processed_logits == -float("inf"), dim=-1)):
                raise RuntimeError("all Turbo logits are -inf for at least one request")

            probs = F.softmax(processed_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            next_token = torch.where(
                finished[:, None],
                torch.full_like(next_token, self.hp.stop_speech_token),
                next_token,
            )
            was_finished = finished.clone()
            finished |= next_token[:, 0] == self.hp.stop_speech_token
            valid = ~was_finished & ~finished
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            yield next_token[:, 0], valid, finished.clone()

            if torch.all(finished):
                break

            query_mask = (~finished).to(dtype=torch.long)[:, None]
            attention_mask = torch.cat([attention_mask, query_mask], dim=1)
            query_position_ids = attention_mask.cumsum(-1)[:, -1:] - 1
            query_position_ids.masked_fill_(query_mask == 0, 0)
            speech_embed = self.speech_emb(next_token)
            outputs = self.tfmr(
                inputs_embeds=speech_embed,
                attention_mask=attention_mask,
                position_ids=query_position_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values

    @torch.inference_mode()
    def iter_inference_turbo_live_batch(
        self,
        t3_cond,
        text_supplier: Callable[
            [Sequence[bool]],
            tuple[Tensor, Tensor, Sequence[int], Sequence[bool], Sequence[bool]],
        ],
        temperature=0.8,
        top_k=1000,
        top_p=0.95,
        repetition_penalty=1.2,
        max_gen_len=1000,
        idle_sleep_seconds=0.005,
        use_cuda_graph=True,
        cuda_graph_capture_after_tokens=12,
    ) -> Iterator[tuple[Tensor, Tensor, Tensor]]:
        """Generate speech while the conditioning text is still growing.

        Turbo is a decoder-only model whose text appears before speech in the
        prompt. When text grows, the KV cache is rebuilt from the longer text
        and the speech tokens already generated. This preserves the spoken
        prefix while allowing the next speech token to see newly arrived text.

        ``text_supplier`` receives the rows currently waiting at EOS and
        returns padded text tokens, their attention mask, monotonically
        increasing row versions, input-complete flags, and cancellation flags.
        """

        logits_processors = LogitsProcessorList()
        if temperature > 0 and temperature != 1.0:
            logits_processors.append(TemperatureLogitsWarper(temperature))
        if top_k > 0:
            logits_processors.append(TopKLogitsWarper(top_k))
        if top_p < 1.0:
            logits_processors.append(TopPLogitsWarper(top_p))
        if repetition_penalty != 1.0:
            logits_processors.append(
                RepetitionPenaltyLogitsProcessor(repetition_penalty),
            )

        initial = text_supplier([])
        text_tokens = torch.atleast_2d(initial[0]).to(
            dtype=torch.long,
            device=self.device,
        )
        batch_size = text_tokens.shape[0]
        if batch_size < 1:
            raise ValueError("live text batch must contain at least one request")

        start_token = self.hp.start_speech_token
        stop_token = self.hp.stop_speech_token
        histories: list[list[int]] = [[start_token] for _ in range(batch_size)]
        current_versions = [-1] * batch_size
        waiting = [False] * batch_size
        finished = [False] * batch_size
        attention_mask = None
        past_key_values = None
        outputs = None
        last_positions = None
        rebuilt = False
        cuda_graph = None
        cuda_graph_pending = False
        static_attention_mask = None
        static_cache_position = None
        static_position_ids = None
        static_embed = None
        static_hidden_states = None
        graph_physical_position = 0

        def clear_cuda_graph() -> None:
            nonlocal cuda_graph, cuda_graph_pending
            nonlocal static_attention_mask, static_cache_position
            nonlocal static_position_ids, static_embed, static_hidden_states
            nonlocal graph_physical_position
            cuda_graph = None
            cuda_graph_pending = False
            static_attention_mask = None
            static_cache_position = None
            static_position_ids = None
            static_embed = None
            static_hidden_states = None
            graph_physical_position = 0

        def prepare_cuda_graph_prefix():
            """Rebuild the completed prompt into a static cache after startup."""

            nonlocal past_key_values, cuda_graph_pending
            nonlocal static_attention_mask, static_cache_position
            nonlocal static_position_ids, static_embed
            nonlocal graph_physical_position, last_positions

            max_speech_length = max(len(history) for history in histories)
            speech_tokens = torch.full(
                (batch_size, max_speech_length),
                start_token,
                dtype=torch.long,
                device=self.device,
            )
            speech_attention_mask = torch.zeros_like(speech_tokens)
            speech_lengths = []
            for index, history in enumerate(histories):
                length = len(history)
                speech_lengths.append(length)
                speech_tokens[index, :length] = torch.as_tensor(
                    history,
                    dtype=torch.long,
                    device=self.device,
                )
                speech_attention_mask[index, :length] = 1

            embeds, len_cond = self.prepare_input_embeds(
                t3_cond=t3_cond,
                text_tokens=text_tokens,
                speech_tokens=speech_tokens,
                cfg_weight=0.0,
            )
            prefix_attention_mask = torch.cat(
                [
                    torch.ones(
                        batch_size,
                        len_cond,
                        dtype=torch.long,
                        device=self.device,
                    ),
                    text_attention_mask,
                    speech_attention_mask,
                ],
                dim=1,
            )
            prefix_position_ids = prefix_attention_mask.cumsum(-1) - 1
            prefix_position_ids.masked_fill_(prefix_attention_mask == 0, 0)

            max_cache_len = embeds.shape[1] + max_gen_len + 1
            past_key_values = StaticCache(
                config=self.cfg,
                max_cache_len=max_cache_len,
            )
            prefill_cache_position = torch.arange(
                embeds.shape[1],
                dtype=torch.long,
                device=self.device,
            )
            prefix_outputs = self.tfmr(
                inputs_embeds=embeds,
                attention_mask=prefix_attention_mask,
                position_ids=prefix_position_ids,
                past_key_values=past_key_values,
                cache_position=prefill_cache_position,
                use_cache=True,
            )
            static_attention_mask = torch.zeros(
                batch_size,
                max_cache_len,
                dtype=prefix_attention_mask.dtype,
                device=self.device,
            )
            static_attention_mask[:, : prefix_attention_mask.shape[1]].copy_(
                prefix_attention_mask,
            )
            static_cache_position = torch.tensor(
                [embeds.shape[1]],
                dtype=torch.long,
                device=self.device,
            )
            static_position_ids = prefix_attention_mask.sum(
                dim=-1,
                keepdim=True,
            )
            static_embed = torch.empty(
                batch_size,
                1,
                self.cfg.hidden_size,
                dtype=embeds.dtype,
                device=self.device,
            )
            graph_physical_position = embeds.shape[1]
            last_positions = torch.tensor(
                [
                    len_cond + text_tokens.shape[1] + length - 1
                    for length in speech_lengths
                ],
                dtype=torch.long,
                device=self.device,
            )
            cuda_graph_pending = True
            return prefix_outputs, prefix_attention_mask

        def run_cuda_graph_step(
            next_tokens: Tensor,
            continuing: Tensor,
        ) -> tuple[Tensor]:
            """Consume one sampled token through a fixed-shape CUDA graph."""

            nonlocal cuda_graph, cuda_graph_pending, static_hidden_states
            nonlocal graph_physical_position

            static_embed.copy_(self.speech_emb(next_tokens[:, None]))
            static_attention_mask[:, graph_physical_position].copy_(
                continuing.to(dtype=static_attention_mask.dtype),
            )

            if cuda_graph_pending:
                # CUDA graph capture requires eager warm-up on a side stream.
                # Keep it after the first yielded token so capture cannot inflate
                # time-to-first-speech-token.
                warmup_cache = StaticCache(
                    config=self.cfg,
                    max_cache_len=static_attention_mask.shape[1],
                )
                warmup_stream = torch.cuda.Stream()
                warmup_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(warmup_stream):
                    for _ in range(3):
                        self.tfmr(
                            inputs_embeds=static_embed,
                            attention_mask=static_attention_mask,
                            position_ids=static_position_ids,
                            past_key_values=warmup_cache,
                            cache_position=static_cache_position,
                            use_cache=True,
                        )
                torch.cuda.current_stream().wait_stream(warmup_stream)
                del warmup_cache, warmup_stream

                cuda_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(cuda_graph):
                    captured_outputs = self.tfmr(
                        inputs_embeds=static_embed,
                        attention_mask=static_attention_mask,
                        position_ids=static_position_ids,
                        past_key_values=past_key_values,
                        cache_position=static_cache_position,
                        use_cache=True,
                    )
                static_hidden_states = captured_outputs[0]
                cuda_graph_pending = False
                hidden_states = static_hidden_states
            else:
                cuda_graph.replay()
                hidden_states = static_hidden_states

            static_cache_position.add_(1)
            static_position_ids.add_(continuing[:, None])
            graph_physical_position += 1
            return (hidden_states,)

        while True:
            supplied = text_supplier(waiting)
            text_tokens = torch.atleast_2d(supplied[0]).to(
                dtype=torch.long,
                device=self.device,
            )
            text_attention_mask = torch.atleast_2d(supplied[1]).to(
                dtype=torch.long,
                device=self.device,
            )
            versions = list(supplied[2])
            input_done = list(supplied[3])
            cancelled = list(supplied[4])
            if text_tokens.shape != text_attention_mask.shape:
                raise ValueError("live text attention mask must match text tokens")
            if text_tokens.shape[0] != batch_size:
                raise ValueError("live text batch size cannot change")
            if not (len(versions) == len(input_done) == len(cancelled) == batch_size):
                raise ValueError("live text metadata must match batch size")

            changed = [
                version != current_versions[index]
                for index, version in enumerate(versions)
            ]
            needs_rebuild = outputs is None or any(changed)
            for index, did_change in enumerate(changed):
                if did_change:
                    waiting[index] = False
                    current_versions[index] = versions[index]
                if cancelled[index]:
                    finished[index] = True

            if needs_rebuild:
                clear_cuda_graph()
                max_speech_length = max(len(history) for history in histories)
                speech_tokens = torch.full(
                    (batch_size, max_speech_length),
                    start_token,
                    dtype=torch.long,
                    device=self.device,
                )
                speech_attention_mask = torch.zeros_like(speech_tokens)
                speech_lengths = []
                for index, history in enumerate(histories):
                    length = len(history)
                    speech_lengths.append(length)
                    speech_tokens[index, :length] = torch.tensor(
                        history,
                        dtype=torch.long,
                        device=self.device,
                    )
                    speech_attention_mask[index, :length] = 1

                embeds, len_cond = self.prepare_input_embeds(
                    t3_cond=t3_cond,
                    text_tokens=text_tokens,
                    speech_tokens=speech_tokens,
                    cfg_weight=0.0,
                )
                attention_mask = torch.cat(
                    [
                        torch.ones(
                            batch_size,
                            len_cond,
                            dtype=torch.long,
                            device=self.device,
                        ),
                        text_attention_mask,
                        speech_attention_mask,
                    ],
                    dim=1,
                )
                position_ids = attention_mask.cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 0)
                outputs = self.tfmr(
                    inputs_embeds=embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                last_positions = torch.tensor(
                    [
                        len_cond + text_tokens.shape[1] + length - 1
                        for length in speech_lengths
                    ],
                    dtype=torch.long,
                    device=self.device,
                )
                rebuilt = True

            active_indices = [
                index
                for index in range(batch_size)
                if not finished[index] and not waiting[index]
            ]
            if not active_indices:
                if all(finished):
                    break
                time.sleep(idle_sleep_seconds)
                continue

            if rebuilt:
                rows = torch.arange(batch_size, device=self.device)
                hidden = outputs[0][rows, last_positions]
            else:
                hidden = outputs[0][:, -1, :]
            speech_logits = self.speech_head(hidden)

            active_set = set(active_indices)
            sampled_tokens = []
            for index, history in enumerate(histories):
                if index not in active_set:
                    sampled_tokens.append(
                        torch.tensor(stop_token, device=self.device),
                    )
                    continue
                generated_ids = torch.as_tensor(
                    history,
                    dtype=torch.long,
                    device=self.device,
                ).unsqueeze(0)
                processed_logits = logits_processors(
                    generated_ids,
                    speech_logits[index : index + 1],
                )
                probs = F.softmax(processed_logits, dim=-1)
                sampled_tokens.append(
                    torch.multinomial(probs, num_samples=1)[0, 0],
                )
            next_tokens = torch.stack(sampled_tokens)
            next_token_values = next_tokens.detach().cpu().tolist()

            valid_values = [False] * batch_size
            for index in active_indices:
                token_value = next_token_values[index]
                if token_value == stop_token:
                    if input_done[index]:
                        finished[index] = True
                    else:
                        waiting[index] = True
                    continue

                histories[index].append(token_value)
                valid_values[index] = True
                if len(histories[index]) - 1 >= max_gen_len:
                    finished[index] = True

            valid = torch.tensor(
                valid_values,
                dtype=torch.bool,
                device=self.device,
            )
            finished_tensor = torch.tensor(
                finished,
                dtype=torch.bool,
                device=self.device,
            )
            yield next_tokens, valid, finished_tensor

            if all(finished):
                break

            continuing = valid & ~finished_tensor
            generated_token_count = max(len(history) for history in histories) - 1
            should_prepare_cuda_graph = (
                use_cuda_graph
                and self.device.type == "cuda"
                and batch_size <= 4
                and all(input_done)
                and cuda_graph is None
                and not cuda_graph_pending
                and generated_token_count >= cuda_graph_capture_after_tokens
            )
            if should_prepare_cuda_graph:
                outputs, attention_mask = prepare_cuda_graph_prefix()
                rebuilt = True
            elif cuda_graph_pending or cuda_graph is not None:
                outputs = run_cuda_graph_step(
                    next_tokens,
                    continuing,
                )
                rebuilt = False
            else:
                query_mask = continuing.to(dtype=torch.long)[:, None]
                attention_mask = torch.cat([attention_mask, query_mask], dim=1)
                query_position_ids = attention_mask.cumsum(-1)[:, -1:] - 1
                query_position_ids.masked_fill_(query_mask == 0, 0)
                speech_embed = self.speech_emb(next_tokens[:, None])
                outputs = self.tfmr(
                    inputs_embeds=speech_embed,
                    attention_mask=attention_mask,
                    position_ids=query_position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                rebuilt = False

    @torch.inference_mode()
    def inference_turbo(
        self,
        t3_cond,
        text_tokens,
        temperature=0.8,
        top_k=1000,
        top_p=0.95,
        repetition_penalty=1.2,
        max_gen_len=1000,
    ):
        generated_speech_tokens = list(
            self.iter_inference_turbo(
                t3_cond=t3_cond,
                text_tokens=text_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_gen_len=max_gen_len,
                show_progress=True,
            )
        )

        if not generated_speech_tokens:
            return torch.empty(
                text_tokens.size(0), 0, dtype=torch.long, device=self.device
            )

        all_tokens = torch.cat(generated_speech_tokens, dim=1)

        # Remove EOS token if present
        if all_tokens.size(1) > 0 and all_tokens[0, -1] == self.hp.stop_speech_token:
            all_tokens = all_tokens[:, :-1]

        return all_tokens

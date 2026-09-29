"""Native cached T3 sampling and experimental prefix/window audio decoding.

No text-generation or audio-generation monkeypatching, and no pretrained updates.
The audio encoder uses full attention: emitted samples are provisional and cannot
be revised when a later prefix changes their predictions.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import math
import time
from typing import Iterator

import torch
from transformers import (
    LogitsProcessorList, RepetitionPenaltyLogitsProcessor, TemperatureLogitsWarper,
    TopKLogitsWarper, TopPLogitsWarper,
)


@dataclass(frozen=True)
class StreamConfig:
    chunk_tokens: int = 5
    overlap_ms: float = 40.0
    max_gen_len: int = 500  # Native T3 predicts once, then up to this many times.
    temperature: float = 0.8
    top_k: int = 1000
    top_p: float = 0.95
    repetition_penalty: float = 1.2
    decoder_steps: int = 2
    decoder_seed: int = 18
    vocoder_seed: int | None = None  # Defaults to decoder_seed + 2, independently.
    left_context_tokens: int | None = 25  # None keeps all generated acoustic history.
    audio_mode: str = "conditioned_history"
    vocoder_lookahead_ms: float = 0.0

    def __post_init__(self):
        if self.chunk_tokens < 1 or self.max_gen_len < 0 or self.decoder_steps < 1:
            raise ValueError("Chunk size/decoder steps must be positive; cap must be nonnegative")
        if self.left_context_tokens is not None and self.left_context_tokens < 0:
            raise ValueError("Left context must be nonnegative or None")
        if self.audio_mode not in ("legacy", "committed_mels", "conditioned_history"):
            raise ValueError("Unknown audio mode")
        if not math.isfinite(self.vocoder_lookahead_ms) or self.vocoder_lookahead_ms < 0:
            raise ValueError("Vocoder lookahead must be finite and nonnegative")
        if not math.isfinite(self.overlap_ms) or self.overlap_ms < 0:
            raise ValueError("Overlap must be finite and nonnegative")
        if not math.isfinite(self.temperature) or self.temperature < 0 or self.top_k < 0:
            raise ValueError("Temperature and top-k must be finite/nonnegative")
        if not 0 < self.top_p <= 1 or not math.isfinite(self.repetition_penalty) or self.repetition_penalty <= 0:
            raise ValueError("Invalid sampling settings")


@dataclass(frozen=True)
class SpeechToken:
    token_id: int
    final: bool
    stop_reason: str | None


@dataclass(frozen=True)
class AudioChunk:
    samples: torch.Tensor  # Detached CPU float32 [samples], ready for playback.
    sample_rate: int
    start_sample: int
    final: bool
    speech_tokens: int
    elapsed_seconds: float
    boundary_revision_rms: float


def _processors(config):
    result = LogitsProcessorList()
    if config.temperature > 0 and config.temperature != 1:
        result.append(TemperatureLogitsWarper(config.temperature))
    if config.top_k > 0:
        result.append(TopKLogitsWarper(config.top_k))
    if config.top_p < 1:
        result.append(TopPLogitsWarper(config.top_p))
    if config.repetition_penalty != 1:
        result.append(RepetitionPenaltyLogitsProcessor(config.repetition_penalty))
    return result


@torch.inference_mode()
def stream_speech_tokens(t3, condition, text_tokens, config=StreamConfig(), *,
                         text_vectors=None) -> Iterator[SpeechToken]:
    """Preserve native Turbo/Nano sampling, yielding before the next model step.

Full text or raw text vectors precede speech BOS in the cache. No streaming input.
The only intentional native-loop correction is stopping on an initial EOS too.
"""
    if text_tokens.ndim != 2 or text_tokens.shape[0] != 1 or text_tokens.shape[1] == 0:
        raise ValueError("Expected nonempty text IDs [1, length]")
    if t3.training:
        raise ValueError("T3 must be in evaluation mode")
    if not t3.is_gpt:
        raise ValueError("This path requires the native GPT2 Turbo/Nano backbone")
    bos = torch.full_like(text_tokens[:, :1], t3.hp.start_speech_token)
    if text_vectors is None:
        embeds, _ = t3.prepare_input_embeds(
            t3_cond=condition, text_tokens=text_tokens, speech_tokens=bos, cfg_weight=0.0)
    else:
        if (text_vectors.ndim != 3 or text_vectors.shape[0] != 1
                or text_vectors.shape[1] == 0 or text_vectors.shape[2] != t3.dim
                or not bool(torch.isfinite(text_vectors).all())):
            raise ValueError(f"Expected finite text vectors [1, length, {t3.dim}]")
        reference = t3.text_emb.weight
        if text_vectors.device != reference.device or text_vectors.dtype != reference.dtype:
            raise ValueError("Text vector device and dtype must match the model")
        cond = t3.prepare_conditioning(condition)
        embeds = torch.cat((cond, text_vectors, t3.speech_emb(bos)), dim=1)
    if embeds.shape[1] + config.max_gen_len > t3.tfmr.config.max_position_embeddings:
        raise ValueError("Input plus generation cap exceeds the GPT2 position table")
    processors = _processors(config)
    output = t3.tfmr(inputs_embeds=embeds, use_cache=True)
    past = output.past_key_values
    hidden = output[0][:, -1:]
    generated = []
    for step in range(config.max_gen_len + 1):
        history = bos if not generated else torch.cat(generated, dim=1)
        logits = processors(history, t3.speech_head(hidden)[:, -1, :])
        if not bool(torch.isfinite(logits).any()) or bool(torch.isnan(logits).any()):
            raise RuntimeError("Speech sampling has no valid logits")
        token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
        generated.append(token)
        token_id = int(token.item())
        reason = "eos" if token_id == t3.hp.stop_speech_token else (
            "cap" if step == config.max_gen_len else None)
        yield SpeechToken(token_id, reason is not None, reason)
        if reason is not None:
            return
        output = t3.tfmr(inputs_embeds=t3.speech_emb(token),
                         past_key_values=past, use_cache=True)
        past, hidden = output.past_key_values, output[0]


class LookaheadMaskDecoder(torch.nn.Module):
    """Align the pinned S3Gen mask after its partial-path lookahead crop.

    Upstream flow.inference crops h but calculates mask from uncropped h_masks.
    This frozen boundary wrapper changes no decoder weights or full-path behavior.
    """
    def __init__(self, decoder, lookahead_frames):
        super().__init__()
        self.decoder = decoder
        self.lookahead_frames = lookahead_frames

    def forward(self, *, mu, mask, **kwargs):
        excess = mask.shape[-1] - mu.shape[-1]
        if excess not in (0, self.lookahead_frames):
            raise ValueError("Unexpected flow mask/frame mismatch")
        return self.decoder(mu=mu, mask=mask[..., :mu.shape[-1]], **kwargs)


@contextmanager
def isolated_decoder_rng(device, seed):
    """Preserve speech sampling RNG even when the decoder runs on Metal."""
    cuda_devices = [device.index or 0] if device.type == "cuda" else []
    mps_state = torch.mps.get_rng_state() if device.type == "mps" else None
    with torch.random.fork_rng(devices=cuda_devices):
        try:
            torch.random.default_generator.manual_seed(seed)
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            elif device.type == "mps":
                torch.mps.manual_seed(seed)
            yield
        finally:
            if mps_state is not None:
                torch.mps.set_rng_state(mps_state)


class PrefixAudioDecoder:
    """Partial S3Gen calls with stable generated-prefix noise and isolated RNG.

Prefixes are recomputed, including the vocoder. This deliberately favors a small
inspectable prototype over a new cache implementation with unverified semantics.
"""
    def __init__(self, model, config, *, stable_prompt_noise=False):
        self.model, self.config = model, config
        self.device = model.t3.text_emb.weight.device
        self.noise = None
        self.prompt_noise = None
        self.stable_prompt_noise = stable_prompt_noise
        self.lookahead = int(model.s3gen.flow.pre_lookahead_len)
        self.ratio = int(model.s3gen.flow.token_mel_ratio)
        self.silence_id = 4299  # S3GEN_SIL, verified by the pinned loader.
        self.valid_vocab = int(model.s3gen.flow.vocab_size)
        if model.sr % 25:
            raise ValueError("Expected sample rate divisible by the native 25 Hz token rate")
        self.samples_per_token = model.sr // 25
        self.records = []

    def _clock(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()
        return time.perf_counter()

    @torch.inference_mode()
    def decode(self, tokens, *, final, start_token=0, mel_only=False,
               reference_conditions=None, noise_start_token=None):
        if start_token < 0:
            raise ValueError("Window token offset must be nonnegative")
        if not tokens:
            return torch.empty(0)
        content = list(tokens) + ([self.silence_id] * 3 if final else [])
        if not final and len(content) <= self.lookahead:
            return torch.empty(0)
        steps = len(content) - (0 if final else self.lookahead)
        reference = self.model.t3.text_emb.weight
        # Allocate once with a private CPU generator; appending prefixes retains noise.
        if self.noise is None:
            generator = torch.Generator().manual_seed(self.config.decoder_seed)
            capacity = self.config.max_gen_len + 4
            self.noise = torch.randn(1, 80, capacity * self.ratio, generator=generator).to(
                device=self.device, dtype=self.model.s3gen.dtype)
            if self.stable_prompt_noise:
                prompt_frames = self.model.conds.gen["prompt_feat"].shape[1]
                prompt_generator = torch.Generator().manual_seed(self.config.decoder_seed + 1)
                self.prompt_noise = torch.randn(
                    1, 80, prompt_frames, generator=prompt_generator).to(
                        device=self.device, dtype=self.model.s3gen.dtype)
        noise_start_token = start_token if noise_start_token is None else noise_start_token
        if not 0 <= noise_start_token <= start_token:
            raise ValueError("Noise history must precede the current token position")
        noise_start = noise_start_token * self.ratio
        noise_end = (start_token + steps) * self.ratio
        if noise_end > self.noise.shape[-1]:
            raise ValueError("Speech prefix exceeds configured decoder capacity")
        ids = torch.tensor([content], device=reference.device, dtype=torch.long)
        noise = self.noise[..., noise_start:noise_end]
        if self.prompt_noise is not None:
            # Native CFM accepts noise for all mu frames (prompt_len=0), so the
            # voice-prompt noise can be fixed across changing window lengths too.
            noise = torch.cat((self.prompt_noise, noise), dim=-1)
        # S3Gen also samples reference noise; it must not consume the T3 RNG stream.
        started = self._clock()
        with isolated_decoder_rng(self.device, self.config.decoder_seed):
            mels = self.model.s3gen.forward(
                speech_tokens=ids, ref_wav=None, ref_sr=None,
                ref_dict=dict(self.model.conds.gen if reference_conditions is None
                              else reference_conditions), finalize=final,
                n_cfm_timesteps=self.config.decoder_steps,
                noised_mels=noise, skip_vocoder=True)
        if mel_only:
            return mels
        # CFM draws length-dependent noise even when we overwrite it; sharing its
        # RNG with HiFTGAN changes harmonic starting phases on every prefix.
        vocoder_seed = (self.config.decoder_seed + 2 if self.config.vocoder_seed is None
                        else self.config.vocoder_seed)
        with isolated_decoder_rng(self.device, vocoder_seed):
            wave, _ = self.model.s3gen.hift_inference(mels)
        fade = getattr(self.model.s3gen, "trim_fade", None)
        if fade is not None and not self.model.s3gen.training:
            wave[:, :len(fade)] *= fade
        wave = wave.reshape(-1).detach().float().cpu()
        if not bool(torch.isfinite(wave).all()):
            raise RuntimeError("Decoder produced nonfinite audio")
        if len(wave) != steps * self.samples_per_token:
            raise RuntimeError("Unexpected native token-to-waveform length")
        self.records.append({"start_token": start_token, "input_tokens": len(content),
                             "output_samples": len(wave), "final": final,
                             "vocoder_seed": vocoder_seed,
                             "decode_seconds": self._clock() - started})
        return wave


class AudioStitcher:
    """Emit contiguous slices and blend a withheld tail with its new prediction."""
    def __init__(self, overlap_samples, lookahead_samples=0):
        self.overlap = overlap_samples
        self.lookahead = lookahead_samples
        self.emitted = 0
        self.pending = torch.empty(0)
        self.finished = False

    def push(self, waveform, *, final, start_sample=0):
        if self.finished:
            raise RuntimeError("Audio stream is already finalized")
        if waveform.ndim != 1 or not bool(torch.isfinite(waveform).all()):
            raise ValueError("Expected finite mono prefix waveform")
        if start_sample < 0 or start_sample > self.emitted:
            raise RuntimeError("Decoder window leaves a gap before the committed audio boundary")
        end = start_sample + (len(waveform) if final else max(
            0, len(waveform) - self.lookahead - self.overlap))
        if end < self.emitted:
            raise RuntimeError("Decoder prefix shrank below the committed audio boundary")
        start = self.emitted
        samples = waveform[start - start_sample:end - start_sample].clone()
        n = min(len(self.pending), len(samples), self.overlap)
        revision = 0.0
        if n:
            revision = float((self.pending[:n] - samples[:n]).square().mean().sqrt())
            ramp = torch.linspace(0, 1, n + 2, dtype=samples.dtype)[1:-1]
            samples[:n] = self.pending[:n] * (1 - ramp) + samples[:n] * ramp
        self.emitted = end
        self.pending = waveform[end - start_sample:].clone() if not final else torch.empty(0)
        self.finished = final
        return start, samples, revision


class CommittedMelAudioDecoder(PrefixAudioDecoder):
    """Freeze generated mel frames and preserve stable HiFT excitation.

The expensive token-to-mel call may use a fixed token window. HiFT still decodes
the accumulated mel sequence; bounding that cheaper stage is a separate change.
"""
    def __init__(self, model, config, *, stable_prompt_noise=True):
        super().__init__(model, config, stable_prompt_noise=stable_prompt_noise)
        self.mels = None
        self.source_cache = None
        # The pinned five kernel-3 F0 convolutions need five future mel frames.
        self.source_guard = 5 * self.samples_per_token // self.ratio

    @torch.inference_mode()
    def decode(self, tokens, *, final, start_token=0):
        started = self._clock()
        mels = super().decode(tokens, final=final, start_token=start_token, mel_only=True)
        if not mels.numel():
            return torch.empty(0)
        if mels.ndim != 3 or mels.shape[:2] != (1, 80):
            raise RuntimeError("Expected native mel frames [1,80,frames]")
        committed = 0 if self.mels is None else self.mels.shape[-1]
        local_offset = committed - start_token * self.ratio
        if local_offset < 0 or local_offset > mels.shape[-1]:
            raise RuntimeError("Mel window fails to cover the committed boundary")
        new = mels[..., local_offset:].clone()
        self.mels = new if self.mels is None else torch.cat((self.mels, new), dim=-1)
        seed = self.config.decoder_seed + 2 if self.config.vocoder_seed is None else self.config.vocoder_seed
        with isolated_decoder_rng(self.device, seed):
            wave, source = self.model.s3gen.hift_inference(self.mels, self.source_cache)
        stable_end = source.shape[-1] if final else max(0, source.shape[-1] - self.source_guard)
        self.source_cache = source[..., :stable_end].clone()
        fade = getattr(self.model.s3gen, "trim_fade", None)
        if fade is not None and not self.model.s3gen.training:
            wave[:, :len(fade)] *= fade
        wave = wave.reshape(-1).detach().float().cpu()
        expected = self.mels.shape[-1] * self.samples_per_token // self.ratio
        if len(wave) != expected or not bool(torch.isfinite(wave).all()):
            raise RuntimeError("Invalid continuous waveform")
        self.records.append({"start_token": start_token,
                             "input_tokens": len(tokens) + (3 if final else 0),
                             "committed_mel_frames": self.mels.shape[-1],
                             "new_mel_frames": new.shape[-1], "source_cache_samples": stable_end,
                             "output_samples": len(wave), "final": final, "vocoder_seed": seed,
                             "decode_seconds": self._clock() - started})
        return wave


class AnchoredMelAudioDecoder(PrefixAudioDecoder):
    def __init__(self, model, config, *, stable_prompt_noise=True):
        super().__init__(model, config, stable_prompt_noise=stable_prompt_noise)
        self.mels = None
        prompt = model.conds.gen
        self.original_prompt_frames = prompt['prompt_feat'].shape[1]
        if self.original_prompt_frames != prompt['prompt_token'].shape[-1] * self.ratio:
            raise ValueError('Reference tokens and acoustic frames must align')

    @torch.inference_mode()
    def decode(self, tokens, *, final, start_token=0):
        started = self._clock()
        committed = 0 if self.mels is None else self.mels.shape[-1] // self.ratio
        offset = committed - start_token
        if offset < 0 or offset > len(tokens):
            raise RuntimeError('Token window does not cover the acoustic continuation')
        history_start = start_token
        if self.config.left_context_tokens is not None:
            history_start = max(history_start, committed - self.config.left_context_tokens)
        history = list(tokens[history_start-start_token:offset])
        new_tokens = list(tokens[offset:])
        ref = dict(self.model.conds.gen)
        if history:
            ids = torch.tensor([history], device=ref['prompt_token'].device,
                               dtype=ref['prompt_token'].dtype)
            frames = self.mels[..., history_start*self.ratio:committed*self.ratio]
            if frames.shape[-1] != len(history) * self.ratio:
                raise RuntimeError('Cached acoustic history does not match its tokens')
            ref['prompt_token'] = torch.cat((ref['prompt_token'], ids), dim=-1)
            ref['prompt_token_len'] = ref['prompt_token_len'] + len(history)
            ref['prompt_feat'] = torch.cat((ref['prompt_feat'], frames.transpose(1, 2)), dim=1)
            if ref.get('prompt_feat_len') is not None:
                ref['prompt_feat_len'] = ref['prompt_feat_len'] + frames.shape[-1]
        fresh = super().decode(new_tokens, final=final, start_token=committed,
                               mel_only=True, reference_conditions=ref,
                               noise_start_token=history_start)
        if not fresh.numel():
            return torch.empty(0)
        if fresh.ndim != 3 or fresh.shape[:2] != (1, 80):
            raise RuntimeError('Expected new acoustic frames [1,80,frames]')
        self.mels = fresh.clone() if self.mels is None else torch.cat((self.mels, fresh), dim=-1)
        seed = self.config.decoder_seed + 2 if self.config.vocoder_seed is None else self.config.vocoder_seed
        with isolated_decoder_rng(self.device, seed):
            waveform, _ = self.model.s3gen.hift_inference(self.mels)
        fade = getattr(self.model.s3gen, 'trim_fade', None)
        if fade is not None and not self.model.s3gen.training:
            waveform[:, :len(fade)] *= fade
        waveform = waveform.reshape(-1).detach().float().cpu()
        expected = self.mels.shape[-1] * self.samples_per_token // self.ratio
        if len(waveform) != expected or not bool(torch.isfinite(waveform).all()):
            raise RuntimeError('Invalid conditioned streaming waveform')
        self.records.append(dict(start_token=history_start, new_token_offset=committed,
                                 input_tokens=len(new_tokens) + (3 if final else 0),
                                 conditioning_history_tokens=len(history),
                                 conditioned_acoustic_frames=len(history)*self.ratio,
                                 committed_mel_frames=self.mels.shape[-1],
                                 new_mel_frames=fresh.shape[-1], output_samples=len(waveform),
                                 excitation_cache=False, final=final, vocoder_seed=seed,
                                 decode_seconds=self._clock()-started))
        return waveform


def make_audio_decoder(model, config, *, stable_prompt_noise=False):
    if config.audio_mode == "conditioned_history":
        return AnchoredMelAudioDecoder(model, config, stable_prompt_noise=True)
    if config.audio_mode == "committed_mels":
        return CommittedMelAudioDecoder(model, config, stable_prompt_noise=True)
    return PrefixAudioDecoder(model, config, stable_prompt_noise=stable_prompt_noise)


def make_audio_stitcher(model, config):
    delay = config.vocoder_lookahead_ms if config.audio_mode != "legacy" else 0
    return AudioStitcher(round(config.overlap_ms * model.sr / 1000),
                         round(delay * model.sr / 1000))


class NanoStreamer:
    """Wrap a frozen native Nano model; input is complete text or raw vectors."""
    def __init__(self, model, config=StreamConfig(), *, normalize_text=None):
        if model.t3.dim != 768 or model.t3.hp.llama_config_name != "GPT2_small":
            raise ValueError("Expected Chatterbox Nano with its 768-wide GPT2-small backbone")
        if model.t3.text_emb.weight.device.type not in ("cpu", "cuda", "mps"):
            raise ValueError("Audio streaming supports CPU, CUDA or MPS")
        self.model, self.config = model, config
        self.normalize_text = normalize_text or (lambda text: text)
        flow = model.s3gen.flow
        if hasattr(flow, "decoder") and not isinstance(flow.decoder, LookaheadMaskDecoder):
            flow.decoder = LookaheadMaskDecoder(
                flow.decoder, flow.pre_lookahead_len * flow.token_mel_ratio)
        for module in (model.t3, model.s3gen, getattr(model, "ve", None)):
            if module is not None:
                module.eval().requires_grad_(False)
        self.metrics = {}

    def _clock(self):
        device = self.model.t3.text_emb.weight.device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()
            if self.metrics:
                for key, value in (("max_observed_mps_allocated_bytes", torch.mps.current_allocated_memory()),
                                   ("max_observed_mps_driver_bytes", torch.mps.driver_allocated_memory())):
                    self.metrics[key] = max(self.metrics.get(key, 0), value)
        return time.perf_counter()

    def stream(self, text, *, audio_prompt_path=None, text_vectors=None):
        """Yield CPU audio chunks as tokens arrive, including a terminal event.

        Metrics exclude model load, text preprocessing and reference preparation.
        This instance supports a single active stream and no concurrent consumers.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Supply a nonempty full text input")
        if audio_prompt_path:
            self.model.prepare_conditionals(audio_prompt_path)
        if self.model.conds is None:
            raise ValueError("A native voice condition or reference audio is required")
        text = self.normalize_text(text)
        device = self.model.t3.text_emb.weight.device
        ids = self.model.tokenizer(text, return_tensors="pt").input_ids.to(device)
        decoder = make_audio_decoder(self.model, self.config)
        stitcher = make_audio_stitcher(self.model, self.config)
        self.metrics = {"speech_token_ids": [], "stop_reason": None,
                        "first_speech_token_seconds": None, "first_audio_chunk_seconds": None,
                        "boundary_revision_rms": [], "chunk_count": 0,
                        "input_tokens": ids.shape[1], "sample_rate": self.model.sr,
                        "watermarked": False, "input_streaming": False,
                        "history_conditioning": self.config.audio_mode == "conditioned_history"}
        started = self._clock()
        valid = []
        since_decode = 0
        for event in stream_speech_tokens(self.model.t3, self.model.conds.t3, ids,
                                         self.config, text_vectors=text_vectors):
            if self.metrics["first_speech_token_seconds"] is None:
                self.metrics["first_speech_token_seconds"] = self._clock() - started
            self.metrics["speech_token_ids"].append(event.token_id)
            if (0 <= event.token_id < decoder.valid_vocab
                    and event.token_id != self.model.t3.hp.stop_speech_token):
                valid.append(event.token_id)
                since_decode += 1
            if event.final or (since_decode >= self.config.chunk_tokens and len(valid) > decoder.lookahead):
                window_start = 0 if self.config.left_context_tokens is None else max(
                    0, stitcher.emitted // decoder.samples_per_token - self.config.left_context_tokens)
                waveform = decoder.decode(valid[window_start:], final=event.final,
                                          start_token=window_start)
                start, samples, revision = stitcher.push(
                    waveform, final=event.final,
                    start_sample=(0 if self.config.audio_mode != "legacy" else
                                  window_start * decoder.samples_per_token))
                since_decode = 0
                elapsed = self._clock() - started
                if len(samples) and self.metrics["first_audio_chunk_seconds"] is None:
                    self.metrics["first_audio_chunk_seconds"] = elapsed
                self.metrics["boundary_revision_rms"].append(revision)
                if event.final:
                    self.metrics.update(stop_reason=event.stop_reason, total_seconds=elapsed,
                                        audio_samples=stitcher.emitted,
                                        audio_seconds=stitcher.emitted / self.model.sr,
                                        decoder_calls=decoder.records,
                                        max_decoder_input_tokens=max(
                                            (r["input_tokens"] + r.get("conditioning_history_tokens", 0)
                                             for r in decoder.records), default=0),
                                        max_conditioning_history_tokens=max(
                                            (r.get("conditioning_history_tokens", 0)
                                             for r in decoder.records), default=0))
                if len(samples) or event.final:
                    self.metrics["chunk_count"] += 1
                    yield AudioChunk(samples, self.model.sr, start, event.final,
                                     len(valid), elapsed, revision)

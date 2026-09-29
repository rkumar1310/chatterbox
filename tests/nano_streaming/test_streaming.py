from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import GPT2Config, GPT2Model

from chatterbox.nano_streaming.streaming import (
    AudioStitcher, CommittedMelAudioDecoder, LookaheadMaskDecoder, NanoStreamer, PrefixAudioDecoder, StreamConfig,
    isolated_decoder_rng, stream_speech_tokens,
)


class TinyT3(nn.Module):
    """Actual two-layer GPT2 cache, with native text/condition/speech ordering."""
    is_gpt = True

    def __init__(self):
        super().__init__()
        self.dim = 16
        self.hp = SimpleNamespace(start_speech_token=10, stop_speech_token=11,
                                  llama_config_name="GPT2_small")
        self.text_emb = nn.Embedding(16, self.dim)
        self.speech_emb = nn.Embedding(12, self.dim)
        self.speech_head = nn.Linear(self.dim, 12)
        self.tfmr = GPT2Model(GPT2Config(vocab_size=16, n_embd=self.dim, n_layer=2,
                                        n_head=2, n_positions=128, attn_pdrop=0,
                                        embd_pdrop=0, resid_pdrop=0))
        # Keep random ordinary-token sampling, preventing accidental EOS in parity checks.
        self.speech_head.bias.data[10:] = -100
        self.eval().requires_grad_(False)

    def prepare_conditioning(self, condition):
        return torch.zeros(1, 2, self.text_emb.weight.shape[-1])

    def prepare_input_embeds(self, *, t3_cond, text_tokens, speech_tokens, cfg_weight):
        return torch.cat((self.prepare_conditioning(t3_cond), self.text_emb(text_tokens),
                          self.speech_emb(speech_tokens)), dim=1), 2


def tiny():
    torch.manual_seed(4)
    return TinyT3()


def full_prefix_tokens(model, ids, cap):
    """Independent full recomputation control for the cached generation path."""
    bos = torch.tensor([[model.hp.start_speech_token]])
    prefix, _ = model.prepare_input_embeds(t3_cond=None, text_tokens=ids,
                                          speech_tokens=bos, cfg_weight=0)
    result = []
    with torch.inference_mode():
        for _ in range(cap + 1):
            hidden = model.tfmr(inputs_embeds=prefix, use_cache=False)[0][:, -1:]
            token = torch.multinomial(model.speech_head(hidden)[:, -1].softmax(-1), 1)
            result.append(int(token.item()))
            if result[-1] == model.hp.stop_speech_token:
                break
            prefix = torch.cat((prefix, model.speech_emb(token)), dim=1)
    return result


def test_real_gpt2_cache_matches_full_prefix_and_yields_before_next_step():
    model = tiny()
    ids = torch.tensor([[2, 3, 4]])
    config = StreamConfig(max_gen_len=6, temperature=1, top_k=0, top_p=1, repetition_penalty=1)
    calls = []
    hook = model.tfmr.register_forward_hook(lambda *args: calls.append(1))
    before = {n: p.clone() for n, p in model.named_parameters()}
    torch.manual_seed(9)
    generator = stream_speech_tokens(model, None, ids, config)
    first = next(generator)
    assert len(calls) == 1  # No precomputed later predictions before the first yield.
    assert not first.final
    events = [first, *generator]
    hook.remove()
    assert events[-1].stop_reason == "cap" and len(events) == 7
    torch.manual_seed(9)
    assert [e.token_id for e in events] == full_prefix_tokens(model, ids, 6)
    assert all(p.grad is None and torch.equal(before[n], p) for n, p in model.named_parameters())


def test_initial_eos_is_honored_without_another_forward():
    model = tiny()
    model.speech_head.weight.data.zero_()
    model.speech_head.bias.data.fill_(-100)
    model.speech_head.bias.data[11] = 100
    events = list(stream_speech_tokens(model, None, torch.tensor([[2]]), StreamConfig(max_gen_len=3)))
    assert len(events) == 1 and events[0].token_id == 11
    assert events[0].final and events[0].stop_reason == "eos"


def test_pinned_native_t3_sampler_parity():
    native = pytest.importorskip("chatterbox.models.t3.t3")
    model = tiny()
    ids = torch.tensor([[2, 3]])
    config = StreamConfig(max_gen_len=7)
    torch.manual_seed(17)
    expected = native.T3.inference_turbo(
        model, None, ids, temperature=config.temperature, top_k=config.top_k,
        top_p=config.top_p, repetition_penalty=config.repetition_penalty,
        max_gen_len=config.max_gen_len)
    torch.manual_seed(17)
    actual = list(stream_speech_tokens(model, None, ids, config))
    assert expected[0].tolist() == [event.token_id for event in actual]


def test_raw_embedding_injection_parity_and_wrong_width_rejection():
    model = tiny()
    ids = torch.tensor([[2, 3]])
    config = StreamConfig(max_gen_len=3)
    torch.manual_seed(17)
    ordinary = list(stream_speech_tokens(model, None, ids, config))
    torch.manual_seed(17)
    injected = list(stream_speech_tokens(model, None, ids, config,
                                        text_vectors=model.text_emb(ids)))
    assert ordinary == injected
    with pytest.raises(ValueError, match="text vectors"):
        list(stream_speech_tokens(model, None, ids, config, text_vectors=torch.zeros(1, 2, 1024)))
    with pytest.raises(ValueError, match="position table"):
        list(stream_speech_tokens(model, None, ids, StreamConfig(max_gen_len=200)))


def test_stitching_preserves_sample_positions_and_flushes_tail_once():
    stitcher = AudioStitcher(4)
    full = torch.arange(31).float()
    pieces = []
    expected = 0
    for count, final in [(12, False), (21, False), (31, True)]:
        start, samples, rms = stitcher.push(full[:count], final=final)
        assert start == expected and rms == 0
        expected += len(samples)
        pieces.append(samples)
    assert torch.allclose(torch.cat(pieces), full)
    with pytest.raises(RuntimeError, match="already finalized"):
        stitcher.push(full, final=True)


def test_changed_boundary_blends_with_pending_audio_without_duplication():
    stitcher = AudioStitcher(4)
    _, first, _ = stitcher.push(torch.zeros(10), final=False)
    start, last, revision = stitcher.push(torch.ones(15), final=True)
    assert start == 6 and len(first) + len(last) == 15
    assert revision == 1
    assert torch.all(last[:4] > 0) and torch.all(last[:4] < 1)
    assert torch.all(last[4:] == 1)
    with pytest.raises(RuntimeError, match="shrank"):
        other = AudioStitcher(2)
        other.push(torch.ones(12), final=False)
        other.push(torch.ones(8), final=True)


class FakeDecoder(nn.Module):
    dtype = torch.float32

    def __init__(self):
        super().__init__()
        self.flow = SimpleNamespace(pre_lookahead_len=3, token_mel_ratio=2, vocab_size=6561)
        self.calls = []

    def forward(self, *, speech_tokens, finalize, noised_mels, **kwargs):
        torch.randn(13)  # Deliberately try to perturb the token generator's RNG.
        usable = speech_tokens.shape[1] - (0 if finalize else 3)
        prompt_frames = kwargs["ref_dict"].get("prompt_feat", torch.empty(1, 0, 80)).shape[1]
        assert noised_mels.shape[-1] == prompt_frames + usable * 2
        self.calls.append((speech_tokens.clone(), finalize, noised_mels.clone()))
        values = speech_tokens[:, :usable].float().repeat_interleave(2, dim=1)
        return values[:, None, :].expand(-1, 80, -1)

    def hift_inference(self, mels, cache_source=None):
        wave = mels[:, 0, :].repeat_interleave(20, dim=-1)
        return wave, wave[:, None, :].clone()


def fake_model():
    model = SimpleNamespace(t3=tiny(), s3gen=FakeDecoder(), sr=1000,
                            conds=SimpleNamespace(t3=None, gen={"prompt_feat": torch.zeros(1, 0, 80),
                                "prompt_token": torch.empty(1, 0, dtype=torch.long),
                                "prompt_token_len": torch.tensor([0]), "prompt_feat_len": None}))
    # NanoStreamer validates model identity, while the actual test backbone stays tiny.
    # Its 768 label is used only for that check; text vectors are tested separately.
    model.t3.dim = 768
    model.tokenizer = lambda *args, **kwargs: SimpleNamespace(input_ids=torch.tensor([[2, 3]]))
    return model


def test_partial_decoder_preserves_rng_noise_prefix_lookahead_and_final_silence():
    model = fake_model()
    decoder = PrefixAudioDecoder(model, StreamConfig(max_gen_len=12))
    torch.manual_seed(17)
    initial = torch.random.get_rng_state().clone()
    assert len(decoder.decode([1, 2, 3], final=False)) == 0
    decoder.decode([1, 2, 3, 4, 5], final=False)
    decoder.decode([1, 2, 3, 4, 5, 6], final=True)
    assert torch.equal(torch.random.get_rng_state(), initial)
    first, last = model.s3gen.calls
    assert not first[1] and last[1]
    assert torch.equal(first[2], last[2][..., :first[2].shape[-1]])
    assert last[0][0, -3:].tolist() == [4299, 4299, 4299]


def test_native_flow_lookahead_mask_correction_and_full_path_parity():
    native = pytest.importorskip("chatterbox.models.s3gen.flow")

    class Encoder(nn.Module):
        def forward(self, values, lengths):
            h = values.repeat_interleave(2, dim=1)
            return h, torch.ones(1, 1, h.shape[1], dtype=torch.bool)

        def output_size(self):
            return 8

    class Decoder(nn.Module):
        def forward(self, *, mu, mask, **kwargs):
            assert mask.shape[-1] == mu.shape[-1]
            return mu * mask, None

    flow = native.CausalMaskedDiffWithXvec(input_size=8, output_size=4,
                                         spk_embed_dim=4, vocab_size=20,
                                         encoder=Encoder(), decoder=Decoder())
    inputs = dict(token=torch.tensor([[1, 2, 3, 4, 5]]), token_len=torch.tensor([5]),
                  prompt_token=torch.tensor([[1, 2]]), prompt_token_len=torch.tensor([2]),
                  prompt_feat=torch.zeros(1, 4, 4), prompt_feat_len=None,
                  embedding=torch.ones(1, 4))
    original, _ = flow.inference(**inputs, finalize=True)
    with pytest.raises(AssertionError):
        flow.inference(**inputs, finalize=False)
    flow.decoder = LookaheadMaskDecoder(flow.decoder, 6)
    unchanged, _ = flow.inference(**inputs, finalize=True)
    partial, _ = flow.inference(**inputs, finalize=False)
    assert torch.equal(original, unchanged)
    assert partial.shape == (1, 4, 4)
    with pytest.raises(ValueError, match="mask/frame"):
        flow.decoder(mu=torch.zeros(1, 4, 8), mask=torch.ones(1, 1, 9))


def test_end_to_end_stream_produces_audio_before_completion_and_matches_tokens():
    model = fake_model()
    config = StreamConfig(chunk_tokens=4, overlap_ms=2, max_gen_len=11, audio_mode="legacy")
    streamer = NanoStreamer(model, config)
    torch.manual_seed(17)
    generator = streamer.stream("Hello")
    first = next(generator)
    assert not first.final and len(first.samples) > 0
    assert first.speech_tokens == 4 and len(model.s3gen.calls) == 1
    chunks = [first, *generator]
    assert chunks[-1].final and streamer.metrics["stop_reason"] == "cap"
    assert sum(len(c.samples) for c in chunks) == streamer.metrics["audio_samples"]
    assert all(c.start_sample == sum(len(x.samples) for x in chunks[:i])
               for i, c in enumerate(chunks))
    torch.manual_seed(17)
    control = list(stream_speech_tokens(model.t3, None, torch.tensor([[2, 3]]), config))
    assert streamer.metrics["speech_token_ids"] == [e.token_id for e in control]
    assert all(not p.requires_grad for p in model.t3.parameters())


def test_empty_input_and_empty_eos_are_explicit():
    model = fake_model()
    streamer = NanoStreamer(model, StreamConfig(max_gen_len=3))
    with pytest.raises(ValueError, match="nonempty"):
        list(streamer.stream(" "))
    model.t3.speech_head.weight.data.zero_()
    model.t3.speech_head.bias.data.fill_(-100)
    model.t3.speech_head.bias.data[11] = 100
    chunks = list(streamer.stream("Hello"))
    assert len(chunks) == 1 and chunks[0].final and len(chunks[0].samples) == 0
    assert streamer.metrics["stop_reason"] == "eos"
    assert streamer.metrics["first_audio_chunk_seconds"] is None


def test_window_stitching_uses_absolute_offsets_and_rejects_gaps():
    stitcher = AudioStitcher(4)
    full = torch.arange(80).float()
    pieces = []
    for left, right, final in [(0, 30, False), (20, 50, False), (40, 80, True)]:
        start, samples, revision = stitcher.push(full[left:right], final=final,
                                                start_sample=left)
        assert start == sum(len(x) for x in pieces) and revision == 0
        pieces.append(samples)
    assert torch.allclose(torch.cat(pieces), full)
    with pytest.raises(RuntimeError, match="gap"):
        AudioStitcher(4).push(full, final=False, start_sample=1)


def test_evicted_window_reuses_noise_at_global_token_positions():
    model = fake_model()
    decoder = PrefixAudioDecoder(model, StreamConfig(max_gen_len=30))
    decoder.decode(list(range(1, 11)), final=False)
    decoder.decode(list(range(5, 16)), final=False, start_token=4)
    first, second = model.s3gen.calls
    assert torch.equal(first[2][..., 8:14], second[2][..., :6])
    assert decoder.records[-1]["start_token"] == 4


def test_quality_comparison_keeps_prompt_and_generated_noise_fixed():
    model = fake_model()
    model.conds.gen["prompt_feat"] = torch.zeros(1, 6, 80)
    decoder = PrefixAudioDecoder(model, StreamConfig(max_gen_len=30), stable_prompt_noise=True)
    decoder.decode(list(range(1, 11)), final=False)
    decoder.decode(list(range(5, 16)), final=False, start_token=4)
    first, second = model.s3gen.calls
    assert torch.equal(first[2][..., :6], second[2][..., :6])
    assert torch.equal(first[2][..., 14:20], second[2][..., 6:12])


def test_native_vocoder_phase_is_independent_of_flow_random_draw_length():
    native = pytest.importorskip("chatterbox.models.s3gen.hifigan")

    class MelDecoder(FakeDecoder):
        def __init__(self):
            super().__init__()
            self.phase = native.SineGen(1000, harmonic_num=8)
            self.sources = []

        def forward(self, *, speech_tokens, **kwargs):
            # Reproduce CFM's random draw even if caller supplies fixed mel noise.
            torch.randn(1, 80, speech_tokens.shape[-1] * 2)
            return super().forward(speech_tokens=speech_tokens, **kwargs)

        def hift_inference(self, mels):
            # Hold the diagnostic F0 constant to isolate phase/noise from mel drift.
            source = self.phase(torch.full((1, 1, 1000), 150.))[0]
            self.sources.append(source)
            return mels[:, 0, :].repeat_interleave(20, dim=-1), source

    model = fake_model()
    model.s3gen = MelDecoder()
    decoder = PrefixAudioDecoder(model, StreamConfig(max_gen_len=30))
    state = torch.random.get_rng_state().clone()
    decoder.decode(list(range(1, 11)), final=False)
    decoder.decode(list(range(1, 16)), final=False)
    assert torch.equal(*model.s3gen.sources)
    assert torch.equal(state, torch.random.get_rng_state())


def test_window_eviction_bounds_decoder_work_and_preserves_all_sample_positions():
    model = fake_model()
    windowed = NanoStreamer(model, StreamConfig(chunk_tokens=5, overlap_ms=40,
                                               left_context_tokens=2, max_gen_len=45,
                                               audio_mode="legacy"))
    torch.manual_seed(17)
    chunks = list(windowed.stream("Hello"))
    records = windowed.metrics["decoder_calls"]
    assert len(records) > 5 and records[-1]["start_token"] > 20
    assert max(r["input_tokens"] for r in records) <= 2 + 5 + 3 + 1 + 3
    assert chunks[-1].final and windowed.metrics["stop_reason"] == "cap"
    assert all(c.start_sample == sum(len(x.samples) for x in chunks[:i])
               for i, c in enumerate(chunks))
    # Fake decoder has no context dependence: any dropped/duplicated window audio
    # therefore differs exactly from this independent growing-prefix control.
    full = NanoStreamer(fake_model(), StreamConfig(chunk_tokens=5, overlap_ms=40, max_gen_len=45,
                                                  audio_mode="legacy"))
    torch.manual_seed(17)
    expected = list(full.stream("Hello"))
    assert windowed.metrics["speech_token_ids"] == full.metrics["speech_token_ids"]
    assert torch.equal(torch.cat([c.samples for c in chunks]),
                       torch.cat([c.samples for c in expected]))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Requires Metal")
def test_real_mps_and_cpu_rng_restored_after_decoder_exception():
    torch.manual_seed(17)
    cpu_before = torch.random.get_rng_state().clone()
    mps_before = torch.mps.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="decoder failed"):
        with isolated_decoder_rng(torch.device("mps"), 18):
            torch.randn(7)
            torch.randn(7, device="mps")
            raise RuntimeError("decoder failed")
    assert torch.equal(cpu_before, torch.random.get_rng_state())
    assert torch.equal(mps_before, torch.mps.get_rng_state())


def test_vocoder_lookahead_is_separate_from_crossfade_and_flushes_on_final():
    stitcher = AudioStitcher(4, lookahead_samples=10)
    waveform = torch.arange(60).float()
    parts = []
    for end, final in ((12, False), (30, False), (42, False), (60, True)):
        start, piece, _ = stitcher.push(waveform[:end], final=final)
        assert start == sum(len(p) for p in parts)
        assert start + len(piece) == (end if final else max(0, end - 14))
        parts.append(piece)
    assert torch.allclose(torch.cat(parts), waveform)


def test_committed_mels_ignore_revised_old_frames_and_keep_native_source_cache():
    model = fake_model()
    decoder = CommittedMelAudioDecoder(model, StreamConfig(max_gen_len=30))
    first = decoder.decode([1, 2, 3, 4, 5, 6], final=False)
    before = decoder.mels.clone()
    source_before = decoder.source_cache.clone()
    calls = []
    original = model.s3gen.hift_inference

    def capture(mels, cache):
        calls.append(cache.clone())
        return original(mels, cache)

    model.s3gen.hift_inference = capture
    last = decoder.decode([90, 91, 92, 4, 5, 6, 7, 8], final=True)
    assert torch.equal(decoder.mels[..., :before.shape[-1]], before)
    assert torch.equal(calls[0], source_before)
    assert torch.equal(last[:len(first)], first)
    assert decoder.mels.shape[-1] == (8 + 3) * 2


def test_continuous_stream_emits_before_eos_and_retains_exact_global_length():
    model = fake_model()
    settings = StreamConfig(chunk_tokens=5, overlap_ms=40, max_gen_len=45,
                            left_context_tokens=3, vocoder_lookahead_ms=300)
    streamer = NanoStreamer(model, settings)
    torch.manual_seed(17)
    chunks = list(streamer.stream("Hello"))
    assert chunks[0].final is False and chunks[0].speech_tokens < 46
    assert chunks[-1].final and streamer.metrics["stop_reason"] == "cap"
    assert sum(len(c.samples) for c in chunks) == (46 + 3) * 40
    assert all(c.start_sample == sum(len(x.samples) for x in chunks[:i])
               for i,c in enumerate(chunks))
    records = streamer.metrics["decoder_calls"]
    assert records[-1]["start_token"] > 0
    assert records[-1]["committed_mel_frames"] == (46 + 3) * 2




def test_conditioned_history_matches_tokens_frames_and_noise_after_window_moves():
    from chatterbox.nano_streaming.streaming import AnchoredMelAudioDecoder
    model = fake_model()
    model.conds.gen.update(prompt_token=torch.tensor([[9]]), prompt_token_len=torch.tensor([1]),
                           prompt_feat=torch.full((1, 2, 80), 9.0),
                           prompt_feat_len=torch.tensor([2]), embedding=torch.ones(1, 4))
    native = model.s3gen.forward
    captures = []

    def capture(**kwargs):
        captures.append(kwargs)
        return native(**kwargs)

    model.s3gen.forward = capture
    decoder = AnchoredMelAudioDecoder(model, StreamConfig(max_gen_len=30, left_context_tokens=2))
    decoder.decode(list(range(1, 9)), final=False)
    before = decoder.mels.clone()
    waveform = decoder.decode(list(range(3, 12)), final=False, start_token=2)
    ref = captures[-1]['ref_dict']
    assert ref['prompt_token'].tolist() == [[9, 4, 5]]
    assert ref['prompt_token_len'].tolist() == [3] and ref['prompt_feat_len'].tolist() == [6]
    assert torch.equal(ref['prompt_feat'][:, 2:, :], before[..., 6:10].transpose(1, 2))
    assert captures[-1]['speech_tokens'].tolist() == [[6, 7, 8, 9, 10, 11]]
    assert torch.equal(captures[-1]['noised_mels'][..., 2:], decoder.noise[..., 6:16])
    assert torch.equal(decoder.mels[..., :10], before)
    assert len(waveform) == 8 * 40 and decoder.records[-1]['conditioning_history_tokens'] == 2
    assert model.conds.gen['prompt_token'].tolist() == [[9]]
    assert model.conds.gen['prompt_feat'].shape == (1, 2, 80)
    assert model.s3gen.forward is capture


def test_live_stream_uses_conditioned_decoder_before_generation_finishes():
    from chatterbox.nano_streaming.streaming import AnchoredMelAudioDecoder, make_audio_decoder
    model = fake_model()
    config = StreamConfig(max_gen_len=45, left_context_tokens=3)
    assert config.audio_mode == 'conditioned_history' and config.vocoder_lookahead_ms == 0
    assert isinstance(make_audio_decoder(model, config), AnchoredMelAudioDecoder)
    torch.manual_seed(17)
    expected = list(stream_speech_tokens(model.t3, None, torch.tensor([[2, 3]]), config))
    calls = []
    hook = model.t3.tfmr.register_forward_hook(lambda *args: calls.append(1))
    streamer = NanoStreamer(model, config)
    torch.manual_seed(17)
    generator = streamer.stream('Hello')
    first = next(generator)
    assert not first.final and first.speech_tokens == config.chunk_tokens
    assert len(calls) == config.chunk_tokens < len(expected)
    chunks = [first, *generator]
    hook.remove()
    assert streamer.metrics['speech_token_ids'] == [event.token_id for event in expected]
    assert streamer.metrics['history_conditioning']
    assert streamer.metrics['max_conditioning_history_tokens'] == 3
    assert all(chunk.start_sample == sum(len(c.samples) for c in chunks[:i])
               for i, chunk in enumerate(chunks))
    assert sum(len(c.samples) for c in chunks) == (46 + 3) * 40
    assert chunks[-1].final and streamer.metrics['stop_reason'] == 'cap'


@pytest.mark.parametrize("kwargs", [{"chunk_tokens": 0}, {"top_p": 0},
                                    {"overlap_ms": float("nan")}, {"max_gen_len": -1},
                                    {"left_context_tokens": -1}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        StreamConfig(**kwargs)

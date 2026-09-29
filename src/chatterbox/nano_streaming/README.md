# Chatterbox Nano audio streaming

Import `load_nano` and `StreamConfig` from `chatterbox.nano_streaming`, then consume `streamer.stream(text)` as chunks arrive; the command-line entry point is `python -m chatterbox.nano_streaming`.

Default streaming uses five-token chunks, 25-token generated history, the native three-token lookahead, and 40 ms audio overlap, without an extra 500 ms delay.

The fixes correct the cropped attention mask, condition new acoustic frames on actual generated history, preserve committed frames, keep global noise positions stable, and isolate waveform random states from native token sampling.

Installation, examples, validation commands and performance boundaries are documented in the clone's root `STREAMING.md`; the packaged source manifest supports local and wheel installations without requiring Reflex or a Git checkout.

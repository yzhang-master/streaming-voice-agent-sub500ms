# streaming-voice-agent-sub500ms

> **Era:** 2025+ · Staff AI Engineer focus area (low-latency streaming agents). Technique from 2025+, reimplemented in 2026 with NumPy and deterministic local adapters.

## Problem

A voice agent that waits for endpointing, a final transcript, and a complete response serializes all of
its latency. This repository implements the alternative as an executable pipeline: incremental speech
frames trigger partial transcripts, a response can start on a trusted partial, and each completed
sentence is synthesized independently.

The local adapters make scheduling and accounting reproducible without audio or LLM SDKs. They do
**not** evaluate neural recognition, speech quality, model reasoning, network jitter, or speaker/playback
latency.

## Method

```text
PCM16 WAV -> 20 ms frames -> RMS-energy VAD (pre-roll + hysteresis)
          -> incremental scripted ASR -> early-start scripted responder
          -> incremental sentence chunker -> per-sentence deterministic PCM TTS
```

The VAD computes `20 log10(sqrt(mean(x²)) / 32768)`. Two speech frames start a turn; eight silent
frames endpoint it. Two frames of pre-roll reach the ASR adapter. The acoustic speech end remains the
last speech frame's end, rather than the later endpoint-detection time.

The scripted ASR emits monotonic partials as speech frames arrive. The responder starts when the
fixture's final-words trigger appears, or only on the final transcript in the baseline. Response pieces
are streamed into a boundary chunker; sentence audio is ready after explicit adapter processing delays.
Voice-to-voice latency is:

```text
first sentence-audio-ready timestamp - last input speech-frame end timestamp
```

Fixtures are real mono PCM16 WAV bytes containing deterministic speech-like tones and silence, replayed
through the same framing and VAD code as any other input. They are not recordings of intelligible speech.

## Results

Eight committed WAV turns, replayed on CPU. p50/p95 use NumPy linear interpolation. These are
**audio-timeline simulation results**, not measurements of a production service.

| Response scheduling | p50 voice-to-voice | p95 voice-to-voice | Min–max | Early LLM starts |
|---|---:|---:|---:|---:|
| Start on partial ASR | **52.0 ms** | **59.2 ms** | 48–62 ms | 8/8 |
| Wait for final ASR | 292.0 ms | 299.2 ms | 288–302 ms | 0/8 |

Early scheduling reduced p50 by 240.0 ms and p95 by 240.0 ms. The LLM began a median 240 ms before the
final ASR update. Endpointing alone contributes 160 ms after acoustic speech end; ASR, responder, and
TTS delays are explicit in the adapters. Both modes emitted two non-empty synthesized PCM sentences per
turn. Raw per-turn timestamps and output sample counts are in `results/replay.json`.

The repository name is a target scenario, not a blanket guarantee. Although this controlled replay is
below 500 ms, no production sub-500 ms claim is made.

The 17 tests were also mutation-checked. Replacing RMS with mean absolute energy, adding an endpoint
frame, disabling sentence boundaries, and disabling early-trigger matching each made targeted tests
fail. Every mutation was restored and the intact suite then passed. Commands and summaries are in
`results/mutation.json`.

## Reproduce

Python 3.10+ is sufficient; the test environment used Python 3.12.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
ruff check .
pytest -q
python scripts/replay.py --smoke   # two real WAV turns, no output file
python scripts/replay.py           # all eight turns, rewrites results/replay.json
python scripts/mutation_check.py   # expects failures, restores sources, writes mutation.json
```

To recreate the deterministic PCM fixtures and manifest:

```bash
python scripts/generate_fixtures.py
```

## Limitations

- ASR transcripts, response text, and response triggers are scripted. Audio energy controls VAD and
  transcript timing; there is no neural speech recognizer or language model.
- TTS produces intelligible-text metadata plus non-speech tonal PCM. It does not measure naturalness or
  synthesis quality. Audio is synthesized but not sent through an audio device.
- Processing delays are configured constants/formulas. `replay_wall_ms` records local execution time,
  but the reported voice latency uses the deterministic event timeline and excludes actual compute
  jitter, streaming transport, playback buffering, barge-in, echo cancellation, and turn interruption.
- Eight similar synthetic traces are enough for a smoke benchmark, not a statistically representative
  p95 for production traffic. The simple sentence boundary rule does not understand abbreviations.
- No external ASR/LLM/TTS adapter or production latency integration is included.

## References

- J. Sohn, N. S. Kim, W. Sung. *A Statistical Model-Based Voice Activity Detection.* IEEE Signal
  Processing Letters, 1999.
- J. Ramírez et al. *Voice Activity Detection: Fundamentals and Speech Recognition System Robustness.*
  In *Robust Speech Recognition and Understanding*, 2007.
- Y. He et al. *Streaming End-to-End Speech Recognition for Mobile Devices.* ICASSP, 2019.
- A. Graves. *Sequence Transduction with Recurrent Neural Networks.* 2012. arXiv:1211.3711.

## License

MIT

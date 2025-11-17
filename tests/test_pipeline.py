import numpy as np

from streaming_voice_agent.audio import iter_audio_frames
from streaming_voice_agent.pipeline import StreamingVoiceAgent


def run(early_start: bool):
    sample_rate = 16_000
    speech_ms = 680
    speech_t = np.arange(sample_rate * speech_ms // 1000) / sample_rate
    speech = (3_000 * np.sin(2 * np.pi * 180 * speech_t)).astype(np.int16)
    samples = np.concatenate(
        [
            np.zeros(sample_rate // 10, dtype=np.int16),
            speech,
            np.zeros(sample_rate // 2, dtype=np.int16),
        ]
    )
    agent = StreamingVoiceAgent(
        transcript="what time is it now",
        response_chunks=("The local time ", "is nine thirty. ", "I can help."),
        early_trigger="it now",
        early_start=early_start,
    )
    for frame in iter_audio_frames(samples, sample_rate):
        agent.process(frame)
    return agent.finish()


def test_end_to_end_early_start_and_final_only_replay() -> None:
    early = run(True)
    late = run(False)
    assert early.final_asr_ms == late.final_asr_ms
    assert early.llm_started_ms < late.llm_started_ms
    assert early.llm_lead_ms > 0
    assert early.voice_to_voice_ms < late.voice_to_voice_ms
    assert [item.sentence for item in early.audio] == [
        "The local time is nine thirty.",
        "I can help.",
    ]
    assert all(len(item.samples) > 0 for item in early.audio)
    assert early.speech_end_ms == 780


def test_final_only_waits_for_vad_endpoint() -> None:
    trace = run(False)
    assert trace.llm_started_ms == trace.final_asr_ms
    assert trace.endpoint_detected_ms == trace.speech_end_ms + 160
    assert trace.first_audio_ms > trace.final_asr_ms

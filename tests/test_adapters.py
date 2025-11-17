import numpy as np
import pytest

from streaming_voice_agent.asr import ScriptedStreamingASR
from streaming_voice_agent.audio import AudioFrame
from streaming_voice_agent.chunking import Sentence
from streaming_voice_agent.llm import ScriptedEarlyLLM
from streaming_voice_agent.tts import ScriptedTTS


def speech_frame(index: int) -> AudioFrame:
    return AudioFrame(np.full(320, 2_000, dtype=np.int16), 16_000, index * 20, (index + 1) * 20)


def test_asr_emits_monotonic_partials_and_one_final() -> None:
    asr = ScriptedStreamingASR("what time is it", frames_per_word=2, processing_ms=5)
    updates = []
    for index in range(8):
        updates.extend(asr.push((speech_frame(index),)))
    assert [update.text for update in updates] == [
        "what",
        "what time",
        "what time is",
        "what time is it",
    ]
    assert updates[-1].ready_at_ms == 165
    final = asr.finalize(200)
    assert final.final
    assert final.text == "what time is it"
    assert final.ready_at_ms == 205
    with pytest.raises(RuntimeError):
        asr.push((speech_frame(8),))


def test_llm_starts_once_on_trigger_but_not_in_final_only_mode() -> None:
    asr = ScriptedStreamingASR("hello there", frames_per_word=1)
    partial = asr.push((speech_frame(0),))[0]
    early = ScriptedEarlyLLM(("Hi. ", "Again."), "hello")
    pieces = early.consume(partial)
    assert len(pieces) == 2
    final = asr.finalize(50)
    assert early.consume(final) == []

    final_only = ScriptedEarlyLLM(("Hi.",), "hello", early_start=False)
    assert final_only.consume(partial) == []
    assert len(final_only.consume(final)) == 1


def test_tts_returns_nonempty_pcm_and_processing_deadline() -> None:
    tts = ScriptedTTS(base_processing_ms=40, processing_ms_per_char=1)
    audio = tts.synthesize(Sentence("A short reply.", 100))
    assert audio.ready_at_ms == 154
    assert audio.samples.dtype == np.int16
    assert len(audio.samples) == 16_000 * 540 // 1000
    assert np.max(np.abs(audio.samples.astype(np.int32))) > 0

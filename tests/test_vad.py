import numpy as np
import pytest

from streaming_voice_agent.audio import AudioFrame
from streaming_voice_agent.vad import EnergyVAD, VADConfig, rms_dbfs


def frame(amplitude: float, index: int, shape: float = 1.0) -> AudioFrame:
    samples = np.full(320, amplitude * shape, dtype=np.int16)
    return AudioFrame(samples, 16_000, index * 20, (index + 1) * 20)


def test_rms_dbfs_uses_root_mean_square_not_mean_absolute_value() -> None:
    samples = np.array([10_000, -10_000, 0, 0], dtype=np.int16)
    expected = 20 * np.log10(np.sqrt(np.mean(samples.astype(float) ** 2)) / 32768)
    assert rms_dbfs(samples) == pytest.approx(expected)
    assert rms_dbfs(np.zeros(8, dtype=np.int16)) == float("-inf")


def test_vad_requires_speech_and_silence_confirmation() -> None:
    vad = EnergyVAD(VADConfig(min_speech_frames=2, min_silence_frames=3, preroll_frames=1))
    assert not vad.process(frame(0, 0)).started
    assert not vad.process(frame(2_000, 1)).started
    start = vad.process(frame(2_000, 2))
    assert start.started
    assert [item.start_ms for item in start.speech_frames] == [0, 20, 40]
    assert vad.process(frame(2_000, 3)).speech_frames[0].start_ms == 60
    assert not vad.process(frame(0, 4)).ended
    assert not vad.process(frame(0, 5)).ended
    end = vad.process(frame(0, 6))
    assert end.ended
    assert end.speech_end_ms == 80


def test_short_noise_burst_does_not_trigger_and_preroll_is_bounded() -> None:
    vad = EnergyVAD(VADConfig(min_speech_frames=2, preroll_frames=2))
    for index in range(4):
        vad.process(frame(0, index))
    vad.process(frame(3_000, 4))
    vad.process(frame(0, 5))
    start = vad.process(frame(3_000, 6))
    assert not start.started
    start = vad.process(frame(3_000, 7))
    assert start.started
    assert [item.start_ms for item in start.speech_frames] == [60, 100, 120, 140]


def test_flush_endpoints_active_speech_without_invented_silence() -> None:
    vad = EnergyVAD(VADConfig(min_speech_frames=1))
    vad.process(frame(2_000, 0))
    vad.process(frame(2_000, 1))
    end = vad.flush()
    assert end.ended
    assert end.speech_end_ms == 40
    assert not vad.flush().ended


def test_invalid_hysteresis_is_rejected() -> None:
    with pytest.raises(ValueError):
        VADConfig(min_speech_frames=0)

import numpy as np
import pytest

from streaming_voice_agent.audio import iter_audio_frames, read_pcm16_wav, write_pcm16_wav


def test_pcm16_wav_round_trip_and_real_frame_boundaries(tmp_path) -> None:
    samples = np.arange(-1_000, 1_000, dtype=np.int16)
    path = tmp_path / "audio.wav"
    write_pcm16_wav(path, samples, 16_000)
    restored, sample_rate = read_pcm16_wav(path)
    assert np.array_equal(restored, samples)
    frames = list(iter_audio_frames(restored, sample_rate, frame_ms=20))
    assert len(frames) == 7
    assert frames[0].start_ms == 0
    assert frames[-1].end_ms == 125
    assert np.array_equal(np.concatenate([frame.samples for frame in frames]), samples)


def test_stereo_wav_is_rejected(tmp_path) -> None:
    import wave

    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as target:
        target.setnchannels(2)
        target.setsampwidth(2)
        target.setframerate(16_000)
        target.writeframes(np.zeros(20, dtype=np.int16).tobytes())
    with pytest.raises(ValueError, match="mono PCM16"):
        read_pcm16_wav(path)

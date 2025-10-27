from __future__ import annotations

import wave
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AudioFrame:
    samples: np.ndarray
    sample_rate: int
    start_ms: int
    end_ms: int

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


def read_pcm16_wav(path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2:
            raise ValueError("only mono PCM16 WAV files are supported")
        sample_rate = source.getframerate()
        samples = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").copy()
    return samples, sample_rate


def write_pcm16_wav(path: str | Path, samples: np.ndarray, sample_rate: int) -> None:
    pcm = np.asarray(samples, dtype=np.int16)
    with wave.open(str(path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(sample_rate)
        target.writeframes(pcm.astype("<i2", copy=False).tobytes())


def iter_audio_frames(
    samples: np.ndarray, sample_rate: int, frame_ms: int = 20
) -> Iterator[AudioFrame]:
    if sample_rate <= 0 or frame_ms <= 0:
        raise ValueError("sample_rate and frame_ms must be positive")
    pcm = np.asarray(samples, dtype=np.int16)
    frame_size = sample_rate * frame_ms // 1000
    if frame_size == 0:
        raise ValueError("frame_ms is shorter than one sample")
    for offset in range(0, len(pcm), frame_size):
        chunk = pcm[offset : offset + frame_size]
        start_ms = round(offset * 1000 / sample_rate)
        end_ms = round((offset + len(chunk)) * 1000 / sample_rate)
        yield AudioFrame(chunk, sample_rate, start_ms, end_ms)

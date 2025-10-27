from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from streaming_voice_agent.audio import AudioFrame


@dataclass(frozen=True)
class VADConfig:
    threshold_dbfs: float = -45.0
    min_speech_frames: int = 2
    min_silence_frames: int = 8
    preroll_frames: int = 2

    def __post_init__(self) -> None:
        if min(self.min_speech_frames, self.min_silence_frames) < 1:
            raise ValueError("speech and silence confirmation require at least one frame")
        if self.preroll_frames < 0:
            raise ValueError("preroll_frames cannot be negative")


@dataclass(frozen=True)
class VADUpdate:
    started: bool = False
    ended: bool = False
    speech_frames: tuple[AudioFrame, ...] = ()
    speech_end_ms: int | None = None


def rms_dbfs(samples: np.ndarray) -> float:
    pcm = np.asarray(samples, dtype=np.float64)
    if pcm.size == 0:
        return float("-inf")
    rms = np.sqrt(np.mean(np.square(pcm)))
    if rms == 0:
        return float("-inf")
    return float(20 * np.log10(rms / 32768.0))


class EnergyVAD:
    def __init__(self, config: VADConfig | None = None) -> None:
        self.config = config or VADConfig()
        self._active = False
        self._candidates: list[AudioFrame] = []
        self._preroll: deque[AudioFrame] = deque(maxlen=self.config.preroll_frames)
        self._silence_frames = 0
        self._last_speech_end_ms: int | None = None

    @property
    def active(self) -> bool:
        return self._active

    def process(self, frame: AudioFrame) -> VADUpdate:
        is_speech = rms_dbfs(frame.samples) >= self.config.threshold_dbfs
        if not self._active:
            if not is_speech:
                self._candidates.clear()
                self._preroll.append(frame)
                return VADUpdate()
            self._candidates.append(frame)
            if len(self._candidates) < self.config.min_speech_frames:
                return VADUpdate()
            frames = (*self._preroll, *self._candidates)
            self._active = True
            self._candidates.clear()
            self._preroll.clear()
            self._silence_frames = 0
            self._last_speech_end_ms = frame.end_ms
            return VADUpdate(started=True, speech_frames=frames)

        if is_speech:
            self._silence_frames = 0
            self._last_speech_end_ms = frame.end_ms
            return VADUpdate(speech_frames=(frame,))

        self._silence_frames += 1
        if self._silence_frames < self.config.min_silence_frames:
            return VADUpdate()
        self._active = False
        self._silence_frames = 0
        self._preroll.append(frame)
        return VADUpdate(ended=True, speech_end_ms=self._last_speech_end_ms)

    def flush(self) -> VADUpdate:
        if not self._active:
            return VADUpdate()
        self._active = False
        return VADUpdate(ended=True, speech_end_ms=self._last_speech_end_ms)

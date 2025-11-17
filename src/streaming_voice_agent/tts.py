from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from streaming_voice_agent.chunking import Sentence


@dataclass(frozen=True)
class SynthesizedAudio:
    sentence: str
    samples: np.ndarray
    sample_rate: int
    ready_at_ms: int

    @property
    def duration_ms(self) -> int:
        return round(len(self.samples) * 1000 / self.sample_rate)


class ScriptedTTS:
    """Deterministic PCM synthesizer used to exercise sentence-level streaming."""

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        base_processing_ms: int = 45,
        processing_ms_per_char: float = 0.5,
        milliseconds_per_word: int = 180,
    ) -> None:
        self.sample_rate = sample_rate
        self.base_processing_ms = base_processing_ms
        self.processing_ms_per_char = processing_ms_per_char
        self.milliseconds_per_word = milliseconds_per_word

    def synthesize(self, sentence: Sentence) -> SynthesizedAudio:
        words = sentence.text.split()
        if not words:
            raise ValueError("cannot synthesize an empty sentence")
        duration_ms = max(120, len(words) * self.milliseconds_per_word)
        count = self.sample_rate * duration_ms // 1000
        time = np.arange(count, dtype=np.float64) / self.sample_rate
        envelope = np.sin(np.pi * np.arange(count) / max(count - 1, 1)) ** 2
        frequency = 150.0 + (sum(map(ord, sentence.text)) % 100)
        signal = 1800 * envelope * np.sin(2 * np.pi * frequency * time)
        signal += 700 * envelope * np.sin(2 * np.pi * frequency * 2 * time)
        processing_ms = self.base_processing_ms + math.ceil(
            len(sentence.text) * self.processing_ms_per_char
        )
        return SynthesizedAudio(
            sentence=sentence.text,
            samples=np.asarray(np.round(signal), dtype=np.int16),
            sample_rate=self.sample_rate,
            ready_at_ms=sentence.ready_at_ms + processing_ms,
        )

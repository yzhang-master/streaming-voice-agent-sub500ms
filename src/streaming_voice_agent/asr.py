from __future__ import annotations

from dataclasses import dataclass

from streaming_voice_agent.audio import AudioFrame
from streaming_voice_agent.vad import rms_dbfs


@dataclass(frozen=True)
class ASRUpdate:
    text: str
    final: bool
    ready_at_ms: int


class ScriptedStreamingASR:
    """Replays a known transcript in proportion to consumed speech frames."""

    def __init__(
        self,
        transcript: str,
        *,
        frames_per_word: int = 5,
        processing_ms: int = 25,
        speech_threshold_dbfs: float = -45.0,
    ) -> None:
        self._words = transcript.split()
        if not self._words:
            raise ValueError("a scripted transcript cannot be empty")
        if frames_per_word < 1 or processing_ms < 0:
            raise ValueError("frames_per_word must be positive and processing_ms non-negative")
        self._frames_per_word = frames_per_word
        self._processing_ms = processing_ms
        self._threshold_dbfs = speech_threshold_dbfs
        self._speech_frames = 0
        self._emitted_words = 0
        self._finalized = False

    def push(self, frames: tuple[AudioFrame, ...]) -> list[ASRUpdate]:
        if self._finalized:
            raise RuntimeError("cannot push audio after finalization")
        updates: list[ASRUpdate] = []
        for frame in frames:
            if rms_dbfs(frame.samples) < self._threshold_dbfs:
                continue
            self._speech_frames += 1
            due_words = min(self._speech_frames // self._frames_per_word, len(self._words))
            if due_words > self._emitted_words:
                self._emitted_words = due_words
                updates.append(
                    ASRUpdate(
                        " ".join(self._words[:due_words]),
                        final=False,
                        ready_at_ms=frame.end_ms + self._processing_ms,
                    )
                )
        return updates

    def finalize(self, detected_at_ms: int) -> ASRUpdate:
        if self._finalized:
            raise RuntimeError("transcript was already finalized")
        self._finalized = True
        self._emitted_words = len(self._words)
        return ASRUpdate(
            " ".join(self._words), final=True, ready_at_ms=detected_at_ms + self._processing_ms
        )

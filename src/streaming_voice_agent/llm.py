from __future__ import annotations

from dataclasses import dataclass

from streaming_voice_agent.asr import ASRUpdate


@dataclass(frozen=True)
class ResponsePiece:
    text: str
    ready_at_ms: int


class ScriptedEarlyLLM:
    """Starts a scripted response on a partial transcript or waits for the final one."""

    def __init__(
        self,
        response_chunks: tuple[str, ...],
        early_trigger: str,
        *,
        early_start: bool = True,
        first_chunk_ms: int = 45,
        chunk_ms: int = 12,
    ) -> None:
        if not response_chunks or not "".join(response_chunks).strip():
            raise ValueError("a scripted response cannot be empty")
        if not early_trigger.strip():
            raise ValueError("early_trigger cannot be empty")
        self._chunks = response_chunks
        self._early_trigger = " ".join(early_trigger.lower().split())
        self._early_start = early_start
        self._first_chunk_ms = first_chunk_ms
        self._chunk_ms = chunk_ms
        self.started_at_ms: int | None = None

    def consume(self, update: ASRUpdate) -> list[ResponsePiece]:
        if self.started_at_ms is not None:
            return []
        normalized = " ".join(update.text.lower().split())
        if not update.final and not (
            self._early_start and self._early_trigger in normalized
        ):
            return []
        self.started_at_ms = update.ready_at_ms
        return [
            ResponsePiece(
                text,
                update.ready_at_ms + self._first_chunk_ms + index * self._chunk_ms,
            )
            for index, text in enumerate(self._chunks)
        ]

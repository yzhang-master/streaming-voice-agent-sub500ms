from __future__ import annotations

import re
from dataclasses import dataclass

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])(?:\s+|$)")


@dataclass(frozen=True)
class Sentence:
    text: str
    ready_at_ms: int


class SentenceChunker:
    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, text: str, ready_at_ms: int) -> list[Sentence]:
        self._buffer += text
        sentences: list[Sentence] = []
        while match := _SENTENCE_BOUNDARY.search(self._buffer):
            end = match.start()
            sentence = self._buffer[:end].strip()
            self._buffer = self._buffer[match.end() :]
            if sentence:
                sentences.append(Sentence(sentence, ready_at_ms))
        return sentences

    def flush(self, ready_at_ms: int) -> list[Sentence]:
        sentence = self._buffer.strip()
        self._buffer = ""
        return [Sentence(sentence, ready_at_ms)] if sentence else []

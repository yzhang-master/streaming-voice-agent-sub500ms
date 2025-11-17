from streaming_voice_agent.chunking import SentenceChunker


def test_chunker_emits_as_soon_as_sentence_boundary_arrives() -> None:
    chunker = SentenceChunker()
    assert chunker.feed("The local time ", 10) == []
    first = chunker.feed("is nine thirty. ", 20)
    assert [item.text for item in first] == ["The local time is nine thirty."]
    assert first[0].ready_at_ms == 20
    assert chunker.feed("I can also ", 30) == []
    second = chunker.feed("set a reminder.", 40)
    assert [item.text for item in second] == ["I can also set a reminder."]
    assert chunker.flush(50) == []


def test_chunker_handles_multiple_sentences_decimals_and_unterminated_tail() -> None:
    chunker = SentenceChunker()
    sentences = chunker.feed("Pi is 3.14. First! Second? tail", 12)
    assert [item.text for item in sentences] == ["Pi is 3.14.", "First!", "Second?"]
    tail = chunker.flush(20)
    assert [item.text for item in tail] == ["tail"]


def test_fragmented_input_preserves_exact_text() -> None:
    chunker = SentenceChunker()
    output = []
    for index, piece in enumerate(["Hello", " there.", " How ", "are you?"]):
        output.extend(chunker.feed(piece, index))
    assert [item.text for item in output] == ["Hello there.", "How are you?"]

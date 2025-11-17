from __future__ import annotations

from dataclasses import dataclass, field

from streaming_voice_agent.asr import ASRUpdate, ScriptedStreamingASR
from streaming_voice_agent.audio import AudioFrame
from streaming_voice_agent.chunking import Sentence, SentenceChunker
from streaming_voice_agent.llm import ResponsePiece, ScriptedEarlyLLM
from streaming_voice_agent.tts import ScriptedTTS, SynthesizedAudio
from streaming_voice_agent.vad import EnergyVAD, VADConfig


@dataclass
class TurnTrace:
    asr_updates: list[ASRUpdate] = field(default_factory=list)
    response_pieces: list[ResponsePiece] = field(default_factory=list)
    sentences: list[Sentence] = field(default_factory=list)
    audio: list[SynthesizedAudio] = field(default_factory=list)
    speech_end_ms: int | None = None
    endpoint_detected_ms: int | None = None
    final_asr_ms: int | None = None
    llm_started_ms: int | None = None
    first_audio_ms: int | None = None

    @property
    def voice_to_voice_ms(self) -> int | None:
        if self.speech_end_ms is None or self.first_audio_ms is None:
            return None
        return self.first_audio_ms - self.speech_end_ms

    @property
    def llm_lead_ms(self) -> int | None:
        if self.final_asr_ms is None or self.llm_started_ms is None:
            return None
        return self.final_asr_ms - self.llm_started_ms


class StreamingVoiceAgent:
    def __init__(
        self,
        *,
        transcript: str,
        response_chunks: tuple[str, ...],
        early_trigger: str,
        early_start: bool = True,
        vad_config: VADConfig | None = None,
    ) -> None:
        self.vad = EnergyVAD(vad_config)
        self.asr = ScriptedStreamingASR(transcript)
        self.llm = ScriptedEarlyLLM(response_chunks, early_trigger, early_start=early_start)
        self.chunker = SentenceChunker()
        self.tts = ScriptedTTS()
        self.trace = TurnTrace()

    def _consume_asr(self, update: ASRUpdate) -> None:
        self.trace.asr_updates.append(update)
        if update.final:
            self.trace.final_asr_ms = update.ready_at_ms
        for piece in self.llm.consume(update):
            if self.trace.llm_started_ms is None:
                self.trace.llm_started_ms = self.llm.started_at_ms
            self.trace.response_pieces.append(piece)
            for sentence in self.chunker.feed(piece.text, piece.ready_at_ms):
                self.trace.sentences.append(sentence)
                self.trace.audio.append(self.tts.synthesize(sentence))

    def process(self, frame: AudioFrame) -> None:
        update = self.vad.process(frame)
        if update.speech_frames:
            for asr_update in self.asr.push(update.speech_frames):
                self._consume_asr(asr_update)
        if update.ended:
            self.trace.speech_end_ms = update.speech_end_ms
            self.trace.endpoint_detected_ms = frame.end_ms
            self._consume_asr(self.asr.finalize(frame.end_ms))

    def finish(self) -> TurnTrace:
        update = self.vad.flush()
        if update.ended:
            self.trace.speech_end_ms = update.speech_end_ms
            last_audio_end = self.trace.asr_updates[-1].ready_at_ms if self.trace.asr_updates else 0
            self.trace.endpoint_detected_ms = max(self.trace.speech_end_ms or 0, last_audio_end)
            self._consume_asr(self.asr.finalize(self.trace.endpoint_detected_ms))
        if self.llm.started_at_ms is not None:
            ready_at = self.trace.response_pieces[-1].ready_at_ms
            for sentence in self.chunker.flush(ready_at):
                self.trace.sentences.append(sentence)
                self.trace.audio.append(self.tts.synthesize(sentence))
        if self.trace.audio:
            self.trace.first_audio_ms = min(item.ready_at_ms for item in self.trace.audio)
        return self.trace


def run_turn(frames: list[AudioFrame], **agent_kwargs: object) -> TurnTrace:
    agent = StreamingVoiceAgent(**agent_kwargs)  # type: ignore[arg-type]
    for frame in frames:
        agent.process(frame)
    return agent.finish()

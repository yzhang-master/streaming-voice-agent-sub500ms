from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from streaming_voice_agent.audio import write_pcm16_wav

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

UTTERANCES = [
    (
        "time",
        "what time is it now",
        ("The local time ", "is nine thirty. ", "I can also set a reminder."),
        "it now",
    ),
    (
        "weather",
        "what is the weather today",
        ("It is mild and dry today. ", "A light jacket should be enough."),
        "weather today",
    ),
    (
        "reminder",
        "can you set a reminder",
        ("Your reminder is set for six. ", "I will keep it on this device."),
        "a reminder",
    ),
    (
        "calendar",
        "what is on my calendar",
        ("You have one design review at ten. ", "The rest of the morning is free."),
        "my calendar",
    ),
    (
        "traffic",
        "how is the traffic downtown",
        ("Traffic downtown is moving slowly. ", "Allow ten extra minutes."),
        "traffic downtown",
    ),
    (
        "status",
        "what is the latest project status",
        ("The project is on track this week. ", "Two checks remain before release."),
        "project status",
    ),
    (
        "music",
        "can you play some quiet music",
        ("I will start a quiet instrumental mix. ", "You can ask me to stop."),
        "quiet music",
    ),
    (
        "meeting",
        "what is the next team meeting",
        ("The next team meeting starts at two. ", "It is in the main room."),
        "team meeting",
    ),
]


def synthesize_speech_like_audio(
    seed: int, duration_ms: int, sample_rate: int = 16_000
) -> np.ndarray:
    count = sample_rate * duration_ms // 1000
    time = np.arange(count, dtype=np.float64) / sample_rate
    rng = np.random.default_rng(seed)
    frequency = 120 + seed * 13
    modulation = 0.65 + 0.35 * np.sin(2 * np.pi * (3 + seed % 3) * time) ** 2
    signal = modulation * (
        3_200 * np.sin(2 * np.pi * frequency * time)
        + 1_100 * np.sin(2 * np.pi * frequency * 2.03 * time)
    )
    signal += rng.normal(0, 35, count)
    return np.asarray(np.clip(np.round(signal), -32768, 32767), dtype=np.int16)


def main() -> None:
    audio_dir = DATA / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for seed, (name, transcript, response_chunks, trigger) in enumerate(UTTERANCES, start=1):
        sample_rate = 16_000
        speech_ms = len(transcript.split()) * 100 + 80
        sections = [
            np.zeros(sample_rate // 10, dtype=np.int16),
            synthesize_speech_like_audio(seed, speech_ms, sample_rate),
            np.zeros(sample_rate * 2 // 5, dtype=np.int16),
        ]
        relative_path = Path("audio") / f"{name}.wav"
        write_pcm16_wav(DATA / relative_path, np.concatenate(sections), sample_rate)
        manifest.append(
            {
                "id": name,
                "path": str(relative_path),
                "transcript": transcript,
                "response_chunks": list(response_chunks),
                "early_trigger": trigger,
                "sample_rate": sample_rate,
            }
        )
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

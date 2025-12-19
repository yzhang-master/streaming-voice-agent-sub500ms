from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MUTATIONS = [
    (
        "vad-rms-becomes-mean-absolute",
        "src/streaming_voice_agent/vad.py",
        "rms = np.sqrt(np.mean(np.square(pcm)))",
        "rms = np.mean(np.abs(pcm))",
        ["tests/test_vad.py"],
    ),
    (
        "vad-requires-extra-silence-frame",
        "src/streaming_voice_agent/vad.py",
        "if self._silence_frames < self.config.min_silence_frames:",
        "if self._silence_frames <= self.config.min_silence_frames:",
        ["tests/test_vad.py", "tests/test_pipeline.py"],
    ),
    (
        "sentence-boundaries-disabled",
        "src/streaming_voice_agent/chunking.py",
        '_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])(?:\\s+|$)")',
        '_SENTENCE_BOUNDARY = re.compile(r"$^")',
        ["tests/test_chunking.py", "tests/test_pipeline.py"],
    ),
    (
        "early-trigger-never-matches",
        "src/streaming_voice_agent/llm.py",
        "self._early_start and self._early_trigger in normalized",
        "self._early_start and False",
        ["tests/test_adapters.py", "tests/test_pipeline.py"],
    ),
]


def main() -> None:
    records = []
    for name, relative_path, original, mutant, tests in MUTATIONS:
        path = ROOT / relative_path
        source = path.read_text(encoding="utf-8")
        if source.count(original) != 1:
            raise RuntimeError(f"mutation target is not unique: {name}")
        try:
            path.write_text(source.replace(original, mutant), encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, "-m", "pytest", *tests],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
        finally:
            path.write_text(source, encoding="utf-8")
        if completed.returncode == 0:
            raise RuntimeError(f"tests unexpectedly passed mutation: {name}")
        summary = completed.stdout.strip().splitlines()[-1]
        print(f"PASS (tests rejected): {name}: {summary}")
        records.append(
            {
                "mutation": name,
                "path": relative_path,
                "pytest_returncode": completed.returncode,
                "pytest_summary": summary,
            }
        )
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "restored_after_each_mutation": True,
        "mutations": records,
    }
    result_path = ROOT / "results" / "mutation.json"
    result_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

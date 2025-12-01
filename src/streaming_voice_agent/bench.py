from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from streaming_voice_agent.audio import iter_audio_frames, read_pcm16_wav
from streaming_voice_agent.pipeline import TurnTrace, run_turn

ROOT = Path(__file__).resolve().parents[2]


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of no measurements")
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def trace_result(trace: TurnTrace, wall_ms: float) -> dict[str, Any]:
    if trace.voice_to_voice_ms is None:
        raise RuntimeError("pipeline did not produce audio")
    return {
        "voice_to_voice_ms": trace.voice_to_voice_ms,
        "speech_end_ms": trace.speech_end_ms,
        "endpoint_detected_ms": trace.endpoint_detected_ms,
        "final_asr_ms": trace.final_asr_ms,
        "llm_started_ms": trace.llm_started_ms,
        "llm_lead_ms": trace.llm_lead_ms,
        "first_audio_ms": trace.first_audio_ms,
        "sentence_count": len(trace.sentences),
        "pcm_sample_count": sum(len(item.samples) for item in trace.audio),
        "replay_wall_ms": wall_ms,
    }


def replay_item(item: dict[str, Any], data_dir: Path, early_start: bool) -> dict[str, Any]:
    samples, sample_rate = read_pcm16_wav(data_dir / item["path"])
    if sample_rate != item["sample_rate"]:
        raise ValueError(f"sample-rate mismatch for {item['id']}")
    frames = list(iter_audio_frames(samples, sample_rate))
    started = time.perf_counter()
    trace = run_turn(
        frames,
        transcript=item["transcript"],
        response_chunks=tuple(item["response_chunks"]),
        early_trigger=item["early_trigger"],
        early_start=early_start,
    )
    wall_ms = (time.perf_counter() - started) * 1000
    return {"id": item["id"], **trace_result(trace, wall_ms)}


def summarize(turns: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [turn["voice_to_voice_ms"] for turn in turns]
    leads = [turn["llm_lead_ms"] for turn in turns]
    return {
        "turns": len(turns),
        "voice_to_voice_p50_ms": percentile(latencies, 50),
        "voice_to_voice_p95_ms": percentile(latencies, 95),
        "voice_to_voice_min_ms": min(latencies),
        "voice_to_voice_max_ms": max(latencies),
        "early_started_turns": sum(lead > 0 for lead in leads),
        "llm_lead_p50_ms": percentile(leads, 50),
        "llm_lead_p95_ms": percentile(leads, 95),
        "mean_replay_wall_ms_per_turn": float(np.mean([turn["replay_wall_ms"] for turn in turns])),
        "total_output_pcm_samples": sum(turn["pcm_sample_count"] for turn in turns),
    }


def run_benchmark(root: Path = ROOT, smoke: bool = False) -> dict[str, Any]:
    data_dir = root / "data"
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if smoke:
        manifest = manifest[:2]
    modes = {
        mode: [replay_item(item, data_dir, early_start) for item in manifest]
        for mode, early_start in [("early_start", True), ("final_only", False)]
    }
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "measurement": "deterministic audio-timeline replay; not a production latency benchmark",
        "latency_origin": "last speech frame end",
        "latency_target": "first synthesized sentence audio ready",
        "percentile_method": "NumPy linear interpolation",
        "vad": {
            "frame_ms": 20,
            "threshold_dbfs": -45.0,
            "min_speech_frames": 2,
            "min_silence_frames": 8,
            "preroll_frames": 2,
        },
        "modes": {
            mode: {"summary": summarize(turns), "turns": turns} for mode, turns in modes.items()
        },
    }
    early = result["modes"]["early_start"]["summary"]
    late = result["modes"]["final_only"]["summary"]
    result["p50_reduction_ms"] = late["voice_to_voice_p50_ms"] - early["voice_to_voice_p50_ms"]
    result["p95_reduction_ms"] = late["voice_to_voice_p95_ms"] - early["voice_to_voice_p95_ms"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke", action="store_true", help="replay two turns without writing results"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "replay.json")
    args = parser.parse_args()
    result = run_benchmark(smoke=args.smoke)
    if not args.smoke:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for mode in ("early_start", "final_only"):
        summary = result["modes"][mode]["summary"]
        print(
            f"{mode}: p50={summary['voice_to_voice_p50_ms']:.1f} ms "
            f"p95={summary['voice_to_voice_p95_ms']:.1f} ms "
            f"early={summary['early_started_turns']}/{summary['turns']}"
        )


if __name__ == "__main__":
    main()

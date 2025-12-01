import numpy as np
import pytest

from streaming_voice_agent.bench import percentile, run_benchmark


def test_percentile_is_defined_and_validated() -> None:
    assert percentile([10, 20, 30], 50) == 20
    assert percentile([10, 20], 95) == pytest.approx(19.5)
    with pytest.raises(ValueError):
        percentile([], 50)


def test_real_fixture_smoke_replay_exercises_both_modes() -> None:
    result = run_benchmark(smoke=True)
    early = result["modes"]["early_start"]
    late = result["modes"]["final_only"]
    assert early["summary"]["turns"] == 2
    assert early["summary"]["early_started_turns"] == 2
    assert late["summary"]["early_started_turns"] == 0
    assert early["summary"]["voice_to_voice_p95_ms"] < late["summary"]["voice_to_voice_p95_ms"]
    for mode in (early, late):
        for turn in mode["turns"]:
            assert turn["sentence_count"] == 2
            assert turn["pcm_sample_count"] > 0
            assert np.isfinite(turn["replay_wall_ms"])

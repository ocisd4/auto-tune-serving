"""Regression test for AFSBox bestCandidate selection (afsbox-todo, 2026-09-22).

Real-world trigger: a static grid-search ModelTuning on gx10-worker-02
(tuning-74d4d685) had its bestCandidate synced to "trial_1" even though
trial_15 had ~2x the throughput of every other trial on the Pareto front.
`pareto_front[0]` was never a ranking — it's Optuna's `study.best_trials`
order, which in practice tracks trial submission order. This test locks in
the fix: bestCandidate must be picked by actual score, not list position.
"""

from auto_tune_vllm.execution.afsbox import _select_best_pareto_candidate


def test_picks_highest_scoring_trial_not_first_in_list():
    # Mirrors the real incident: trial_1..trial_5 have modest, near-identical
    # improvements; trial_15 (last in the list) has a clear outlier win.
    pareto_front = [
        {"trial": 1, "values": [554.7, 25.5], "baseline_improvements": [1.0, 0.5]},
        {"trial": 2, "values": [565.4, 25.5], "baseline_improvements": [2.9, 0.4]},
        {"trial": 5, "values": [574.9, 25.4], "baseline_improvements": [4.5, 0.6]},
        {"trial": 15, "values": [1129.2, 25.0], "baseline_improvements": [103.0, 2.1]},
    ]
    assert _select_best_pareto_candidate(pareto_front) == "trial_15"


def test_empty_front_returns_none():
    assert _select_best_pareto_candidate([]) is None


def test_single_trial_front_returns_that_trial():
    pareto_front = [{"trial": 7, "values": [100.0, 10.0], "baseline_improvements": [0.0, 0.0]}]
    assert _select_best_pareto_candidate(pareto_front) == "trial_7"


def test_falls_back_to_normalized_values_when_no_baseline_improvements():
    # No baseline_improvements anywhere (baseline trials disabled/unavailable) —
    # must still rank by raw values instead of collapsing to "first in list".
    objectives = [
        {"metric": "output_tokens_per_second", "direction": "maximize"},
        {"metric": "time_to_first_token_ms", "direction": "minimize"},
    ]
    pareto_front = [
        {"trial": 1, "values": [500.0, 30.0]},   # low throughput, high latency — dominated-ish extreme
        {"trial": 2, "values": [900.0, 15.0]},   # best on both axes among these three
        {"trial": 3, "values": [700.0, 25.0]},
    ]
    assert _select_best_pareto_candidate(pareto_front, objectives) == "trial_2"


def test_falls_back_gracefully_when_all_values_tied():
    # hi == lo for every objective column — must not divide by zero, must
    # still return a valid trial name rather than raising.
    objectives = [{"metric": "output_tokens_per_second", "direction": "maximize"}]
    pareto_front = [
        {"trial": 1, "values": [500.0]},
        {"trial": 2, "values": [500.0]},
    ]
    result = _select_best_pareto_candidate(pareto_front, objectives)
    assert result in {"trial_1", "trial_2"}


def test_skips_none_improvements_within_a_trial():
    # A trial with a mix of real and None improvements (baseline unavailable
    # for one objective only) should score off just the real ones, not crash
    # or treat None as 0.
    pareto_front = [
        {"trial": 1, "values": [500.0, 30.0], "baseline_improvements": [None, 1.0]},
        {"trial": 2, "values": [900.0, 15.0], "baseline_improvements": [50.0, None]},
    ]
    assert _select_best_pareto_candidate(pareto_front) == "trial_2"

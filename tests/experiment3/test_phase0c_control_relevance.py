from scripts.experiment3.phase0c_hidden_dynamics.analyze_control_relevance import (
    classify_control_relevance)


def matrix(low, high, low_success=None, high_success=None):
    candidates = ("straight", "left_bias", "right_bias")
    low_success = low_success or [False] * 3
    high_success = high_success or [False] * 3
    return {
        "uniform_low": {key: {"progress_m": low[index],
                              "success": low_success[index]}
                        for index, key in enumerate(candidates)},
        "right_local_high": {key: {"progress_m": high[index],
                                   "success": high_success[index]}
                             for index, key in enumerate(candidates)}}


def test_different_best_with_five_mm_regret_is_relevant():
    result = classify_control_relevance(
        matrix([.02, .03, .01], [.03, .01, .04]), .005)
    assert result["verdict"] == "PHASE0C_CONTROL_RELEVANCE_COMPLETE"


def test_same_best_and_negligible_gap_is_not_relevant():
    result = classify_control_relevance(
        matrix([.02, .01, .005], [.021, .011, .006]), .005)
    assert result["verdict"] == "PHASE0C_CONTROL_NOT_RELEVANT"


def test_condition_specific_success_flip_is_relevant():
    result = classify_control_relevance(
        matrix([.02, .01, .005], [.02, .01, .005],
               [True, False, False], [False, False, False]), .005)
    assert result["verdict"] == "PHASE0C_CONTROL_RELEVANCE_COMPLETE"

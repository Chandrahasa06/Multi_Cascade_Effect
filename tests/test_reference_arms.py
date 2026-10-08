"""Tests for the reference arms: the real count reaches A5; one population in both halves of
each sentence; unevaluable profile features are reported, not dropped; the reference window
holds only its named days; evaluated records come from outside it; leave-one-out; no label
reaches the signal; old prompt versions stay importable."""
import inspect

import numpy as np
import pandas as pd
import pytest

from agents import reference_arms as ra
from agents.escalation_grounding import ESCALATION_FEATURES
from agents.escalation_grounding import patch_hypothesis_support as old_patch
from agents.grounding import HypothesisSupport
from agents.schema import Hypothesis, PredictedRange

ALL_COLS = list(dict.fromkeys(list(ESCALATION_FEATURES) + list(ra.EXTRA_POOL_COLUMNS) +
                              ["Source IP", "first_ts", "Destination Port", "Label"]))


def _pool(n_per_day=60, days=(0, 1, 2), seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for d in days:
        data = {c: rng.lognormal(0, 1, n_per_day) for c in ALL_COLS}
        data["Total Fwd Packets"] = rng.integers(1, 20, n_per_day).astype(float)
        data["Total Backward Packets"] = rng.integers(0, 20, n_per_day).astype(float)
        data["SYN Flag Count"] = rng.integers(0, 2, n_per_day).astype(float)
        data["Max Packet Length"] = data["Min Packet Length"] + 5.0
        df = pd.DataFrame(data)
        df["Label"] = "BENIGN"
        df["weekday_idx"] = d
        df["day"] = f"day{d}"
        df["Source IP"] = "10.0.0.1"
        df["first_ts"] = np.arange(n_per_day, dtype=np.int64) + 1000 * d
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _features(v=1.0):
    f = {c: float(v) for c in ALL_COLS}
    f["Total Fwd Packets"] = 3.0
    f["Total Backward Packets"] = 2.0
    f["SYN Flag Count"] = 1.0
    f["Max Packet Length"] = 10.0
    f["Min Packet Length"] = 5.0
    return f


def _hyp(hid, profile, benign=True):
    return Hypothesis(hypothesis_id=hid, description="d", benign=benign, prior_plausibility=0.5,
                      prediction="a checkable statement here", predicted_feature_profile=profile,
                      contradicting_claim_ids=[])


# ---------- the real count reaches A5 ----------

def test_ungrounded_count_is_shown_not_replaced_by_zero():
    pool = _pool()
    w = ra.build_window(pool, (0,), "monday")
    nb = ra.neighbourhood(w, _features(1.0))
    assert nb["ungrounded"] is True  # the count is below 30
    sentence = ra.sentence_neighbourhood(nb)
    assert f"are within [0.5x, 2x]" in sentence
    assert f"{nb['count_loo']}" in sentence


def test_old_patch_path_still_suppresses_and_is_reachable():
    # the old behaviour is kept for reproducibility: ungrounded -> close count 0
    from agents.escalation_grounding import FeatureNeighbourhood
    support = {"h1": HypothesisSupport("h1", 500, None, [], 1000)}
    nb = FeatureNeighbourhood(features_used=[], band_description="t", neighbourhood_size=29,
                              benign_population_size=1000, ungrounded=True, min_required=30)
    assert old_patch(support, nb)["h1"].close_to_observed_count == 0


def test_new_support_text_carries_the_real_closeness_count():
    pool = _pool()
    w = ra.build_window(pool, (0,), "monday")
    flow = _features(1.0)
    nb = ra.neighbourhood(w, flow)
    h = _hyp("a3_h1", [_pr("trigger_flow_duration", 0.0, 1e9)])
    sup, lines = ra.hypothesis_support(w, flow, [h], nb)
    assert sup["a3_h1"].close_to_observed_count is not None
    assert "are within 2x" in lines["a3_h1"] or "cannot be checked" in lines["a3_h1"]


def _pr(feature, lo, hi):
    return PredictedRange(feature=feature, expected_min=lo, expected_max=hi)


# ---------- one population in both halves ----------

def test_population_size_is_identical_in_both_halves_of_the_sentence():
    pool = _pool()
    w = ra.build_window(pool, (0,), "monday")
    flow = _features(1.0)
    nb = ra.neighbourhood(w, flow)
    R = nb["population_R"]
    h = _hyp("a3_h1", [_pr("trigger_flow_duration", 0.0, 1e9)])
    _, lines = ra.hypothesis_support(w, flow, [h], nb)
    assert f"of the {R} benign flows in the reference window" in lines["a3_h1"]
    assert f"Of the {R} benign flows in the reference window" in ra.sentence_neighbourhood(nb)


def test_population_R_excludes_the_flows_own_copy():
    pool = _pool()
    w = ra.build_window(pool, (0,), "monday")
    own = {c: float(w.esc[0, j]) for j, c in enumerate(ESCALATION_FEATURES)}
    flow = _features(0.0)
    flow.update(own)
    nb = ra.neighbourhood(w, flow)
    assert nb["identical"] >= 1
    assert nb["population_R"] == w.size - 1


# ---------- unmappable profile features are reported, not dropped ----------

def test_unmappable_profile_is_reported_with_its_feature_names():
    pool = _pool()
    w = ra.build_window(pool, (0,), "monday")
    flow = _features(1.0)
    nb = ra.neighbourhood(w, flow)
    h = _hyp("a3_h2", [_pr("trigger_flows_per_src", 0.0, 100.0)])
    sup, lines = ra.hypothesis_support(w, flow, [h], nb)
    assert "cannot be checked" in lines["a3_h2"]
    assert "trigger_flows_per_src" in lines["a3_h2"]
    assert sup["a3_h2"].matching_profile_count == -1  # undefined, so the zero-support cap cannot fire


def test_every_reference_vocabulary_name_is_classified():
    vocab = ["bwd_fwd_byte_ratio", "bwd_pkt_len_mean", "distinct_dst_ips_per_src", "distinct_dst_ports_per_src",
             "down_up_pkt_ratio", "flow_bytes_per_sec", "flow_duration", "flow_iat_max", "flow_iat_mean",
             "flow_iat_min", "flow_iat_regularity", "flow_pkts_per_sec", "flows_per_src", "fwd_pkt_len_mean",
             "init_win_bytes_bwd", "init_win_bytes_fwd", "no_response_flag", "pkt_len_range", "rst_ratio",
             "syn_ratio", "syn_without_synack_count"]
    assert set(vocab) <= set(ra.PROFILE_MAP)
    kinds = {ra.PROFILE_MAP[v][0] for v in vocab}
    assert kinds <= {"direct", "derived", "not_loaded", "per_source"}


# ---------- the window holds only its named days ----------

def test_window_contains_only_the_named_days():
    pool = _pool(days=(0, 1, 2))
    w = ra.build_window(pool, (0,), "monday")
    assert w.weekdays == (0,)
    assert w.size == int(((pool["weekday_idx"] == 0) & (pool["Label"] == "BENIGN")).sum())


def test_evaluated_records_from_reference_days_are_rejected():
    w = ra.build_window(_pool(days=(0, 1)), (0,), "monday")
    with pytest.raises(AssertionError):
        ra.assert_outside(w, [0])  # a Monday record cannot be evaluated against a Monday window
    ra.assert_outside(w, [1, 2])  # Tuesday onward is fine


def test_window_excludes_non_benign_rows():
    pool = _pool()
    pool.loc[0, "Label"] = "DoS Hulk"
    w = ra.build_window(pool, (0,), "monday")  # the non-benign row is excluded by the benign filter
    assert w.size == int(((pool["weekday_idx"] == 0) & (pool["Label"] == "BENIGN")).sum())


# ---------- leave-one-out ----------

def test_leave_one_out_removes_exactly_one_identical_copy():
    pool = _pool()
    w = ra.build_window(pool, (0,), "monday")
    flow = {c: float(w.esc[3, j]) for j, c in enumerate(ESCALATION_FEATURES)}
    flow.update(_features(0.0))
    flow.update({c: float(w.esc[3, j]) for j, c in enumerate(ESCALATION_FEATURES)})
    nb = ra.neighbourhood(w, flow)
    assert nb["identical"] >= 1
    assert nb["count_loo"] == nb["count_all"] - 1


# ---------- no label reaches the signal ----------

def test_neighbourhood_ignores_a_label_key_in_the_features():
    w = ra.build_window(_pool(), (0,), "monday")
    a = ra.neighbourhood(w, _features(1.0))
    b = ra.neighbourhood(w, dict(_features(1.0), Label="DoS Hulk", is_attack=True))
    assert a == b


def test_arm_signatures_take_no_label():
    for fn in (ra.neighbourhood, ra.hypothesis_support, ra.build_window):
        names = list(inspect.signature(fn).parameters)
        assert not any("label" in n.lower() or "attack" in n.lower() for n in names), (fn.__name__, names)


# ---------- prompt versions ----------

def test_v7_says_0_29_and_v6_stays_importable():
    from agents.prompts import a5_verdict_v6, a5_verdict_v7
    assert a5_verdict_v7.PROMPT_VERSION == "a5_verdict_v7"
    assert a5_verdict_v6.PROMPT_VERSION == "a5_verdict_v6"
    src7 = inspect.getsource(a5_verdict_v7)
    assert "capped at 0.29 regardless" in src7
    assert "capped at 0.3 regardless" not in src7
    assert "capped at 0.3 regardless" in inspect.getsource(a5_verdict_v6)

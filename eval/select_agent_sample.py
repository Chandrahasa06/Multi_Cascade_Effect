"""Selects 20 escalated flows from `results/escalated_flows.csv` (the
3,000-flow sampled run, 282 attacks / 2,718 benign) for the agent
pipeline. SELECTION ONLY -- no agent is run here.

Builds two deliberately separated files:

- `results/agent_input_20.jsonl`: what the agents see. Per record, the
  flow's full CICFlowMeter feature representation (read directly from the
  flow's own original CICIDS2017 CSV row -- see the module docstring
  note below on why this is NOT a live packet-level re-extraction) plus
  the escalation digest (priority, reason, matched rule id/predicted
  class where applicable, feature bins) and an opaque `record_id`. NEVER
  contains `true_label`/`is_attack` or anything derived from them --
  asserted at write time and re-checked by reading the file back.
- `results/agent_key_20.csv`: held back for scoring only (`record_id,
  true_label, is_attack, admitted_by, rule_id`).

=== Why CSV-native features, not a live CICFlowMeter re-extraction ===

`controlplane/extractor.py` already does real per-flow CICFlowMeter
extraction, but it needs the flow's own raw packets pulled back out of a
PCAP (`collect_flow_packets`). Only Monday and Friday have a PCAP in this
project; the 300k-flow sample this selection draws from spans all 8 CSV
days, so most candidate flows have no PCAP to re-extract from at all.
CICIDS2017's TrafficLabelling CSVs are themselves CICFlowMeter's own
historical output, so this module reads that same already-computed
feature vector directly from the flow's original CSV row, using the
SAME allow-list anonymisation philosophy as controlplane/extractor.py
(drop identity columns, keep Destination Port) -- but applied to the
CSV's own ~78-column schema, not the `cicflowmeter` Python package's
82-column raw output, so the exact feature count differs from that
module's "77" and is reported honestly as whatever it actually is
(see `main()`'s printed count), not padded or trimmed to match.

Run: python -m eval.select_agent_sample
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from adapters.csv_flow_adapter import load_csv, parse_timestamp
from controlplane.extractor import make_flow_id
from dataplane.escalation_policy import PRIORITY2_FEATURES, build_signature_table, compute_priority2_source_features
from eval.escalation_data import CSV_DIR, DAY_FILES, load_pool
from eval.escalation_eval import _relabel_for_build_signature_table, build_p2_fit_pool
from eval.generalization_experiments import signatures_for

OUT_DIR = Path("results")
ESCALATED_FLOWS_CSV = OUT_DIR / "escalated_flows.csv"
AGENT_INPUT_PATH = OUT_DIR / "agent_input_20.jsonl"
AGENT_KEY_PATH = OUT_DIR / "agent_key_20.csv"
REPORT_PATH = OUT_DIR / "escalation_report.md"

RANDOM_STATE = 20
N_ATTACK = 12
N_BENIGN = 8
MAX_PER_CLASS_FIRST_PASS = 2
VERDICT_THRESHOLD = 0.3  # the project's derived benign_plausibility boundary -- documented, not applied here

_DAY_TO_FILENAME = {day: fname for day, fname, _ in DAY_FILES}

#: identity columns dropped from the raw CICIDS2017 CSV row, mirroring
#: controlplane/extractor.py's _IDENTITY_COLUMNS exactly (translated to
#: this CSV's own column names) plus Label itself, which must never
#: reach the agent-facing file.
_IDENTITY_COLUMNS_TO_DROP = frozenset({
    "Flow ID", "Source IP", "Destination IP", "Source Port", "Protocol", "Timestamp", "Label",
})

PRIORITY_LABEL_TO_INT = {"P1": 1, "P2": 2, "P3": 3}
PRIORITY_REASON = {"P1": "rule_match", "P2": "uncommon_signature", "P3": "sampled"}

#: keys explicitly permitted to contain a "forbidden" substring below --
#: class_predicted is the tree's OWN prediction (not ground truth), kept
#: deliberately per the brief, and flagged separately in the report.
_ALLOWED_KEYS_WITH_FORBIDDEN_SUBSTRING = frozenset({"class_predicted"})
_FORBIDDEN_KEY_SUBSTRINGS = ("label", "is_attack", "ground_truth", "true_")


# --------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------- #

def select_attack_flows(attacks: pd.DataFrame, rng: np.random.Generator, n: int = N_ATTACK) -> pd.DataFrame:
    """At most 2 per class, in ALPHABETICAL class order (a neutral order
    that doesn't itself favor big classes, unlike sorting by size first),
    until n is filled; if classes run out before n is reached, tops up
    from the largest remaining classes. Sample without replacement
    throughout -- each class's own flows are shuffled once before being
    drawn from, so "at most 2" isn't always the same 2 flows."""
    classes = sorted(attacks["true_label"].unique())
    remaining: Dict[str, List[int]] = {}
    for c in classes:
        idx = attacks.index[attacks["true_label"] == c].to_numpy()
        remaining[c] = idx[rng.permutation(len(idx))].tolist()

    picked: List[int] = []
    for c in classes:
        take = min(MAX_PER_CLASS_FIRST_PASS, len(remaining[c]), n - len(picked))
        picked.extend(remaining[c][:take])
        remaining[c] = remaining[c][take:]
        if len(picked) >= n:
            break

    if len(picked) < n:
        order = sorted(remaining, key=lambda c: len(remaining[c]), reverse=True)
        for c in order:
            if len(picked) >= n:
                break
            take = min(len(remaining[c]), n - len(picked))
            picked.extend(remaining[c][:take])
            remaining[c] = remaining[c][take:]

    assert len(picked) == n, f"could only fill {len(picked)}/{n} attack slots -- not enough attack flows available"
    return attacks.loc[picked]


def select_benign_flows(benign: pd.DataFrame, rng: np.random.Generator, n: int = N_BENIGN) -> pd.DataFrame:
    idx = benign.index.to_numpy()
    chosen = idx[rng.permutation(len(idx))[:n]]
    return benign.loc[chosen]


def reconstruct_batch1_flow_ids(all_flows: pd.DataFrame) -> set:
    """`results/agent_key_20.csv` only carries the fresh, opaque
    `record_id` this module assigns on output (agents.controlplane's
    "never a hash/derivation of the real identity" convention) -- it has
    no column linking back to `escalated_flows.csv`'s own `flow_id`, so
    "which 20 flows did batch 1 actually draw" isn't a lookup, it's a
    replay: select_attack_flows/select_benign_flows are pure functions of
    (all_flows, seed), and RANDOM_STATE/N_ATTACK/N_BENIGN haven't changed
    since batch 1 ran, so re-running them with the SAME seed reproduces
    the SAME 20 flow_ids exactly. Batch 2 uses this set only to EXCLUDE
    those rows before drawing its own sample with a different seed --
    it never re-selects them."""
    rng = np.random.default_rng(RANDOM_STATE)
    attack_sel = select_attack_flows(all_flows[all_flows["is_attack"]], rng, N_ATTACK)
    benign_sel = select_benign_flows(all_flows[~all_flows["is_attack"]], rng, N_BENIGN)
    return set(attack_sel["flow_id"]) | set(benign_sel["flow_id"])


# --------------------------------------------------------------------- #
# Full CICFlowMeter-style feature extraction (from the original CSV row)
# --------------------------------------------------------------------- #

_raw_day_cache: Dict[str, pd.DataFrame] = {}


def _load_aligned_raw_day(day: str) -> pd.DataFrame:
    """Reproduces eval.escalation_data._clean_one_day's row alignment
    (dropna Label -> reset_index; dropna unparseable Timestamp ->
    reset_index) WITHOUT that module's narrow column subset, so `row_idx`
    (as encoded in `flow_uid`) indexes the same physical row here as it
    did when the pool/flow_uid were originally built, while keeping every
    raw CICFlowMeter column."""
    if day in _raw_day_cache:
        return _raw_day_cache[day]
    raw = load_csv(CSV_DIR / _DAY_TO_FILENAME[day])
    raw = raw.dropna(subset=["Label"]).reset_index(drop=True)
    ts = raw["Timestamp"].apply(parse_timestamp)
    first_ts = ts.apply(lambda t: t[0] if t else np.nan)
    if first_ts.isna().any():
        raw = raw.loc[first_ts.notna()].reset_index(drop=True)
    _raw_day_cache[day] = raw
    return raw


def parse_flow_uid(flow_uid: str) -> Tuple[str, int, str]:
    day, row_idx_s, orig_flow_id = flow_uid.split("::", 2)
    return day, int(row_idx_s), orig_flow_id


def full_cicflowmeter_features(flow_uid: str) -> Dict[str, float]:
    day, row_idx, _ = parse_flow_uid(flow_uid)
    row = _load_aligned_raw_day(day).iloc[row_idx]
    features: Dict[str, float] = {}
    for col, val in row.items():
        if col in _IDENTITY_COLUMNS_TO_DROP:
            continue
        try:
            features[col] = float(val)
        except (TypeError, ValueError):
            features[col] = float("nan")
    return features


# --------------------------------------------------------------------- #
# Escalation digest (priority, reason, matched rule, feature bins)
# --------------------------------------------------------------------- #

def fit_signature_table_for_bins():
    """Same fit as the main per-day analysis (eval.escalation_eval.main):
    Monday+Tuesday chronological first-half BENIGN rows -- reused here
    only to compute each selected flow's own feature-bin tuple for its
    digest, not to re-run any part of the policy."""
    pool = load_pool()
    pool = compute_priority2_source_features(pool)
    fit_benign = build_p2_fit_pool(pool)
    table = build_signature_table(_relabel_for_build_signature_table(fit_benign), n_bins=6, floor=5)
    return pool, table


def compute_feature_bins(selected_flow_ids: List[str], pool: pd.DataFrame, signature_table) -> Dict[str, tuple]:
    mask = pool["flow_uid"].isin(set(selected_flow_ids))
    sub = pool.loc[mask, ["flow_uid"] + list(PRIORITY2_FEATURES)].set_index("flow_uid")
    sub = sub.loc[selected_flow_ids]  # preserve caller's order
    sigs = signatures_for(sub.reset_index(drop=True), list(PRIORITY2_FEATURES), signature_table.bin_edges)
    return dict(zip(selected_flow_ids, sigs))


def build_escalation_digest(row: pd.Series, bins_tuple: tuple) -> dict:
    admitted_by = row["admitted_by"]
    is_p1 = admitted_by == "P1"
    rule_id = row["rule_id"]
    class_predicted = row["class_predicted"]
    return {
        "priority": PRIORITY_LABEL_TO_INT[admitted_by],
        "reason": PRIORITY_REASON[admitted_by],
        "matched_rule_id": int(rule_id) if (is_p1 and not pd.isna(rule_id)) else None,
        "class_predicted": (class_predicted if (is_p1 and isinstance(class_predicted, str) and class_predicted) else None),
        "feature_bins": {f: int(b) for f, b in zip(PRIORITY2_FEATURES, bins_tuple)},
    }


# --------------------------------------------------------------------- #
# Ground-truth isolation
# --------------------------------------------------------------------- #

def assert_no_ground_truth(record: dict, _path: str = "") -> None:
    """Recursively fails if any key looks ground-truth-shaped, with one
    explicit, documented exception (`class_predicted` -- the tree's own
    prediction, not the label). This is a KEY-based check: `is_attack`/
    `true_label` are never computed into any VALUE in this module either
    (by construction -- see build_records), so guarding the keys that
    could carry them is the concrete, testable form of "no ground truth
    reaches this file"."""
    if isinstance(record, dict):
        for k, v in record.items():
            if isinstance(k, str):
                kl = k.lower()
                if k not in _ALLOWED_KEYS_WITH_FORBIDDEN_SUBSTRING and any(s in kl for s in _FORBIDDEN_KEY_SUBSTRINGS):
                    raise AssertionError(f"ground-truth-shaped key found at {_path}.{k!r}")
            assert_no_ground_truth(v, f"{_path}.{k}")
    elif isinstance(record, (list, tuple)):
        for i, v in enumerate(record):
            assert_no_ground_truth(v, f"{_path}[{i}]")


# --------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------- #

def build_records(selected: pd.DataFrame, features_by_id: Dict[str, dict], bins_by_id: Dict[str, tuple],
                   rng: np.random.Generator) -> Tuple[List[dict], pd.DataFrame]:
    """Shuffles the final 20's order (else "first 12 = attack, last 8 =
    benign" would itself leak ground truth by position) before assigning
    opaque record_ids."""
    order = rng.permutation(len(selected))
    shuffled = selected.iloc[order].reset_index(drop=True)

    records: List[dict] = []
    key_rows: List[dict] = []
    for _, row in shuffled.iterrows():
        fid = row["flow_id"]
        record_id = make_flow_id()
        rec = {
            "record_id": record_id,
            "features": features_by_id[fid],
            "escalation": build_escalation_digest(row, bins_by_id[fid]),
        }
        assert_no_ground_truth(rec)
        records.append(rec)
        key_rows.append({
            "record_id": record_id, "true_label": row["true_label"], "is_attack": bool(row["is_attack"]),
            "admitted_by": row["admitted_by"],
            "rule_id": (int(row["rule_id"]) if not pd.isna(row["rule_id"]) else ""),
        })
    return records, pd.DataFrame(key_rows)


def write_and_verify_jsonl(records: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    # load back and re-check -- an independent pass over what's actually
    # on disk, not a re-check of the in-memory objects already validated.
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            loaded = json.loads(line)
            assert_no_ground_truth(loaded, _path=f"line{line_no}")


# --------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------- #

_REPORT_SECTION_HEADER = "\n# Selected sample for the agent pipeline (n=20)\n"


def build_report_section(all_flows: pd.DataFrame, selected: pd.DataFrame, key_df: pd.DataFrame,
                          records: List[dict]) -> str:
    lines: List[str] = [_REPORT_SECTION_HEADER]
    lines.append(
        f"Drawn from `results/escalated_flows.csv` (3,000 escalated flows: 282 attacks, 2,718 benign). "
        f"`random_state={RANDOM_STATE}`, reported here for reproducibility. Selection only -- no agent "
        "was run for this task.\n"
    )

    n_total = len(all_flows)
    n_attack_total = int(all_flows["is_attack"].sum())
    n_benign_total = n_total - n_attack_total
    lines.append(
        "**The 40% attack rate in this 20-record sample is NOT the natural rate.** Escalated flows are "
        f"{n_attack_total}/{n_total} = {n_attack_total / n_total:.1%} attacks; a natural, unweighted draw "
        "of 20 would carry about 2 attacks and 18 benign. This sample deliberately over-represents attacks "
        "(12 of 20) so the 7 rare attack classes have a chance to appear at all -- it measures whether the "
        "agents can DISCRIMINATE attack-shaped records from benign ones, not how the pipeline would perform "
        "at deployment base rates. Any accuracy figure computed from these 20 records must carry that "
        "caveat every time it's quoted.\n"
    )

    attack_sel = selected[selected["is_attack"]]
    class_counts = attack_sel["true_label"].value_counts()
    lines.append("## Attack class breakdown of the 12\n")
    lines.append("\n".join(f"- {lbl}: {n}" for lbl, n in class_counts.items()) + "\n")
    n_classes_present_pool = all_flows.loc[all_flows["is_attack"], "true_label"].nunique()
    lines.append(
        f"\n{len(class_counts)} of {n_classes_present_pool} attack classes present in the 282-attack pool "
        "are represented here (at most 2 per class, alphabetical first pass, largest-class top-up if a "
        "pass doesn't fill the quota -- see `eval/select_agent_sample.py::select_attack_flows`).\n"
    )

    priority_counts = selected["admitted_by"].value_counts()
    attack_priority_counts = attack_sel["admitted_by"].value_counts()
    lines.append("## Priority breakdown of all 20\n")
    lines.append("\n".join(f"- {p}: {int(priority_counts.get(p, 0))}" for p in ("P1", "P2", "P3")) + "\n")
    non_p1_attack_classes = sorted(attack_sel.loc[attack_sel["admitted_by"] != "P1", "true_label"].unique())
    non_p1_desc = ", ".join(non_p1_attack_classes) if non_p1_attack_classes else "none -- all 12 came via P1"
    lines.append(
        f"\nWithin the 12 attacks specifically: "
        + ", ".join(f"{p}={int(attack_priority_counts.get(p, 0))}" for p in ("P1", "P2", "P3"))
        + f". **The 12 are P1-heavy** ({int(attack_priority_counts.get('P1', 0))}/12 admitted by Priority 1) "
        "-- this mirrors the source pool, where 259/282 (91.8%) of escalated attacks were caught by P1's "
        "rules, not P2 or P3. A consequence for interpreting agent results: most attack records will carry "
        "a `class_predicted` hint (see below), so this sample is better read as a test of whether agents "
        "corroborate or override a rule's own guess, not as a blind detection test from raw features alone "
        f"-- the P2/P3-admitted attacks in this specific 20 ({non_p1_desc}) are the closest thing here to "
        "a blind test.\n"
    )

    n_with_hint = sum(1 for r in records if r["escalation"].get("class_predicted"))
    hinted_ids = [r["record_id"] for r in records if r["escalation"].get("class_predicted")]
    lines.append("## `class_predicted` hint\n")
    lines.append(
        f"**{n_with_hint} of 20 records carry a `class_predicted` field** (present only for Priority-1-"
        "admitted records, per the brief). This is the TREE'S OWN PREDICTION from its matched rule, not "
        "the ground-truth label -- but it stays in the record because the brief specifies it should, "
        "while flagging plainly that for these records the agent is effectively being told \"a rule "
        "thinks this is <class>\" before it ever looks at the raw features. Any strong agreement between "
        "agent output and `class_predicted` on these records is not independent corroboration; it may "
        "just be the agent reading the hint.\n"
    )
    lines.append(f"Record IDs carrying the hint: {', '.join(hinted_ids)}\n")

    lines.append("## The 20 records\n")
    table_rows = [
        "| record_id | priority | admitted_by | attack class | has class_predicted hint |",
        "|---|---|---|---|---|",
    ]
    key_by_id = key_df.set_index("record_id")
    for r in records:
        rid = r["record_id"]
        k = key_by_id.loc[rid]
        attack_class = k["true_label"] if k["is_attack"] else ""
        admitted_by = {1: "P1", 2: "P2", 3: "P3"}[r["escalation"]["priority"]]
        has_hint = "yes" if r["escalation"].get("class_predicted") else ""
        table_rows.append(f"| {rid} | {r['escalation']['priority']} | {admitted_by} | {attack_class} | {has_hint} |")
    lines.append("\n".join(table_rows) + "\n")

    lines.append(
        "\n## Scoring methodology for the run that follows (not computed here -- no agent has run yet)\n"
    )
    lines.append(
        f"Agents will output a continuous `benign_plausibility` in [0,1], not a label. Scoring thresholds "
        f"it at the project's derived verdict boundary, **{VERDICT_THRESHOLD}** (not tuned to these 20 "
        "records, and not to be tuned to maximise accuracy on them after the fact). Once real agent output "
        "exists, report:\n"
        "- the confusion matrix (predicted attack/benign at the 0.3 threshold vs. `agent_key_20.csv`'s "
        "`is_attack`), with counts, not just percentages, and a confidence interval on any derived rate "
        "(n=20 is small enough that this matters a great deal);\n"
        "- `benign_plausibility`'s distribution for the true-attack group and the true-benign group "
        "reported SEPARATELY (e.g. min/median/max or a small histogram for each) -- the separation between "
        "the two distributions is more informative at this n than accuracy at any single threshold, and "
        "should be reported even if the 0.3 threshold happens to land badly;\n"
        "- explicit acknowledgement that a P1-heavy attack set (see above) makes any confusion-matrix cell "
        "hard to attribute cleanly to \"the agent detected the attack\" versus \"the agent read the hint\".\n"
    )

    lines.append("## Honesty notes\n")
    lines.append(
        "- n=20 supports almost nothing on its own -- every count above is reported as a count, not a "
        "percentage, except where a percentage is explicitly qualified by its denominator.\n"
        "- Records were not reselected after seeing any result; this selection runs once, from a fixed "
        f"`random_state={RANDOM_STATE}`, and the two output files are final.\n"
        "- Every number in this section states its own n.\n"
    )
    return "\n".join(lines)


def upsert_report_section(section_text: str) -> None:
    """Idempotent: replaces a previously-appended copy of this exact
    section (if this script has been run before) rather than duplicating
    it, without disturbing anything eval/escalation_eval.py's own
    write_report wrote earlier in the same file."""
    existing = REPORT_PATH.read_text(encoding="utf-8") if REPORT_PATH.exists() else ""
    marker = _REPORT_SECTION_HEADER.strip()
    idx = existing.find(marker)
    base = existing[:idx].rstrip("\n") if idx != -1 else existing.rstrip("\n")
    REPORT_PATH.write_text(base + "\n" + section_text + "\n", encoding="utf-8")


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #

def main() -> int:
    print(f"reading {ESCALATED_FLOWS_CSV} ...")
    all_flows = pd.read_csv(ESCALATED_FLOWS_CSV)
    print(f"total {len(all_flows):,}: {int(all_flows['is_attack'].sum())} attacks, "
          f"{int((~all_flows['is_attack']).sum())} benign")

    rng = np.random.default_rng(RANDOM_STATE)
    attack_sel = select_attack_flows(all_flows[all_flows["is_attack"]], rng, N_ATTACK)
    benign_sel = select_benign_flows(all_flows[~all_flows["is_attack"]], rng, N_BENIGN)
    selected = pd.concat([attack_sel, benign_sel], ignore_index=True)
    assert len(selected) == N_ATTACK + N_BENIGN
    print(f"selected {len(attack_sel)} attacks (classes: "
          f"{dict(attack_sel['true_label'].value_counts())}) + {len(benign_sel)} benign")

    print("fitting signature table (reused, not re-fit) for feature-bin digests ...")
    pool, signature_table = fit_signature_table_for_bins()

    flow_ids = selected["flow_id"].tolist()
    pool_rows = pool.loc[pool["flow_uid"].isin(set(flow_ids)), ["flow_uid", "Label"]].set_index("flow_uid")
    for _, row in selected.iterrows():
        assert pool_rows.loc[row["flow_id"], "Label"] == row["true_label"], (
            f"Label mismatch for {row['flow_id']} -- flow_uid join is unreliable"
        )

    bins_by_id = compute_feature_bins(flow_ids, pool, signature_table)

    print("extracting full CICFlowMeter-style features from each flow's original CSV row ...")
    features_by_id = {fid: full_cicflowmeter_features(fid) for fid in flow_ids}
    n_feat = len(next(iter(features_by_id.values())))
    print(f"{n_feat} feature columns per record (see module docstring for why this isn't exactly 77)")

    rng2 = np.random.default_rng(RANDOM_STATE + 1)  # separate stream for output-order shuffling
    records, key_df = build_records(selected, features_by_id, bins_by_id, rng2)

    write_and_verify_jsonl(records, AGENT_INPUT_PATH)
    print(f"wrote {AGENT_INPUT_PATH} ({len(records)} records) -- no ground truth, verified by reading back")

    key_df.to_csv(AGENT_KEY_PATH, index=False)
    print(f"wrote {AGENT_KEY_PATH} ({len(key_df)} rows)")

    section = build_report_section(all_flows, selected, key_df, records)
    upsert_report_section(section)
    print(f"updated {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

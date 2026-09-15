"""Does the 5-feature selector (results/seven_feature_selector_report.md)
generalize to attack types it was never examined against? Feature
selection (which 5 of 18 to keep) was done by looking at Friday's own
crossing rates -- Friday alone can't validate it. This tests the SAME,
UNCHANGED fitted config (Monday PCAP benign, chronological split, p99.5)
against Tuesday/Wednesday/Thursday (CSV path, real CICIDS2017 attack
labels the selector has never been tuned against) and Friday (PCAP) side
by side.

Two caveats stated explicitly, not glossed over:
  - Tuesday-Thursday are CSV-adapter-derived (synthetic packet spacing,
    minute-resolution timestamps on Tue-Thu vs Monday's own second
    resolution); Friday here is real PCAP. None of these 5 features are
    IAT-based, so timestamp coarseness should matter less than it did for
    the project's earlier (now-fixed) percentile-saturation bugs -- but
    it's a real adapter difference, stated, not assumed irrelevant.
  - The fitted thresholds come from Monday PCAP (adapter2); they're being
    applied here to Tuesday-Thursday's CSV-adapter (adapter1) features
    unchanged, per instruction. This is a second, distinct cross-adapter
    gap on top of the cross-day one this experiment is actually testing
    for -- flagged so a weak Tue-Thu result isn't misread as pure
    feature-selection overfitting when part of it could be adapter
    mismatch.

No API calls, no re-simulation -- reads already-cached CSV-path feature
parquets (results/cache/*__eval__adapter1__features4.parquet, EVAL-mode:
unambiguous 1:1 label per flow) directly, and the same PCAP loader used
throughout this session for Friday.

Run: python -m eval.five_feature_cross_day
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

from eval.reduced_selector_report import load_data as load_pcap_data
from eval.run_pcap_sweep import load_friday_pcap
from eval.seven_feature_selector import COUNT_5, RATIO_5, escalate_subset, fit_subset

OUT_MD = Path("results/five_feature_cross_day_report.md")
CACHE_DIR = Path("results/cache")

PERCENTILE = 99.5

# (day label, list of (cache_basename, attack classes present, matcher) ) --
# matcher is a substring/exact rule since Thursday's labels carry mojibake
# from the original CICIDS2017 release (see module docstring / STATUS.md).
TUESDAY_CLASSES = ["FTP-Patator", "SSH-Patator"]
WEDNESDAY_CLASSES = ["DoS Hulk", "DoS GoldenEye", "DoS Slowhttptest", "DoS slowloris", "Heartbleed"]
THURSDAY_CLASSES = ["Web Attack: Brute Force", "Web Attack: XSS", "Web Attack: SQL Injection", "Infiltration"]
FRIDAY_CLASSES = ["Bot", "PortScan", "DDoS"]

TINY_CLASS_FLOOR = 50  # below this raw n, flag recall as statistically unreliable


def load_csv_day(*basenames: str) -> pd.DataFrame:
    frames = []
    for name in basenames:
        path = CACHE_DIR / f"{name}__eval__adapter1__features4.parquet"
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def normalize_thursday_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Thursday's Web Attack labels carry a mojibake separator (original
    CICIDS2017 release encoding issue, already noted in STATUS.md's
    "Known data quirks") -- normalize by matching content, not the exact
    byte sequence, so this script isn't silently broken by which specific
    replacement character ended up in the cached parquet."""
    df = df.copy()
    label = df["label"].astype(str)
    is_brute = label.str.contains("Brute Force", case=False, na=False)
    is_xss = label.str.contains("XSS", case=False, na=False)
    is_sqli = label.str.contains("Sql Injection", case=False, na=False) | label.str.contains("SQL Injection", case=False, na=False)
    df.loc[is_brute, "label"] = "Web Attack: Brute Force"
    df.loc[is_xss, "label"] = "Web Attack: XSS"
    df.loc[is_sqli, "label"] = "Web Attack: SQL Injection"
    return df


def day_report(name: str, df: pd.DataFrame, attack_classes: Sequence[str], escalated: pd.Series) -> dict:
    is_benign = df["label"] == "BENIGN"
    n_total = len(df)
    n_benign = int(is_benign.sum())
    n_attack = n_total - n_benign
    n_escalated = int(escalated.sum())
    n_escalated_benign = int((escalated & is_benign).sum())
    n_escalated_attack = n_escalated - n_escalated_benign

    per_class = {}
    for cls in attack_classes:
        mask = df["label"] == cls
        n_cls = int(mask.sum())
        n_cls_escalated = int((escalated & mask).sum())
        recall = (n_cls_escalated / n_cls) if n_cls else None
        per_class[cls] = {
            "n": n_cls, "escalated": n_cls_escalated, "recall": recall,
            "unreliable": n_cls > 0 and n_cls < TINY_CLASS_FLOOR,
        }

    return {
        "day": name, "n_total": n_total, "n_benign": n_benign, "n_attack": n_attack,
        "n_escalated": n_escalated, "n_escalated_benign": n_escalated_benign, "n_escalated_attack": n_escalated_attack,
        "escalation_rate": n_escalated / n_total if n_total else None,
        "benign_fpr": n_escalated_benign / n_benign if n_benign else None,
        "per_class": per_class,
    }


def render_day(r: dict) -> List[str]:
    lines = [f"## {r['day']}\n"]
    lines.append(f"Total flows: **{r['n_total']:,}** (benign={r['n_benign']:,}, attack={r['n_attack']:,})\n")
    lines.append(f"Total escalated: **{r['n_escalated']:,}** ({r['escalation_rate']:.3%} of all flows)")
    lines.append(f"  - escalated BENIGN: {r['n_escalated_benign']:,} (benign false-alarm rate: **{r['benign_fpr']:.3%}**)")
    lines.append(f"  - escalated ATTACK: {r['n_escalated_attack']:,}\n")
    lines.append("| attack class | n | escalated | recall |")
    lines.append("|---|---|---|---|")
    for cls, c in r["per_class"].items():
        if c["n"] == 0:
            lines.append(f"| {cls} | 0 | -- | n/a (not present) |")
            continue
        recall_str = f"{c['recall']:.1%}"
        if c["unreliable"]:
            recall_str += f" (n={c['n']} -- **statistically unreliable, raw count only**)"
        lines.append(f"| {cls} | {c['n']:,} | {c['escalated']:,} | {recall_str} |")
    lines.append("")
    return lines


def main() -> int:
    print("fitting 5-feature config on Monday PCAP benign (unchanged from prior reports) ...")
    monday_fit, _, _, _ = load_pcap_data()
    fit5 = fit_subset(monday_fit, COUNT_5, RATIO_5, PERCENTILE)
    missing = [f for f in list(COUNT_5) + list(RATIO_5) if f not in fit5.config.thresholds]
    print(f"active features: {len(fit5.config.thresholds)} (missing/saturated: {missing or 'none'})")
    print()

    print("loading Tuesday/Wednesday/Thursday (CSV path, cached) and Friday (PCAP, cached) ...")
    tuesday_df = load_csv_day("tuesday")
    wednesday_df = load_csv_day("wednesday")
    thursday_df = normalize_thursday_labels(load_csv_day("thursday_morning_webattacks", "thursday_afternoon_infiltration"))
    friday_df, _ = load_friday_pcap()

    days = [
        ("Tuesday (CSV, FTP-Patator/SSH-Patator)", tuesday_df, TUESDAY_CLASSES),
        ("Wednesday (CSV, DoS variants + Heartbleed)", wednesday_df, WEDNESDAY_CLASSES),
        ("Thursday (CSV, Web Attacks + Infiltration)", thursday_df, THURSDAY_CLASSES),
        ("Friday (PCAP, Bot/PortScan/DDoS)", friday_df, FRIDAY_CLASSES),
    ]

    lines: List[str] = []
    lines.append("# 5-feature selector: cross-day generalization test\n")
    lines.append(
        "Tests the SAME, unchanged 5-feature config (COUNT = `flows_per_src`, "
        "`distinct_dst_ports_per_src`, `syn_without_synack_count`; RATIO = `syn_ratio`, "
        "`bwd_fwd_byte_ratio`; rule: >=1 COUNT AND >=1 RATIO; fit on Monday PCAP benign, "
        f"chronological split, p{PERCENTILE}) -- the config selected by looking only at "
        "Friday -- against Tuesday, Wednesday, Thursday (attack types it was never examined "
        "against) and Friday, each day reported separately. No API calls, no re-simulation.\n"
    )
    lines.append(
        f"**Caveats, stated directly:** Tuesday-Thursday are CSV-adapter-derived (synthetic "
        f"packet spacing, minute-resolution timestamps); Friday here is real PCAP. None of "
        f"these 5 features are IAT-based, so timestamp coarseness should matter less than it "
        f"did for this project's earlier (fixed) saturation bugs -- but it's a real adapter "
        f"difference. Separately, and more structurally: the fitted thresholds come from "
        f"Monday **PCAP** (adapter2) and are applied here to Tuesday-Thursday's **CSV** "
        f"(adapter1) features unchanged, per instruction -- a cross-adapter gap layered on "
        f"top of the cross-day question this test is actually asking. A weak Tuesday-Thursday "
        f"result could reflect either overfitting to Friday OR this adapter mismatch; the two "
        f"aren't separated by this test alone.\n"
    )
    lines.append(f"Active thresholds fit: {len(fit5.config.thresholds)} of 5 "
                 f"(missing/saturated: {missing or 'none'}).\n")

    for name, df, classes in days:
        escalated, _, _ = escalate_subset(df, fit5, COUNT_5, RATIO_5)
        r = day_report(name, df, classes, escalated)
        lines.extend(render_day(r))
        print(f"{name}: n={r['n_total']:,} escalated={r['n_escalated']:,} "
              f"({r['escalation_rate']:.3%})  benign_fpr={r['benign_fpr']:.3%}")
        for cls, c in r["per_class"].items():
            if c["n"]:
                print(f"    {cls}: n={c['n']:,} recall={c['recall']:.1%}" + (" [unreliable]" if c["unreliable"] else ""))

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nwrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

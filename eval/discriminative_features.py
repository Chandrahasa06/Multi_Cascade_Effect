"""Identify the top 10 most discriminative features for each attack class in CICIDS2017.

For every attack class, a one-vs-benign binary problem (BENIGN vs. that attack
only, all other attacks excluded) is built and ranked using four independent
feature-importance methods whose ranks are then aggregated.

Required packages (pip install ...):
    numpy pandas scikit-learn matplotlib shap (optional but recommended)

Usage:
    python discriminative_features.py --data-dir path/to/MachineLearningCVE --out-dir ./output
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

try:
    import shap  # type: ignore

    SHAP_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    SHAP_AVAILABLE = False

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    MATPLOTLIB_AVAILABLE = False

RANDOM_SEED = 42
MIN_SAMPLES = 30
TOP_N = 10
MAX_BENIGN_MULTIPLIER = 10
MAX_BENIGN_CAP = 50_000
SHAP_SAMPLE_SIZE = 2000

LEAKAGE_COLUMNS = [
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Source Port",
    "Timestamp",
    "Fwd Header Length.1",
]

# Maps the mojibake / inconsistent raw labels found in the CICIDS2017 CSVs to
# a canonical set of attack-class names.
LABEL_CANONICAL_MAP = {
    "BENIGN": "BENIGN",
    "Bot": "Bot",
    "DDoS": "DDoS",
    "DoS GoldenEye": "DoS GoldenEye",
    "DoS Hulk": "DoS Hulk",
    "DoS Slowhttptest": "DoS Slowhttptest",
    "DoS slowloris": "DoS slowloris",
    "FTP-Patator": "FTP-Patator",
    "Heartbleed": "Heartbleed",
    "Infiltration": "Infiltration",
    "PortScan": "PortScan",
    "SSH-Patator": "SSH-Patator",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("discriminative_features")


def load_data(data_dir: Path) -> pd.DataFrame:
    """Load and concatenate all CICIDS2017 CSVs found in ``data_dir``.

    Numeric columns are downcast to float32 after concatenation to keep the
    combined ~2.8M-row DataFrame manageable in memory.
    """
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    logger.info("Found %d CSV files in %s", len(csv_files), data_dir)
    frames = []
    for csv_path in csv_files:
        logger.info("Reading %s", csv_path.name)
        df = pd.read_csv(csv_path, low_memory=False, encoding="utf-8")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Combined shape before cleaning: %s", combined.shape)

    numeric_cols = combined.select_dtypes(include=["int64", "float64"]).columns
    combined[numeric_cols] = combined[numeric_cols].apply(pd.to_numeric, downcast="float")
    combined[numeric_cols] = combined[numeric_cols].astype("float32")

    return combined


def _fix_label(raw_label: str) -> str:
    """Normalize a single raw label string to its canonical attack-class name."""
    label = raw_label.strip()
    if label in LABEL_CANONICAL_MAP:
        return LABEL_CANONICAL_MAP[label]

    # The dataset's Web Attack labels were corrupted during encoding: the
    # separator between "Web Attack" and the sub-type became a mojibake /
    # replacement character (e.g. "Web Attack � Brute Force"). Recover
    # the sub-type by stripping everything that isn't "Web Attack" + a known
    # suffix.
    if label.lower().startswith("web attack"):
        suffix = re.sub(r"^web attack\W*", "", label, flags=re.IGNORECASE).strip()
        suffix_lower = suffix.lower()
        if "brute" in suffix_lower:
            return "Web Attack - Brute Force"
        if "sql" in suffix_lower:
            return "Web Attack - SQL Injection"
        if "xss" in suffix_lower:
            return "Web Attack - XSS"
        return "Web Attack - Other"

    return label


def clean_data(df: pd.DataFrame, drop_dport: bool = True) -> pd.DataFrame:
    """Clean the combined CICIDS2017 DataFrame.

    Steps: normalize column names, fix mojibake labels, replace inf with NaN
    and drop NaN rows, drop exact duplicates, drop leakage/identifier
    columns, optionally drop Destination Port, and drop zero-variance
    columns.
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    logger.info("Normalized %d column names", len(df.columns))

    if "Label" not in df.columns:
        raise KeyError("Expected a 'Label' column after normalizing column names")
    df["Label"] = df["Label"].astype(str).map(_fix_label)
    logger.info("Canonical label distribution:\n%s", df["Label"].value_counts().to_string())

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n_inf = np.isinf(df[numeric_cols].to_numpy(dtype="float64", na_value=0.0)).sum()
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    logger.info("Replaced %d inf/-inf values with NaN", n_inf)

    n_before = len(df)
    df = df.dropna(subset=numeric_cols)
    logger.info("Dropped %d rows containing NaN in feature columns", n_before - len(df))

    n_before = len(df)
    df = df.drop_duplicates()
    logger.info("Dropped %d exact duplicate rows", n_before - len(df))

    cols_to_drop = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logger.info("Dropped identifier/leakage columns: %s", cols_to_drop)

    if drop_dport and "Destination Port" in df.columns:
        # Destination Port is highly leaky in CICIDS2017: attack traffic is
        # concentrated on a handful of specific ports (e.g. 21/22/80/443),
        # so a model can trivially "detect" attacks by memorizing ports
        # rather than learning genuine traffic-behavior signal.
        df = df.drop(columns=["Destination Port"])
        logger.info("Dropped Destination Port column (--drop-dport, leaky feature)")

    feature_cols = df.select_dtypes(include=[np.number]).columns
    variances = df[feature_cols].var()
    constant_cols = variances[variances == 0].index.tolist()
    if constant_cols:
        df = df.drop(columns=constant_cols)
        logger.info("Dropped %d zero-variance columns: %s", len(constant_cols), constant_cols)

    df = df.reset_index(drop=True)
    logger.info("Final cleaned shape: %s", df.shape)
    return df


def get_attack_classes(df: pd.DataFrame) -> list[str]:
    """Return the sorted list of non-BENIGN attack class labels present in df."""
    classes = sorted(c for c in df["Label"].unique() if c != "BENIGN")
    logger.info("Found %d attack classes: %s", len(classes), classes)
    return classes


def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cliff's delta effect size between samples x and y.

    Uses a sort-based O(n log n) approach rather than the naive O(n*m)
    pairwise comparison, which is important given samples of thousands of
    points.
    """
    x_sorted = np.sort(x)
    n_x, n_y = len(x_sorted), len(y)
    if n_x == 0 or n_y == 0:
        return np.nan
    # For each y value, count how many x values are less-than / greater-than it.
    less = np.searchsorted(x_sorted, y, side="left")
    greater = n_x - np.searchsorted(x_sorted, y, side="right")
    more = np.sum(greater) - np.sum(less)
    return float(more) / (n_x * n_y)


def compute_importances(
    df: pd.DataFrame,
    attack_name: str,
    feature_cols: list[str],
) -> Optional[dict]:
    """Build the one-vs-benign problem for ``attack_name`` and compute all
    four importance signals plus model-quality and interpretability stats.

    Returns None if the attack has fewer than MIN_SAMPLES rows.
    """
    attack_df = df[df["Label"] == attack_name]
    n_attack = len(attack_df)

    if n_attack < MIN_SAMPLES:
        logger.warning(
            "Skipping %s: only %d samples (< MIN_SAMPLES=%d)", attack_name, n_attack, MIN_SAMPLES
        )
        return None

    low_confidence = n_attack < 100

    n_benign_target = min(n_attack * MAX_BENIGN_MULTIPLIER, MAX_BENIGN_CAP)
    benign_df = df[df["Label"] == "BENIGN"]
    n_benign_used = min(n_benign_target, len(benign_df))
    benign_sample = benign_df.sample(n=n_benign_used, random_state=RANDOM_SEED)

    subset = pd.concat([attack_df, benign_sample], ignore_index=True)
    X = subset[feature_cols].to_numpy(dtype="float64")
    y = (subset["Label"] == attack_name).astype(int).to_numpy()

    logger.info(
        "[%s] n_attack=%d n_benign_used=%d low_confidence=%s",
        attack_name,
        n_attack,
        n_benign_used,
        low_confidence,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_SEED
    )

    # 1. Mutual information on the training set.
    mi_scores = mutual_info_classif(X_train, y_train, random_state=RANDOM_SEED)

    # 2. Random forest impurity importance.
    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    rf.fit(X_train, y_train)
    rf_importances = rf.feature_importances_

    # Model quality on the held-out test set.
    y_proba = rf.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    # 3. Permutation importance on the held-out test set (most trustworthy).
    perm_result = permutation_importance(
        rf,
        X_test,
        y_test,
        n_repeats=10,
        scoring="roc_auc",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    perm_importances = perm_result.importances_mean

    # 4. SHAP TreeExplainer mean |SHAP value| on a sample of test rows.
    shap_values_mean = np.full(len(feature_cols), np.nan)
    if SHAP_AVAILABLE:
        try:
            rng = np.random.RandomState(RANDOM_SEED)
            n_shap = min(SHAP_SAMPLE_SIZE, len(X_test))
            idx = rng.choice(len(X_test), size=n_shap, replace=False)
            X_shap = X_test[idx]
            explainer = shap.TreeExplainer(rf)
            sv = explainer.shap_values(X_shap)
            if isinstance(sv, list):
                sv = sv[1]  # positive class
            elif sv.ndim == 3:
                sv = sv[:, :, 1]
            shap_values_mean = np.abs(sv).mean(axis=0)
        except Exception:
            logger.exception("[%s] SHAP computation failed; leaving SHAP values as NaN", attack_name)
    else:
        logger.warning("[%s] shap not installed; skipping SHAP importances", attack_name)

    # Interpretability extras: medians / direction / effect size per feature.
    attack_vals = attack_df[feature_cols]
    benign_vals = benign_sample[feature_cols]
    median_attack = attack_vals.median()
    median_benign = benign_vals.median()

    effect_sizes = {}
    for col in feature_cols:
        a = attack_vals[col].to_numpy(dtype="float64")
        b = benign_vals[col].to_numpy(dtype="float64")
        # Subsample for tractable Cliff's delta on very large groups.
        if len(a) > 2000:
            a = np.random.RandomState(RANDOM_SEED).choice(a, size=2000, replace=False)
        if len(b) > 2000:
            b = np.random.RandomState(RANDOM_SEED).choice(b, size=2000, replace=False)
        effect_sizes[col] = _cliffs_delta(a, b)

    return {
        "attack": attack_name,
        "feature_cols": feature_cols,
        "n_attack": n_attack,
        "n_benign_used": n_benign_used,
        "low_confidence": low_confidence,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "mi_scores": mi_scores,
        "rf_importances": rf_importances,
        "perm_importances": perm_importances,
        "shap_values_mean": shap_values_mean,
        "median_attack": median_attack,
        "median_benign": median_benign,
        "effect_sizes": effect_sizes,
        "X_test": X_test,
        "y_test": y_test,
    }


def aggregate_ranks(result: dict) -> pd.DataFrame:
    """Convert each importance method's scores to ranks and combine them via
    mean normalized rank (lower = more important); ties broken by
    permutation importance.

    Returns a DataFrame sorted by mean_norm_rank ascending, one row per
    feature, with all raw scores and interpretability columns attached.
    """
    feature_cols = result["feature_cols"]
    n = len(feature_cols)

    def to_norm_rank(scores: np.ndarray) -> np.ndarray:
        # Higher score => more important => rank 1. NaNs (e.g. missing SHAP)
        # are ranked last and excluded from the mean via nan-aware handling
        # further down.
        s = pd.Series(scores)
        if s.isna().all():
            return np.full(n, np.nan)
        ranks = s.rank(ascending=False, method="average", na_option="bottom")
        return (ranks / n).to_numpy()

    mi_rank = to_norm_rank(result["mi_scores"])
    rf_rank = to_norm_rank(result["rf_importances"])
    perm_rank = to_norm_rank(result["perm_importances"])
    shap_rank = to_norm_rank(result["shap_values_mean"])

    rank_stack = np.vstack([mi_rank, rf_rank, perm_rank, shap_rank])
    mean_norm_rank = np.nanmean(rank_stack, axis=0)

    median_attack = result["median_attack"]
    median_benign = result["median_benign"]
    direction = [
        "higher than benign" if median_attack[c] > median_benign[c] else "lower than benign"
        for c in feature_cols
    ]

    df = pd.DataFrame(
        {
            "attack": result["attack"],
            "feature": feature_cols,
            "mean_norm_rank": mean_norm_rank,
            "mi_score": result["mi_scores"],
            "rf_importance": result["rf_importances"],
            "perm_importance": result["perm_importances"],
            "shap_value": result["shap_values_mean"],
            "median_attack": [median_attack[c] for c in feature_cols],
            "median_benign": [median_benign[c] for c in feature_cols],
            "direction": direction,
            "effect_size": [result["effect_sizes"][c] for c in feature_cols],
            "low_confidence_flag": result["low_confidence"],
        }
    )
    # Sort by mean_norm_rank ascending, break ties with perm_importance descending.
    df = df.sort_values(
        by=["mean_norm_rank", "perm_importance"], ascending=[True, False]
    ).reset_index(drop=True)
    df.insert(1, "rank", np.arange(1, len(df) + 1))
    return df


def flag_redundancy(top10: pd.DataFrame, df: pd.DataFrame, threshold: float = 0.95) -> list[str]:
    """Return a list of 'feature_a <-> feature_b (rho=...)' strings for pairs
    in the top-10 feature set whose Spearman correlation exceeds ``threshold``.
    """
    features = top10["feature"].tolist()
    if len(features) < 2:
        return []
    corr = df[features].corr(method="spearman")
    pairs = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            rho = corr.iloc[i, j]
            if pd.notna(rho) and abs(rho) > threshold:
                pairs.append(f"{features[i]} <-> {features[j]} (rho={rho:.3f})")
    return pairs


def make_plots(
    top10_all: pd.DataFrame, out_dir: Path
) -> None:
    """Produce the feature x attack heatmap and per-attack bar charts."""
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not installed; skipping plots")
        return

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Heatmap: features (union of all top-10s) x attacks, cell = rank 1..10.
    attacks = top10_all["attack"].unique().tolist()
    features = top10_all["feature"].unique().tolist()
    pivot = top10_all.pivot_table(index="feature", columns="attack", values="rank", aggfunc="first")
    pivot = pivot.reindex(columns=attacks)
    # Order features by how many attacks they appear in, then best rank.
    appearance_count = pivot.notna().sum(axis=1)
    best_rank = pivot.min(axis=1)
    order = pd.DataFrame({"count": appearance_count, "best_rank": best_rank}).sort_values(
        by=["count", "best_rank"], ascending=[False, True]
    ).index
    pivot = pivot.reindex(index=order)

    fig_height = max(6, 0.35 * len(pivot))
    fig_width = max(8, 0.6 * len(attacks))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    masked = np.ma.masked_invalid(pivot.to_numpy(dtype="float64"))
    im = ax.imshow(masked, cmap="viridis_r", aspect="auto", vmin=1, vmax=10)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iat[i, j]
            if pd.notna(val):
                ax.text(j, i, int(val), ha="center", va="center", fontsize=6, color="white")
    fig.colorbar(im, ax=ax, label="Importance rank (1=most important)")
    ax.set_title("Top-10 discriminative feature rank by attack class")
    fig.tight_layout()
    fig.savefig(plots_dir / "feature_attack_heatmap.png", dpi=150)
    plt.close(fig)
    logger.info("Saved heatmap to %s", plots_dir / "feature_attack_heatmap.png")

    # Per-attack horizontal bar chart of top 10 by mean_norm_rank.
    for attack in attacks:
        sub = top10_all[top10_all["attack"] == attack].sort_values("rank")
        fig, ax = plt.subplots(figsize=(8, 5))
        # Lower mean_norm_rank = more important; plot inverted so most
        # important feature has the longest bar.
        importance_score = 1.0 - sub["mean_norm_rank"]
        ax.barh(sub["feature"][::-1], importance_score[::-1], color="steelblue")
        ax.set_xlabel("Importance (1 - mean normalized rank)")
        title = f"Top 10 features: {attack}"
        if sub["low_confidence_flag"].iloc[0]:
            title += " (low confidence: small sample)"
        ax.set_title(title)
        fig.tight_layout()
        safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", attack)
        fig.savefig(plots_dir / f"bar_{safe_name}.png", dpi=150)
        plt.close(fig)
    logger.info("Saved %d per-attack bar charts to %s", len(attacks), plots_dir)


def write_summary_md(top10_all: pd.DataFrame, model_quality: pd.DataFrame, out_path: Path) -> None:
    """Write a markdown summary with a table of the top 10 features per attack."""
    lines = ["# Top 10 Discriminative Features per Attack Class (CICIDS2017)", ""]
    lines.append("## Model quality (one-vs-benign RandomForest, held-out test set)")
    lines.append("")
    lines.append(model_quality.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    for attack in top10_all["attack"].unique():
        sub = top10_all[top10_all["attack"] == attack].sort_values("rank")
        header = f"## {attack}"
        if sub["low_confidence_flag"].iloc[0]:
            header += " ⚠️ low confidence (small sample size)"
        lines.append(header)
        lines.append("")
        display_cols = [
            "rank",
            "feature",
            "mean_norm_rank",
            "median_attack",
            "median_benign",
            "direction",
            "effect_size",
        ]
        lines.append(sub[display_cols].to_markdown(index=False, floatfmt=".4f"))
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote summary markdown to %s", out_path)


def main() -> None:
    """CLI entry point: load, clean, analyze, and write all outputs."""
    parser = argparse.ArgumentParser(
        description="Identify top-10 discriminative features per attack class in CICIDS2017."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory containing the 8 CICIDS2017 CSV files (default: ./data)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./output",
        help="Directory to write output CSVs/plots/markdown to (default: ./output)",
    )
    parser.add_argument(
        "--drop-dport",
        type=lambda v: v.lower() not in ("false", "0", "no"),
        default=True,
        help="Drop the Destination Port column (default True; it is highly leaky in "
        "CICIDS2017 since attack traffic is concentrated on a handful of specific ports)",
    )
    args = parser.parse_args()

    np.random.seed(RANDOM_SEED)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(data_dir)
    df = clean_data(df, drop_dport=args.drop_dport)

    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns]
    logger.info("Using %d feature columns", len(feature_cols))

    attack_classes = get_attack_classes(df)

    top10_rows = []
    model_quality_rows = []
    redundancy_notes = []

    for i, attack in enumerate(attack_classes, start=1):
        logger.info("[%d/%d] Analyzing attack class: %s", i, len(attack_classes), attack)
        result = compute_importances(df, attack, feature_cols)
        if result is None:
            continue

        ranked = aggregate_ranks(result)
        ranked.to_csv(
            out_dir / f"ranked_features_{re.sub(r'[^A-Za-z0-9_-]+', '_', attack)}.csv",
            index=False,
        )

        top10 = ranked.head(TOP_N).copy()
        top10_rows.append(top10)

        redundant_pairs = flag_redundancy(top10, df)
        if redundant_pairs:
            redundancy_notes.append(f"{attack}: " + "; ".join(redundant_pairs))
            logger.info("[%s] Redundant feature pairs (|rho|>0.95): %s", attack, redundant_pairs)

        model_quality_rows.append(
            {
                "attack": attack,
                "n_samples": result["n_attack"],
                "n_benign_used": result["n_benign_used"],
                "roc_auc": result["roc_auc"],
                "pr_auc": result["pr_auc"],
                "low_confidence_flag": result["low_confidence"],
            }
        )

    if not top10_rows:
        logger.error("No attack classes had enough samples to analyze; exiting")
        return

    top10_all = pd.concat(top10_rows, ignore_index=True)
    top10_all.to_csv(out_dir / "top10_per_attack.csv", index=False)
    logger.info("Wrote %s", out_dir / "top10_per_attack.csv")

    model_quality = pd.DataFrame(model_quality_rows)
    model_quality.to_csv(out_dir / "model_quality.csv", index=False)
    logger.info("Wrote %s", out_dir / "model_quality.csv")

    if redundancy_notes:
        (out_dir / "redundancy_notes.txt").write_text("\n".join(redundancy_notes), encoding="utf-8")
        logger.info("Wrote %s", out_dir / "redundancy_notes.txt")

    make_plots(top10_all, out_dir)
    write_summary_md(top10_all, model_quality, out_dir / "summary.md")

    logger.info("Done. All outputs written to %s", out_dir)


if __name__ == "__main__":
    main()

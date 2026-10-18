"""v4 conditions: the registered expression strings, evaluated vectorised, agree with the
per-row definitions; protocol scoping; no flag column is used."""
import numpy as np
import pandas as pd
import pytest

from eval.conditions_v4 import COLUMNS, CONDITIONS, evaluate
from eval.conditions_v4_measure import holds_frame, to_pandas_expr

FLAG_COLUMNS = {"SYN Flag Count", "ACK Flag Count", "FIN Flag Count", "PSH Flag Count", "RST Flag Count",
                "URG Flag Count", "CWE Flag Count", "ECE Flag Count", "Fwd PSH Flags", "Bwd PSH Flags",
                "Fwd URG Flags", "Bwd URG Flags"}


def _rows(n=400, seed=0):
    rng = np.random.default_rng(seed)
    data = {c: rng.choice([0, 1, 3, 5, 6, 11, 19, 20, 32, 500, 1500], size=n).astype(float) for c in COLUMNS}
    data["Init_Win_bytes_backward"] = rng.choice([-1, 0, 235], size=n).astype(float)
    data["Init_Win_bytes_forward"] = rng.choice([-1, 0, 8192], size=n).astype(float)
    df = pd.DataFrame(data)
    df["proto"] = rng.choice(["TCP", "UDP", "other"], size=n)
    return df


def test_no_condition_uses_a_flag_column():
    for c in CONDITIONS:
        assert not any(f in c.expression for f in FLAG_COLUMNS), c.cid
    assert FLAG_COLUMNS.isdisjoint(COLUMNS)


def test_pandas_translation_quotes_columns_and_joins_with_and():
    e = to_pandas_expr("Fwd Packet Length Max == Fwd Packet Length Min AND Total Fwd Packets > 5")
    assert e == "(`Fwd Packet Length Max` == `Fwd Packet Length Min`) & (`Total Fwd Packets` > 5)"


def test_vectorised_registered_expressions_match_per_row_definitions():
    df = _rows()
    H = holds_frame(df)
    for i, row in df.iterrows():
        feats = {c: row[c] for c in COLUMNS}
        expected = evaluate(feats, row["proto"])
        for cid, val in expected.items():
            assert bool(H.loc[i, cid]) == val, (i, cid)


def test_every_condition_is_exercised_both_ways():
    H = holds_frame(_rows(2000, seed=1))
    for c in CONDITIONS:
        if c.cid in ("D5", "D10"):  # impossible combinations by construction of the record (registered as such)
            continue
        assert H[c.cid].any() and (~H[c.cid]).any(), c.cid


def test_tcp_scoped_conditions_never_hold_off_tcp():
    df = _rows(1000, seed=2)
    H = holds_frame(df)
    for c in CONDITIONS:
        if c.scope == "TCP":
            assert not H.loc[df["proto"] != "TCP", c.cid].any(), c.cid


def test_evaluate_reads_no_label():
    feats = {c: 1.0 for c in COLUMNS}
    assert evaluate(feats, "TCP") == evaluate(dict(feats, Label="DoS Hulk"), "TCP")

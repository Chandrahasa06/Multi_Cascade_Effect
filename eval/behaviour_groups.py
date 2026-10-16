"""Collapses near-duplicate records into behaviour groups, so a flood that contributes many
near-identical flows counts once.

Distance: each of the 10 escalation features is min-max scaled over the dataset's own records;
distance = Euclidean / sqrt(10) (0 = identical, 1 = the full range on every feature). Records of
the same class are one behaviour if they are linked by a chain of pairs each within THRESHOLD
(single linkage). A behaviour counts as caught if a strict majority of its records were caught.

Reads only features and the class key; no verdict is used to form the groups.
"""
from __future__ import annotations

import itertools
from typing import Dict, List

import numpy as np

from agents.escalation_grounding import ESCALATION_FEATURES

THRESHOLD = 0.05
SENSITIVITY = (0.01, 0.05, 0.10)


def scaled(features_by_id: Dict[str, dict]) -> Dict[str, np.ndarray]:
    ids = list(features_by_id)
    X = np.array([[float(features_by_id[i][f]) for f in ESCALATION_FEATURES] for i in ids])
    lo, hi = X.min(0), X.max(0)
    Z = (X - lo) / np.where(hi > lo, hi - lo, 1.0)
    return dict(zip(ids, Z))


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.sqrt(len(ESCALATION_FEATURES)))


def groups(features_by_id: Dict[str, dict], class_by_id: Dict[str, str], threshold: float = THRESHOLD) -> List[List[str]]:
    """Behaviour groups (lists of record ids), never mixing classes."""
    Z = scaled(features_by_id)
    parent = {i: i for i in features_by_id}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in itertools.combinations(features_by_id, 2):
        if class_by_id[a] == class_by_id[b] and distance(Z[a], Z[b]) <= threshold:
            parent[find(a)] = find(b)
    out: Dict[str, List[str]] = {}
    for i in features_by_id:
        out.setdefault(find(i), []).append(i)
    return sorted(out.values(), key=lambda g: (class_by_id[g[0]], -len(g)))


def behaviour_score(grps: List[List[str]], is_attack: Dict[str, bool], caught: Dict[str, bool]) -> dict:
    att = [g for g in grps if is_attack[g[0]]]
    caught_b = [g for g in att if sum(caught.get(i, False) for i in g) * 2 > len(g)]
    mixed = [g for g in att if 0 < sum(caught.get(i, False) for i in g) < len(g)]
    return {"behaviours": len(att), "caught": len(caught_b), "mixed": len(mixed)}

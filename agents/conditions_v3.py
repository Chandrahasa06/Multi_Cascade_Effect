"""Code-side evaluation of the priors-v3 conditions, for the report only (agreement between A5's
checklist and the true answers; how often each condition holds by label). Never used to
build a prompt or decide a verdict. Mirrors agents/prompts/priors_text_v3.CONDITIONS exactly.
"""
from __future__ import annotations

from typing import Dict, Optional

YES, NO, NA = "yes", "no", "not_applicable"
SCOPES = {"C1": "TCP", "C2": "any", "C3": "any", "C4": "TCP", "C5": "any", "C6": "any", "C7": "any",
          "C8": "any", "C9": "any", "C10": "TCP", "C11": "any", "C12": "UDP"}


def evaluate(f: Dict[str, float], protocol: Optional[str]) -> Dict[str, str]:
    g = lambda k: float(f[k])  # noqa: E731
    fwd_b, bwd_b = g("Total Length of Fwd Packets"), g("Total Length of Bwd Packets")
    pkts = g("Total Fwd Packets") + g("Total Backward Packets")
    port = int(g("Destination Port"))
    tests = {
        "C1": g("Init_Win_bytes_backward") == -1,
        "C2": g("Total Backward Packets") == 0,
        "C3": fwd_b > 0 and bwd_b == 0,
        "C4": g("Init_Win_bytes_forward") == 0,
        "C5": g("Flow IAT Mean") > 10_000_000,
        "C6": bwd_b > 0 and fwd_b > 10 * bwd_b,
        "C7": pkts >= 3 and g("Packet Length Std") == 0,
        "C8": fwd_b == 0 and bwd_b == 0,
        "C9": port >= 1024 and port not in (8000, 8080, 8443),
        "C10": port in (21, 22, 23, 3389) and pkts <= 10,
        "C11": g("Flow Packets/s") > 1000 and g("Average Packet Size") < 100,
        "C12": port == 53 and bwd_b > 10 * fwd_b and bwd_b > 512,
    }
    out = {}
    for cid, holds in tests.items():
        scope = SCOPES[cid]
        out[cid] = NA if scope != "any" and scope != protocol else (YES if holds else NO)
    return out

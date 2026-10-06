"""The controller's judgement of escalated flows: the seam for rule-A refit.

Rule-A refit updates Priority 2's K only from escalated flows that the
controller judged benign. Where that judgement comes from is this module's job.

- LabelStandInVerdict: uses the TRUE label of each escalated flow as a stand-in
  for a perfect controller. Every number produced with it is an UPPER BOUND on
  what escalated-only feedback can deliver. It is the only place in the refit
  path that reads a label.
- AgentVerdict: the seam for the agent pipeline's real verdict. Not wired. The
  current pipeline's recall is about 50%, so a real verdict will be worse than
  the stand-in. Replacing LabelStandInVerdict with AgentVerdict is the whole
  change needed to measure that.

Nothing in dataplane.escalation_policy.OnlineSignatureTable reads a label. It
takes a boolean verdict only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class ControllerVerdict:
    """Judges escalated flows. judge() returns True where the flow is benign."""

    upper_bound: bool = True

    def judge(self, escalated: pd.DataFrame) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class LabelStandInVerdict(ControllerVerdict):
    """UPPER BOUND. A perfect controller is assumed: escalated flows are judged
    benign exactly when their true label is BENIGN. Use only to bound what
    refit can give; do not report it as a result about the agents."""

    upper_bound = True

    def __init__(self, label_col: str = "Label", benign_label: str = "BENIGN"):
        self.label_col = label_col
        self.benign_label = benign_label

    def judge(self, escalated: pd.DataFrame) -> np.ndarray:
        return (escalated[self.label_col] == self.benign_label).to_numpy()


class AgentVerdict(ControllerVerdict):
    """Seam for the agent pipeline's verdict. Not implemented: the pipeline is
    not yet able to judge a full day's escalations, and its recall is about
    50%. Calling this raises so no run can silently fall back to labels."""

    upper_bound = False

    def judge(self, escalated: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError(
            "agent verdict not wired: the agent pipeline's recall is ~50%, and it "
            "cannot yet judge a full day's escalations. Use LabelStandInVerdict "
            "only as a labelled upper bound.")

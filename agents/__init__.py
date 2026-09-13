"""Five-agent zero-day escalation-review pipeline (A1-A5) plus the
mechanical scoring (contamination, verification, trust) built to check
what the agents claim against the record they claimed it about.

See STATUS.md / README.md for the data-plane and control-plane stages
this pipeline consumes (``EscalationRecord``, from
``controlplane/record.py``).
"""

# Multi-Cascade Effect: zero-day intrusion triage on CICIDS2017

## What it does

A programmable data plane keeps cheap per-flow and per-source counters and escalates a small budget of suspicious flows to a controller. The controller extracts full CICFlowMeter features for each escalated flow and grounds it against real benign traffic: how many ordinary flows on this network look like this one. A five-agent LLM pipeline then reads the flow and that grounding and decides whether it has a benign explanation the data supports.

## The pipeline, in order

| stage | what happens | open this |
|---|---|---|
| 1. Escalation, Priority 1 | supervised decision-tree rules (compiled from `tree.txt`) flag known attack shapes | `dataplane/dt_rules.py`, `eval/parse_tree.py` |
| 1. Escalation, Priority 2 | benign-only signature table with a token-bucket meter; the only zero-day component | `dataplane/escalation_policy.py` |
| 1. Escalation, Priority 3 | deterministic hash sampling, so unseen behaviour is sometimes observed | `dataplane/escalation_policy.py` |
| 1. Escalation, evaluation | the three priorities run over the 2.83M-flow CSV pool | `eval/escalation_eval.py` (writes `escalation_report.md`), `eval/escalation_v3.py` (current defaults), `eval/escalation_data.py` |
| 2. Feature extraction | full CICFlowMeter features and the record handed to the controller | `controlplane/extractor.py`, `controlplane/record.py` |
| 3. Grounding | benign-neighbourhood count: benign flows within [0.5x, 2x] of this flow on ten features | `agents/escalation_grounding.py` |
| 4. Agents | A1 evidence, A2 behaviour, A3 hypotheses, A4 blind replica, A5 verdict | `agents/pipeline.py`, `agents/a1_evidence.py` ... `agents/a5_verdict.py` |
| 4. Agents, checks | schema, flow-level filter, trust scores | `agents/schema.py`, `agents/validators.py`, `agents/trust.py` |
| 5. Blind evaluation | labels held in a separate key file; agents never see them | `eval/run_blind_pipeline.py` |

Tests: `python -m pytest tests -q` (842 passing). The CICIDS2017 CSVs go in `data/`, which is not in the repository.

## Results

**The detector works. The agent layer does not add to it.**

- The benign-neighbourhood count separates attacks from benign flows: AUC 0.968 on the 44 blind-scored flows (29 attacks, 15 benign). That AUC uses a reference that includes the test days, so it is not a deployment figure.
- With deployment-realistic grounding (Monday benign traffic only, flows from Tuesday onward), it catches 20 of 29 attacks and clears 13 of 13 benign flows. The samples are small (Wilson 95% intervals: 51-83% and 77-100%).
- Across nine attempts to make the agents improve on that count, none did. (Seven are in the write-up; the eighth ran the agents only where the count says benign, and the ninth checked what the agents scored before the cap overwrote them.) No configuration caught more attacks than the count, and in most the verdict matched it record for record. Where the agents seemed to detect, they were either following a count shown to them or being overruled by a cap that applied the count. This is a measured negative result, and the most carefully tested part of the work.

| report | question it answers |
|---|---|
| [results/writeup_corrected.md](results/writeup_corrected.md) | Start here. The full argument: setup, the baseline, every attempt, and the limits. |
| [results/count_baseline.md](results/count_baseline.md) | Do the agents beat a one-line threshold on the same count? (No: zero disagreements in 44 records.) |
| [results/agent_reference_arms.md](results/agent_reference_arms.md) | What happens with deployment-realistic grounding (Monday only, Monday+Tuesday)? |
| [results/cap_suppressed_scores.md](results/cap_suppressed_scores.md) | When the agents flag an attack, is it their own judgement? (No: 0 of 20 would have been flagged without the cap.) |
| [results/grounding_explained.md](results/grounding_explained.md) | Exactly how the neighbourhood count is computed, with worked examples. |
| [results/escalation_report.md](results/escalation_report.md) | How the three-priority data-plane policy performs on the full CSV pool. |

The other files in `results/` are local only (not in the repository): intermediate runs, caches and superseded reports.

## Limits

- 45 blind records (30 attack, 15 benign), deliberately attack-weighted. This measures discrimination, not deployment rates.
- One model family and one prompt lineage were tested.
- Priority 1 is supervised and has no zero-day property. Priority 2 carries the zero-day claim; Priority 3 guarantees nothing.

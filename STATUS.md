# Status

Snapshot for picking this back up in a fresh session. See `README.md` for
setup/reproduction instructions and the full "Known Limitations" writeup;
this file is about what's actually true *right now*.

## What's built

- `dataplane/flow_state.py` — `Packet`/`FlowState`, canonical bidirectional 5-tuple keying, O(1) counter updates, FIN-handshake/RST termination, `source_row_id` traceability.
- `dataplane/flow_table.py` — fixed-capacity table with `KeyMode.FIDELITY` (plain 5-tuple, can merge rows) and `KeyMode.EVAL` (5-tuple + source row, exact 1:1); O(1) amortized idle/active timeout and LRU eviction; EVAL mode defaults to unbounded capacity and skips timeout enforcement entirely.
- `dataplane/hyperloglog.py` — from-scratch HyperLogLog sketch (Murmur3-avalanched hash; plain CRC32 was tried first and found to bias badly on sequential input like a port scan).
- `dataplane/src_table.py` — per-source sliding-window counters (6×10s rotating buckets) and HLL sketches (precision=8, ~6.5% error), own LRU capacity (default 8,192, independent of the flow table).
- `dataplane/selector.py` — Tier-1 + per-source feature computation, `Selector`/`SelectorConfig`/`TriggerReason` (structured, not strings), low-confidence flagging for timing features on coarse-resolution days.
- `dataplane/fitting.py` — benign-only percentile fitting (`fit_thresholds` from live flows, `fit_thresholds_from_frame` from a cached DataFrame — the sweep's fast path); benign-only-ness is asserted as the literal first line; feature-kind-aware fitting (continuous / bounded_ratio / boolean, see "Bugs fixed" below).
- `adapters/csv_flow_adapter.py` — CICIDS2017 TrafficLabelling CSV → synthetic `Packet` stream (lossy by construction, documented in-module); utf-8/cp1252 fallback; per-row timestamp-resolution tagging.
- `eval/labels.py` — EVAL-mode join (asserted 1:1) and FIDELITY-mode nearest-timestamp resolution with residual-ambiguity flagging (benign+attack vs attack+attack mixing reported separately).
- `eval/simulate.py` — single-pass simulation + feature extraction per day, cached to parquet keyed on (day, key_mode, adapter version, feature-schema version).
- `eval/sweep.py` — chronological Monday fit/holdout split, vectorized threshold-crossing and escalation-rule logic, per-class recall, 20-point percentile sweep.
- `eval/report.py` / `eval/run_all.py` — ablation orchestration, per-class recall tables (overall-rate- and benign-rate-anchored), trigger-frequency, small-multiples recall-vs-escalation plot, markdown + CSV to `results/`.
- `tests/` — 236 tests, all passing, across all of the above.

## Current state of results — mostly invalidated, re-run in progress

The last full sweep (`results/summary.md` etc., not committed — `results/`
is gitignored) predates both bug fixes below and should not be trusted
for anything derived from per-source features or bounded/boolean
thresholds. Concretely, from that run:

- Baseline (`any` rule): p99.99, 11.57% overall escalation, 0.07% benign, 17.5% overall attack recall.
- `k_of_n(k=2)`: p99.873, 4.69% overall, 0.44% benign, 16.8% recall.
- `k_of_n(k=3)`: p99.621, 6.34% overall, 0.77% benign, 26.5% recall.
- Per-class highlights: FTP/SSH-Patator and Heartbleed ~100% recall everywhere; PortScan ~0%; DoS Hulk 11.8–57.3% depending on operating point.
- Per-source-feature effect: Patator +98.7pp/+99.9pp with source features; Bot roughly flat; Infiltration *worse* with source features (-11.1pp, n=36).
- "Dead" features: `distinct_dst_ips_per_src`, `flow_bytes_per_sec`, `no_response_flag`, `init_win_bytes_fwd`, `init_win_bytes_bwd`, `syn_ratio`.

**Invalidated by the bugs below:** every number that touches a per-source
feature (all of "per-source-feature effect", `flows_per_src`/
`distinct_dst_ports_per_src`/`distinct_dst_ips_per_src`/
`syn_without_synack_count` in every table) or a bounded/boolean feature
(`syn_ratio`, `rst_ratio`, `no_response_flag` — including their presence
on the "dead features" list). That's most of the sweep's substance,
including the headline recall/escalation numbers, since per-source
features contribute to escalation decisions throughout.

**Not invalidated** (unaffected by either bug): `results/join_rates.csv`
(label-join characterization is independent of fitting/per-source-query
logic), the raw extraction metadata (flow counts, flow-table eviction
counts, adapter skip counts), and the Web Attack row-survival count.

A corrected full re-extraction + re-sweep was started and was still
running when this file was written (8 days × 2 key modes, each a
several-minute one-time simulation cost — `eval.run_all_extractions`,
then `eval.run_all`). Restart it or check on it before trusting anything
new in `results/`.

## Bugs just fixed

1. **Percentile fitting was structurally broken for bounded features.**
   `no_response_flag` is boolean; `syn_ratio`/`rst_ratio` are ratios that
   saturate at 0/1 for short flows. When benign mass sits exactly at a
   hard bound, the fitted percentile lands on that bound too, and a
   plain `>`/`<` comparison can then never cross it — the threshold
   looked reasonable but was structurally uncrossable forever. This is
   exactly why PortScan (a SYN-only, no-reply pattern) was reading near
   0% recall. Fixed by classifying features as continuous / bounded_ratio
   / boolean and giving the latter two an explicit saturation check:
   provably, whenever the percentile lands on the bound, benign mass
   there already meets or exceeds that percentile's own false-positive
   budget, so the feature is correctly omitted (and reported `saturated`)
   rather than fit to a dead threshold.

2. **Per-source features were snapshotted at the wrong time.**
   `compute_src_features` was being called in a pass *after* the entire
   day's simulation had finished, using `SrcTable`'s final end-of-file
   state for every flow regardless of when that flow actually closed.
   Since a source's bucket state keeps rotating forward as later packets
   from that *same* source are processed, any source that reappeared
   later in the file had its earlier flows' per-source features silently
   overwritten by whatever the source was doing near end-of-file — buckets
   can only be evicted going forward, never reconstructed to a past
   state. This corrupted every per-source feature value for every flow
   that wasn't the last thing its source did that day. Fixed by moving
   feature computation into the main simulation loop, snapshotted at each
   flow's own closure instant. Regression test in
   `tests/test_simulate.py` reproduces the exact failure shape (same
   source, ports touched early, source reappears 200s later) and checks
   the early flow's snapshot is unaffected by the later activity.

`k_of_n(k=2)` was also made the default escalation rule (was `any`)
given the measured union-bound blowup: ~20 independently-fit features
combined via OR pushed benign escalation to 68.9% at p90 in the old run.

## Not built yet

- `controlplane/extractor.py` — packet window → CICFlowMeter → 78
  features. Blocked on real PCAPs (still downloading per the original
  brief). Write it against a synthetic/tiny sample PCAP first per
  standing instructions, then stop for confirmation before trusting it
  against the real captures.
- `adapters/pcap_flow_adapter.py` (the PCAP-shaped peer of
  `csv_flow_adapter.py`) — not started. `FlowTable`/`FlowState` need no
  changes to support it; it just needs to produce the same `Packet`
  stream.
- `controlplane/anonymise.py`, `controlplane/record.py` (the
  `EscalationRecord` handoff dataclass) — not started.
- No CLI entry point yet; everything runs via `python -m eval.<module>`.

## Known data quirks (CICIDS2017 TrafficLabelling)

- Column names have leading spaces (`' Flow Duration'`); stripped on load.
- `Flow Bytes/s` / `Flow Packets/s` contain Infinity/NaN at zero duration;
  not used directly by the adapter (which derives its own timing from
  `Flow Duration` instead), but any code reading those raw columns must
  filter them explicitly, as called out in the original brief.
- The Infiltration file is misspelled `Infilteration` in the original
  release filename — the adapter treats the path as given, no special
  handling needed, just don't "fix" the filename.
- **Monday has second-resolution timestamps; Tuesday–Friday only have
  minute resolution** (no seconds field, e.g. `7/7/2017 3:30`). Carried
  through as `Packet.timestamp_resolution_us` / `FlowState.
  timestamp_resolution_us`; the selector flags IAT/duration/rate
  features `low_confidence=True` on coarse-resolution flows.
- The WebAttacks file (`Thursday-WorkingHours-Morning-WebAttacks`) has
  two independent problems: its Label column contains an en-dash encoded
  as **Windows-1252, not UTF-8** (`load_csv` falls back to cp1252 on
  decode failure), and **62.9% of its rows (288,602 of 458,968) are
  completely blank** — every column empty, not just the label. The
  adapter's existing missing-field skip logic already discards these
  correctly; confirmed separately that all 2,180 real Web Attack rows
  (Brute Force 1,507 + XSS 652 + SQL Injection 21) survived intact —
  none of the blank rows were mislabeled Web Attack rows.
- Rows sharing a 5-tuple can carry different ground-truth labels
  (`FlowTable.KeyMode.FIDELITY` can merge them); `KeyMode.EVAL` sidesteps
  this by construction rather than solving the underlying ambiguity —
  see the README's "Known Limitations" section.

## Open questions

- **Is Bot/Web-Attack detection genuinely near-impossible from Tier-1 +
  per-source flow statistics, or is it another artifact?** In the last
  (partially invalidated) run, Bot recall barely moved with or without
  per-source features (~2.7–3.4%), and the Web Attack classes stayed
  under 1% almost everywhere except the loosest percentile. That's a
  real, defensible possibility — Bot/web-app-layer attacks may just not
  show up in packet-count/timing/size statistics the way volumetric or
  per-source-pattern attacks do — but it hasn't been checked against a
  version of the sweep with both bugs fixed yet. Re-run and see whether
  it holds; if it does, that's a finding about this feature set's ceiling
  for those classes, not a bug to keep chasing.
- **Low-confidence IAT ablation (slowloris/Slowhttptest specifically):**
  not available yet — the re-run that would answer this (coarse-timestamp
  IAT features included vs excluded from fitting, per-class recall on
  the two slow-attack classes that plausibly depend most on IAT) was
  still in progress when this file was written. `eval/run_all.py`
  already prints and saves this comparison
  (`results/low_confidence_ablation_delta.csv`) — just needs the sweep
  to finish.

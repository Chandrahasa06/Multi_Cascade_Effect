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
- `adapters/pcap_adapter.py` — real pcapng capture → `Packet` stream via direct byte-slicing (no scapy in the production path — an earlier scapy-`PcapReader` version ran at ~230 pkts/sec, full per-packet dissection was the bottleneck). Extends the same sequential pcapng block-parser used for capture verification to also read the ~40 header bytes a 5-tuple table needs: EtherType/VLAN, IP header, TCP/UDP header. IPv4 TCP/UDP only (ARP/IPv6/ICMP/non-first-IP-fragments skipped and counted in `AdapterStats.skip_reasons`). `source_row_id` always `None` (no CSV row to trace to; `KeyMode.EVAL` doesn't apply here, only `FIDELITY`-style keying). `Packet.length` is IPv4 Total Length (L3), not Ethernet wire length — a deliberate semantic change from the scapy version, matching CICFlowMeter's own byte-count convention (`ADAPTER_VERSION` bumped to `"2"` accordingly). scapy remains a project dependency, used only as a correctness-gate reference decoder in `tests/test_pcap_adapter.py` (26 tests, including a per-flow-counter identity check against the scapy path — this is what caught the IP-fragment bug below). Run against both full captures — see below.
- `eval/labels.py` — EVAL-mode join (asserted 1:1) and FIDELITY-mode nearest-timestamp resolution with residual-ambiguity flagging (benign+attack vs attack+attack mixing reported separately).
- `eval/simulate.py` — single-pass simulation + feature extraction per day, cached to parquet keyed on (day, key_mode, adapter version, feature-schema version).
- `eval/sweep.py` — chronological Monday fit/holdout split, vectorized threshold-crossing and escalation-rule logic, per-class recall, 20-point percentile sweep.
- `eval/report.py` / `eval/run_all.py` — ablation orchestration, per-class recall tables (overall-rate- and benign-rate-anchored), trigger-frequency, small-multiples recall-vs-escalation plot, markdown + CSV to `results/`.
- `eval/labels_pcap.py` / `eval/simulate_pcap.py` / `eval/run_pcap_sweep.py` — PCAP-path peers of `eval/labels.py`/`eval/simulate.py`/the sweep orchestration in `eval/run_all.py`: Friday's identity-first session join (real packets have no CSV row to key on), PCAP-path simulate+cache, fit/sweep against real timing. See "PCAP-based fitting, sweep, and Friday session labeling" below.
- `controlplane/record.py` / `controlplane/extractor.py` — escalated-flow → full CICFlowMeter feature extraction and allow-list anonymisation. See "Control plane: CICFlowMeter extraction" below.
- `dataplane/selector.py`'s `flow_iat_regularity` (Bot beaconing hypothesis, tested and falsified — see "Bot IAT-regularity experiment" below) and `Selector.evaluate_features` (reusable escalation-decision core for callers holding already-snapshotted features, e.g. the record-sample generator, without re-deriving per-source features from a stale `SrcTable` — the exact bug already fixed once in this project).
- `agents/` — the five-agent zero-day review pipeline (A1 evidence → A2 behaviour → A3 hypotheses, A4 blind replica, A5 verdict), plus `controlplane/reference.py` (Monday benign reference distribution) and `eval/run_agent_pipeline.py` (cost-controlled runner). See "Five-agent zero-day review pipeline" below — stop points 1-2 done; stop point 3 (the full 200-record run, re-stratified) is **159/200**, parked on a live daily-quota 429 that fires well under the tracked count (see "Architecture doc, self-report coverage gap, PCAP counts, threshold sweep" near the end of this file for the current best guess why).
- `eval/ablation_v4.py` / `eval/ablation_v200.py` / `eval/threshold_sweep.py` / `eval/pcap_escalation_counts.py` / `eval/presentation_figures.py` — zero-API-call re-analysis of already-written pipeline/sweep results (detector-vs-agent comparisons, verdict-threshold optimality, exact PCAP escalation counts, poster/presentation figures). `eval/fault_injection.py` / `eval/run_fault_injection.py` / `eval/poster_figures.py` — the fault-injection poster study (built and unit-tested; the one real run attempted so far is blocked on quota, see below). `results/ARCHITECTURE.md` — full as-implemented pipeline description (topology, verbatim prompts, schemas, trust formula), read fresh from code on request.
- `tests/` — 471 tests, all passing, across all of the above.

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

## PCAPs landed and verified (2017-07-03 Monday, 2017-07-07 Friday)

`data/pcap/Monday-WorkingHours.pcap` (10.8GB) and
`data/pcap/Friday-WorkingHours.pcap` (8.8GB) arrived, in **pcapng**
format (not classic pcap — magic `0x0A0D0D0A`). Verified with a
purpose-written streaming block parser (header-only, sequential-read,
no full-file load — see `git log` for the throwaway verification
script if it's needed again; it wasn't kept in the repo):

- Both files parse cleanly end to end, no truncation, single section,
  single interface (linktype 1 = Ethernet) each.
- Monday: 11,709,971 packets, 2017-07-03T11:55:58.598 → 20:01:34.472
  UTC (~8.09h). Friday: 9,997,874 packets, 2017-07-07T11:59:39.599 →
  20:02:41.169 UTC (~8.05h).
- **Confirmed genuine microsecond resolution** (`if_tsresol` option =
  1e-6s on both), not just microsecond-formatted-but-actually-coarser:
  100% of the first 200,000 packets in each file have a nonzero
  sub-tick component. This is the fine-grained timing the Tue–Fri CSVs
  never had — see "Known data quirks" below.
- 60-second test fixture cut from Monday's temporal midpoint
  (2017-07-03T15:58:46.012 → 15:59:45.949 UTC, 6,786 packets, 2.6MB) at
  `tests/fixtures/monday_60s_slice.pcap`. Extracted by copying pcapng
  blocks verbatim (not re-encoded) so it's byte-identical to what a
  real reader would see for that window — self-contained, re-verified
  standalone. Everything PCAP-related is built and tested against this
  slice only.
- Friday's capture contains three attack sessions in one file (Bot in
  the morning, PortScan and DDoS in the afternoon) that map to three
  separate CICIDS2017 CSVs (`Friday-WorkingHours-Morning` [Bot],
  `Friday-WorkingHours-Afternoon-PortScan`,
  `Friday-WorkingHours-Afternoon-DDos`) — not yet exploited for
  anything; noted for whoever builds the PCAP-side label join.

**Both full captures now run end to end.** Byte-slicing rewrite (see
above) plus a correctness gate against a scapy reference decoder (26
tests in `tests/test_pcap_adapter.py`, all passing) — the gate caught a
real bug before any full run: 2 packets in the 60s slice were non-first
IP fragments (fragment offset != 0, no transport header present); the
naive parser was reading their raw payload continuation as if it were a
fresh TCP/UDP header. Fixed by checking the IP fragment-offset field and
skipping with reason `"ip_fragment"` (260 more found across the full
Monday file, 196 in Friday — rare but real on this dataset).

Results, `FlowTable(key_mode=KeyMode.FIDELITY)`, default 65,536 capacity:

| | Monday | Friday |
|---|---|---|
| packets read | 11,709,971 | 9,997,874 |
| packets emitted (IPv4 TCP/UDP) | 11,625,508 | 9,914,340 |
| packets skipped | 84,463 (83,479 non-IPv4, 671 ICMP, 260 fragment, 53 IGMP) | 83,534 (82,194 non-IPv4, 965 ICMP, 196 fragment, 117 IGMP, 62 SCTP) |
| flows created | 566,864 | 677,963 |
| capacity evictions | 0 | 0 |
| idle timeouts | 371,821 | 293,469 |
| active timeouts | 3,473 | 3,273 |
| still open at flush | 239 | 23 |
| wall clock | 301.5s (5.0 min) | 253.3s (4.2 min) |
| throughput | 38,841 pkts/sec | 39,468 pkts/sec |

**Monday's flow count vs. the CSV**: 566,864 simulated flows vs. 529,918
CICFlowMeter rows — **+6.97%**, not an exact match as expected (see
README's "Known Limitations": our flow-boundary definition isn't
CICFlowMeter's), and small enough not to look like a bug. Notably the
opposite direction and much smaller magnitude than the CSV-driven
synthetic adapter's known effect (529,918 rows → 409,476 flows, -22.7%,
via spurious merging from evenly-spaced synthetic packet timing) — real
per-packet timing here produces *less* incidental merging, not more,
which is the expected direction once real IATs replace synthesized ones.

**Throughput came in well under the 200k pkts/sec target — root cause is
disk I/O, not CPU, confirmed rather than assumed:**
- Small in-memory-cached benchmark slices (60s, 10min, 30min, all
  freshly written by the extraction script and re-read multiple times
  within seconds) consistently hit ~100,000–130,000 pkts/sec — the
  actual CPU-bound ceiling of the parser + `FlowTable.process()`
  together.
- Both full captures' first real pass converged to the *same* ~39,000
  pkts/sec despite different files, different content, and (for Friday)
  not having been freshly touched by an adapter run before — ruling out
  a code-side or content-dependent explanation.
- Explicitly ruled out idle-timeout churn as the cause: a 30-minute
  slice with *more* idle-timeout churn per packet than the full runs
  (1-per-19.3 vs. 1-per-31.3) still ran at the full ~130k pkts/sec.
- A raw sequential read (no parsing at all) measured **805 MB/s on
  Monday and 769 MB/s on Friday — but both measurements were taken
  immediately after that same file had just been fully read by the
  adapter**, i.e. likely a warm-page-cache re-read, not the cold
  first-touch speed the adapter itself paid. That reframes the 805/769
  MB/s numbers as an artifact rather than proof I/O was fine.
- Net: this system's storage appears to cap first-touch sequential reads
  of these files around 35–40 MB/s; re-reads of already-cached data run
  at RAM speed (~800 MB/s). The adapter's own CPU-bound ceiling
  (~100–130k pkts/sec) is not the constraint on a first pass over either
  file as currently observed on this machine.
- Not conclusively proven (would need e.g. a guaranteed-cold read via a
  cache-drop mechanism this environment doesn't expose), but is the
  best-supported explanation from the evidence gathered, and matches
  everyday experience of this environment/machine's storage.

**Closure-reason breakdown and peak concurrency (checked, not
assumed — the 0-eviction result above was suspicious for an 8-hour
capture and needed a real answer):**

| | Monday | Friday |
|---|---|---|
| idle_timeout | 371,821 (65.6%) | 293,469 (43.3%) |
| rst | 75,787 (13.4%) | 306,023 (45.1%) |
| fin | 115,544 (20.4%) | 75,175 (11.1%) |
| active_timeout | 3,473 (0.6%) | 3,273 (0.5%) |
| flush (still open at EOF) | 239 (0.04%) | 23 (0.003%) |
| **peak concurrent flows** | **2,223** | **2,312** |
| capacity | 65,536 | 65,536 |

Idle timeout dominating Monday isn't it "firing too early": it's
CICFlowMeter's own default (`DEFAULT_IDLE_TIMEOUT_US` = 15s, matching
the tool this project is trying to approximate), and peak concurrency
sits at ~3.4-3.5% of capacity on both days regardless of closure-reason
mix — real per-second flow arrival here (~19-23/sec) is nowhere near
enough to pressure a 65,536-slot table at any plausible timeout
setting. Friday's much higher RST share (45.1% vs Monday's 13.4%) is
itself a sane, attack-flavored signal (PortScan/DDoS tooling generates
far more externally-visible resets than ordinary browsing traffic).
Worth noting this is the *first* genuine concurrency measurement in the
project — `adapters/csv_flow_adapter.py`'s own docstring already
disclaims its concurrency numbers as unrealistic (rows fed sequentially
to completion, no true interleaving), a limitation real packet timing
doesn't have.

**Stale-threshold audit (L2→L3 length change, `ADAPTER_VERSION` "1"→"2"):**
confirmed clean. No `save_thresholds`/`load_thresholds` JSON file is
persisted anywhere in the repo (only exercised in a throwaway test).
Every cached feature parquet under `results/cache/` is
`adapter1`-keyed (CSV path) — a completely separate cache namespace
from the PCAP adapter's `adapter2`, and CSV-path lengths were never
PCAP/scapy-derived in the first place, so the length-semantic change
can't reach them. No PCAP-adapter cache existed before this session's
`eval/simulate_pcap.py` runs, so there was nothing stale on that side
either. One real staleness *was* found in the process, unrelated to the
length change: `results/cache/friday_*__adapter1__features2.*` predated
`FEATURE_SCHEMA_VERSION` bumping to `"3"` (the per-source-feature and
bounded-threshold bugfixes) — refreshed before using it as this
session's CSV-path comparison baseline (Monday's `adapter1__features3`
cache was already current and reused as-is).

## PCAP-based fitting, sweep, and Friday session labeling

New modules, mirroring the CSV path's shape: `eval/labels_pcap.py`
(Friday's session join), `eval/simulate_pcap.py` (PCAP-path
simulate+cache, peer of `eval/simulate.py`), `eval/run_pcap_sweep.py`
(fit/sweep orchestration, peer of `eval/sweep.py` usage in
`eval/run_all.py`). 7 new tests in `tests/test_labels_pcap.py`; full
suite 269/269.

Monday PCAP is benign-only by construction (CICIDS2017's own design,
already how the CSV path treats it) — labeled `BENIGN` directly, no
CSV join needed. Chronologically split as usual
(`eval.sweep.split_monday_chronologically`): fit=283,432,
holdout=283,432.

**`exclude_low_confidence=False` vs the CSV path's default `True`,
checked not assumed:** on Monday PCAP, both settings exclude exactly
**0** flows from fitting, for every feature, at every percentile
tested. This isn't a coincidence to shrug off — it's because every
PCAP-derived `Packet.timestamp_resolution_us` is 1 (confirmed
microsecond, see above), which never exceeds
`LOW_CONFIDENCE_RESOLUTION_THRESHOLD_US` (1 second), so
`low_confidence` is never `True` for PCAP features in the first place.
What "dropping the exclusion" actually buys isn't a fitting-time
effect — it's that Monday PCAP's `flow_iat_mean/min/max` etc. are now
*real observed* inter-arrival times, where even the CSV path's
Monday (its most precise day, second-resolution) still only ever had
CICFlowMeter's evenly-*synthesized* spacing within a flow (see
`adapters/csv_flow_adapter.py`'s module docstring) — this is the first
time this project has fit thresholds against genuine per-packet timing
at all, on any day.

### Friday session splitting: two real bugs found and fixed

**Bug 1 — a 12-hour timestamp ambiguity in two of the three Friday
CSVs.** Session windows were first built by literally taking
`min`/`max` of each CSV's own parsed `Timestamp` column. Bot's window
landed correctly (2017-07-07 08:59-12:59 local, matching the PCAP's own
11:59:39 UTC start once the +3h local-to-UTC offset is applied), but
PortScan's and DDoS's windows landed **entirely before the PCAP
capture even starts** — 8 to 11 hours earlier, which is impossible.
Their raw `Timestamp` strings ("1:00"-"3:29" for PortScan, "3:30"-"5:02"
for DDoS) turned out to be unmarked 12-hour afternoon times
(`adapters/csv_flow_adapter.parse_timestamp`'s 24-hour-clock assumption
— correct for every other CICIDS2017 file — reads them 12 hours early).
A flat +12h correction, applied only to these two files
(`eval/labels_pcap.py`'s `pm_hour_correction` flag, with the full
reasoning and cross-check against the PCAP's own real timestamps in its
docstring), lines the three sessions up back-to-back inside the actual
capture with no gaps: Bot ends where corrected-PortScan begins;
corrected-DDoS ends within a minute of the capture's own last packet
(20:02 vs 20:02:41 UTC). This is a genuinely new finding — the CSV-only
pipeline never surfaced it because it only ever compared a file's
Timestamp column to *itself* (row ordering, unaffected by a uniform
offset), never to an external absolute clock. Worth flagging if anyone
downstream ever needs Friday's CSV timestamps in real UTC terms.

**Bug 2 — time-window-first flow assignment silently dropped
legitimate identity matches.** The first working version assigned each
PCAP flow to a session by `first_ts` membership in that session's time
window, *then* looked up its 5-tuple only within that pre-filtered
session. Result: of 52,514 flows time-windowed into "Bot", only 2
resolved to a `Bot` label — even though a direct check found **all
1,228** distinct Bot 5-tuples from the CSV present *somewhere* in
Friday's full-day flow set. Root cause: `FlowTable` runs continuously
across the whole day in FIDELITY mode (no per-session reset), so a
5-tuple used during Bot's official morning window can legitimately
close and reopen as a brand-new `FlowState` later the same day (RST/FIN
closed the earlier instance, the OS then reused the same ephemeral
port) — that later instance's `first_ts` falls outside Bot's
CSV-observed window even though its *identity* still matches a
Bot-labeled row exactly. Fixed by checking 5-tuple identity against all
three sessions *before* falling back to time-window + nearest-timestamp
(`eval/labels_pcap.join_friday_sessions`, identity-first). Regression
tests in `tests/test_labels_pcap.py::TestJoinFridaySessionsIdentityFirst`
reproduce the exact failure shape.

Post-fix Friday session join (all 677,963 flows):

| session | flows in session | matched by 5-tuple | matched by nearest-timestamp |
|---|---|---|---|
| bot | 164,483 | 164,482 | 1 |
| portscan | 309,963 | 309,963 | 0 |
| ddos | 203,080 | 203,079 | 1 |
| *(outside all three)* | 437 | — | — |

437 flows (0.06% of the day) fall outside every session and are
excluded from the labeled set rather than guessed at.

Label counts after the join — PCAP flows vs. the CSV's own row counts
for the same session (different by construction, see README's "Known
Limitations": FIDELITY-mode flow boundaries aren't CICFlowMeter's):

| label | PCAP flows | CSV rows |
|---|---|---|
| Bot | 2,901 | 1,966 |
| PortScan | 158,945 | 158,930 |
| DDoS | 81,075 | 128,027 |
| BENIGN | 434,605 | 414,322 |

### Result: per-class recall, CSV-path vs PCAP-path, at matched held-out benign escalation

Both sweeps: percentile 90.0-99.99, `k_of_n(k=2)`, anchored on
`benign_escalation_rate` (held-out Monday half). CSV path uses its
standard `exclude_low_confidence=True`; PCAP path uses `False` (shown
above to be a no-op on Monday specifically, included for consistency
with how the sweep is meant to run going forward).

| class | target benign rate | CSV recall (n) | PCAP recall (n) |
|---|---|---|---|
| Bot | 1% | 2.7% (1,966) | 1.8% (2,901) |
| Bot | 5% | 3.0% (1,966) | 2.1% (2,901) |
| Bot | 10% | 3.3% (1,966) | 3.0% (2,901) |
| PortScan | 1% | 0.14% (158,930) | **99.7%** (158,945) |
| PortScan | 5% | 0.47% (158,930) | **99.8%** (158,945) |
| PortScan | 10% | 0.77% (158,930) | **99.8%** (158,945) |
| DDoS | 1% | 61.9% (128,027) | **98.7%** (81,075) |
| DDoS | 5% | 63.6% (128,027) | **98.7%** (81,075) |
| DDoS | 10% | 63.7% (128,027) | **98.7%** (81,075) |

**PortScan recovers dramatically** — from effectively 0% (the
bounded/boolean-fitting bug's signature: PortScan is a SYN-only,
no-reply pattern, exactly what `no_response_flag`/`syn_ratio`
saturation was silently killing) to ~99.7-99.8%. Both fixes are in play
here (the bounded-feature saturation fix, already landed before this
session — see "Bugs just fixed" — plus real per-packet timing replacing
CSV-synthesized spacing), and this result can't distinguish their
individual contributions, but the combination clearly resolves what
STATUS.md's last (partially-invalidated) run flagged as "PortScan
~0%." **DDoS improves substantially too** (62-64% → ~98.7%). **Bot
stays low on both paths** (~2-3%, PCAP if anything slightly lower than
CSV) — this now reads as a real, defensible ceiling for this Tier-1 +
per-source feature set rather than an artifact: it survives both the
coarse-timing removal and the bounded-feature fix simultaneously, on
real packet-level data, with a *larger* Bot sample than the CSV path
had (2,901 vs 1,966 flows — more data pointing the same way, not less).
This resolves the open question from the "Bugs just fixed" section in
Bot/Web-Attack's favor: not an artifact.

**Caveat on reading this table**: class sizes differ meaningfully
between paths (own table above) — this is a flow-boundary-definition
difference, not a sampling bug, and recall is computed within each
path's own flow population, not against a shared/aligned set of
"ground truth incidents." Treat the *shape* of the comparison (PortScan
and DDoS recovering, Bot staying flat) as the finding, not the exact
percentage-point gap between CSV and PCAP for a class whose flow count
also moved.

## Not built yet

- The full 200-record agent-pipeline run (stop point 3) — only the
  20-record stratified sample (5/class) has been run live so far, four
  times over as the design was iterated (see "Five-agent zero-day
  review pipeline" below). At ~100 calls/20 records and a measured ~400
  calls/day realistic budget on `gemini-3.5-flash-lite`, the full
  200-record run (1000 calls) plus the single-LLM baseline (200 more)
  will span multiple days even on a good day, longer if the model's
  real ceiling is lower than currently assumed.
- The single-LLM baseline control (`agents/baseline.py` exists, wired,
  never run live) — needed before any "does the five-agent structure
  earn its cost" claim.
- `eval/agent_eval.py` — the full scoring/reporting harness against the
  manifest (confusion matrix, per-class detection rate, trust-score
  breakdowns, weight-sensitivity sweep, contamination/retry-rate
  tables). Everything it needs is already computed and saved per-record
  by `eval/run_agent_pipeline.py`; this module would aggregate across
  records, not needed yet at n=20.
- No CLI entry point for the data-plane/control-plane side yet;
  everything runs via `python -m eval.<module>` or
  `python -m eval.run_agent_pipeline`.

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

## Control plane: CICFlowMeter extraction

`controlplane/extractor.py` (+ `controlplane/record.py`): when the
selector escalates a flow, pull that flow's own packets back out of the
source PCAP and run them through CICFlowMeter to get the full feature
record, anonymise, and package as an `EscalationRecord`.

**Backend: the `cicflowmeter` Python package (PyPI 0.5.0), not Java —
but not through its own CLI either.** Two real bugs found in the
package's CLI/live-sniffing layer, neither in its actual feature
computation:

1. `cicflowmeter.sniffer.main()` calls `create_sniffer(...)` with
   positional arguments in a different order than `create_sniffer`'s
   own signature — `args.fields` lands in the `input_directory`
   parameter, `args.verbose` (always `True`/`False`, never `None`)
   lands in the `fields` parameter, so `fields.split(",")` crashes
   unconditionally regardless of what CLI flags are passed.
2. Even calling `create_sniffer` correctly, its `AsyncSniffer(offline=...)`
   path shells out to `tcpdump` for offline BPF filtering, which isn't
   installed in this environment (Windows, WinPcap only).

Both are CLI/sniffer plumbing bugs, not drift in CICFlowMeter's actual
feature computation (`FlowSession.process()` / `Flow.get_data()`) — so
rather than falling back to Java, `extractor.py` drives those classes
directly with a plain scapy packet iterator (same approach already used
for `adapters/pcap_adapter.py`'s test-only scapy reference), which
works correctly. Verified against the project's own 60s test fixture:
sane per-flow statistics, packet counts matching the flow table's own
count exactly. No Java CICFlowMeter JAR was found anywhere in this
environment and building `ahlashkari/CICFlowMeter` from source (Maven,
native pcap bindings) was out of scope given the Python path already
works — `run_cicflowmeter_java` is a real, wired interface for if the
Python port's drift ever gets worse than what's already worked around
here, not something exercised in this run.

**Feature count: 77, not 78.** This port's raw output has 82 columns;
anonymisation strips 5 identity columns (`src_ip`, `dst_ip`, `src_port`,
`protocol`, `timestamp`) and keeps `dst_port`, leaving 77 numeric
features — a couple short of the traditional CICFlowMeter "78" quoted
in CICIDS2017-era papers (this port's own schema, reported honestly
rather than padded). All 77 are sent to every record, none pruned by
importance — pruning toward known-attack feature importance is exactly
the wrong prior for a selector whose premise is catching something
novel.

**Anonymisation is allow-list, not deny-list**
(`controlplane/extractor.py::anonymize_features`): built by explicit
inclusion (every numeric feature column, plus `dst_port` alone from the
identity-shaped columns), not by deleting a known-bad list — nothing
can pass through that wasn't explicitly decided to. Enforced
independently in `tests/test_extractor.py` by a regex sweep
(`assert_no_leaked_identity`) over every emitted record's full
serialized form (features, trigger reasons, packet window, flow_id),
checked against deliberately-unanonymised input first to confirm the
checker actually catches real leaks rather than trivially passing.
`flow_id` is a fresh random token (`uuid4`), not a hash of the 5-tuple —
a hash is one-way in principle but IPv4/port space is small enough to
be realistically reversible by brute force; a random token carries no
such risk. `EscalationRecord` has no label field at all (not just an
empty one) — ground truth lives in the eval harness, keyed by
`flow_id`, kept in a separate manifest file, never inside the record
sent to agents. 22 tests in `tests/test_extractor.py`.

## Bot IAT-regularity experiment — hypothesis tested and falsified

Hypothesis: Bot recall sits at ~2-3% because Tier-1 features can't see
C2 beaconing, and botnet C2 is periodic — low IAT variance relative to
the mean should separate it, now that real microsecond timing is
available.

**Implementation** (switch-implementable, integer-register only): added
one new O(1) `FlowState` register, `iat_sum_sq` (sum of squared IATs —
a P4 switch accumulates `x*x` in a register exactly as it does `x`
itself, still integer, still no floats or running means stored).
Derived at check time as `flow_iat_regularity = S² / (Q·n)` where
`S = iat_sum`, `Q = iat_sum_sq`, `n = iat_count` — algebraically
equivalent to `1 / (CoV² + 1)`, avoiding a switch-unfriendly `sqrt`,
bounded in `(0, 1]` and saturating at exactly `1.0` for a perfectly
regular beacon (reuses the project's existing saturation-aware
`FeatureKind.BOUNDED_RATIO` fitting, the same machinery already built
for `syn_ratio`/`rst_ratio`). Reported inverted (high = regular) purely
so it fits the existing high-side-anomalous threshold convention with
zero changes needed elsewhere. `FEATURE_SCHEMA_VERSION` bumped 3→4;
both PCAP-path feature caches regenerated (Monday: 462.97s wall clock,
566,864 flows; Friday: 464.57s, 677,963 flows — both reproduced their
prior closure/concurrency numbers exactly, confirming the regeneration
changed nothing but the new feature).

**Result: falsified, cleanly, not a threshold-tuning problem.** At the
5%-benign-rate operating point (p98.376), `flow_iat_regularity`'s
crossing rate is **0.0000%** on Bot (n=2,901), 0% on PortScan, 0% on
DDoS, vs. 0.0663% on held-out benign — and Bot recall is bit-for-bit
identical with the feature included vs. excluded from the `k_of_n(2)`
rule (53/62/87 escalated at the 1%/5%/10% benign-rate points, both
ways). Checked the raw value distribution too, not just the fitted
threshold, to see whether this was a threshold-placement problem rather
than a real absence of signal — it isn't:

| class | n | defined | mean | median | p99 |
|---|---|---|---|---|---|
| Bot | 2,901 | 25.2% | 0.142 | 0.153 | 0.164 |
| PortScan | 158,945 | 0.57% | 0.515 | 0.529 | 0.801 |
| DDoS | 81,075 | 98.8% | 0.109 | 0.101 | 0.217 |
| BENIGN (Friday) | 434,605 | 62.6% | 0.336 | 0.334 | 0.952 |

Bot's mean regularity (0.142) is *lower* than benign's (0.336) — the
opposite direction from the hypothesis, not just a miss. Two things
stand out: **only 25.2% of Bot flows even have enough packets (≥3) for
`flow_iat_regularity` to be defined at all**, vs. 98.8% for DDoS —
most Bot conversations are too short-lived at this project's own
flow-table granularity to characterize their timing at all, which is
itself informative about *why* Bot is hard here, separate from whether
this particular feature helps. And PortScan — not the class this was
built for — shows by far the most regularity (mean 0.515, p99 0.80),
consistent with scan tooling's characteristically fixed inter-probe
delay; doesn't move PortScan's already-~99.7% recall, which other
features already carry.

**Conclusion, per the standing instruction:** Bot is reported as a
genuine limitation of this Tier-1 + per-source feature set, not chased
further. The CICIDS2017 "Bot" label (ARES botnet) likely covers varied
activity beyond simple periodic beaconing (interactive commands, file
transfer, screenshot exfiltration) that wouldn't show this signature
even if this feature had been perfectly built — and a large fraction of
Bot flows are too short-lived for any per-flow IAT statistic, this one
included, to say much of anything.

## EscalationRecord sample for the agent pipeline

`results/escalation_records_sample.jsonl` — 200 anonymised
`EscalationRecord`s from Friday, exactly 50 each of Bot/PortScan/
DDoS/BENIGN (stratified sample from the 5%-benign-rate operating
point's escalated flows: 62/158,597/80,037/52,794 available per
class before sampling — Bot's small pool is itself consistent with
its low recall). Selector config fitted on Monday PCAP fit-half,
`k_of_n(k=2)`, `exclude_low_confidence=False`, config_hash
`9581bdb7e3859338`. Every record: 77 features, real trigger reasons
from the fitted config, packet-window provenance, no ground-truth
label. All 200 independently re-verified against
`assert_no_leaked_identity` (the same regex-based check
`tests/test_extractor.py` uses) after writing to disk — passes.

Ground truth for whoever evaluates the agent pipeline against this
sample lives separately in `results/escalation_records_sample_manifest.INTERNAL.jsonl`
(flow_id → real 5-tuple + label) — named `.INTERNAL` deliberately, not
meant to ship alongside the sample file itself.

Generation took ~11 minutes end to end (546.8s fresh Friday
simulation with the selector evaluated inline per flow, 71.4s packet
re-collection for the 200 sampled flows, plus CICFlowMeter per flow) —
not currently cached; re-running regenerates from scratch.

## Five-agent zero-day review pipeline

`agents/` implements the architecture from the brief: A1 (evidence) →
A2 (behaviour) → A3 (hypotheses) sequential chain, A4 (blind replica,
no chain output, runs concurrently) independent second opinion, A5
(verdict) sees both sides plus their mechanical agreement. Module
layout matches the original spec (`base.py`/`schema.py`/`validators.py`
/`contamination.py`/`verification.py`/`trust.py`/`pipeline.py`, one
prompt file per agent per version under `prompts/`). Every claim is
structured (closed `relation` vocabulary, named `referenced_features`)
specifically so trust scoring can check claims against the record in
code rather than trusting an LLM to grade itself.

**Stop points 1 and 2 are done** (schema on 2 records, then 20 records
stratified 5/class) — done four times over, as real problems surfaced
at each stage and forced a redesign before scaling further. Stop point
3 (the full 200-record run) has not started.

### Model selection — measured live, not assumed

The original ask was Anthropic; switched to Gemini free tier mid-build
(no funding for this project). "gemini-3-flash" doesn't exist as a
model ID. Live investigation against this project's key found:

- `gemini-2.5-flash`, `gemini-2.0-flash`, and both their `-lite`
  variants: all **404, retired** ("no longer available to new users"),
  not a quota issue — the free tier's 2.x generation isn't reachable on
  this key at all, only 3.x is.
- `gemini-3.5-flash`, `gemini-3.6-flash`, `gemini-3.8-flash`: all
  throttled to **5 RPM**; `gemini-3.5-flash` additionally hit a hard
  **20 requests/day** wall mid-run (confirmed via a live 429 naming
  `GenerateRequestsPerDayPerProjectPerModel-FreeTier`, quotaValue 20 —
  not the ~1,500/day originally assumed, ~75x lower).
- `gemini-3.5-flash-lite`: **15 RPM** (measured via live 429), and no
  daily-quota 429 after 269+ successful calls in one day and counting.
  This is the one model on this key that's neither retired nor
  artificially throttled — set as `agents.base.DEFAULT_MODEL`.

`agents/base.py`'s rate limiting (token bucket, 12 RPM — under the
measured 15), daily-quota tracking (per-model, since the quota itself
is per-model), and retry/backoff logic (honors the API's own
`retryDelay` hint; treats `httpx.TransportError` — e.g. a live DNS
failure that once killed a full 20-record run mid-batch — as retryable
alongside 429/503; fails fast with no retry on a daily-quota 429, since
no amount of waiting inside one run fixes a same-day cap) were all
built from these live measurements, not documentation. The exact daily
ceiling for `gemini-3.5-flash-lite` is still not pinned down — finding
it would cost real quota for no benefit once "clearly high enough" was
established — `DAILY_QUOTA_BY_MODEL` carries a placeholder (400) above
the observed floor, tightened only if a real 429 is ever seen.

### Real bugs found and fixed along the way

1. **Trigger-reason / CICFlowMeter unit collision.** The selector's
   Tier-1 trigger reasons (e.g. `flow_duration`) and the CICFlowMeter
   `features` dict can share an identical bare name while disagreeing
   by orders of magnitude — confirmed live, exactly **1,000,000x** for
   `flow_duration` on one real record (trigger reasons are in
   microseconds, CICFlowMeter output in seconds). This silently broke
   evidence-checking (A1's genuinely correct claims were flagged as
   hallucinated/factually wrong ~78% of the time before the fix) and,
   separately, confused the model itself into asserting the *ratio*
   (e.g. "1.71") as if it were the observed value in 2/20 records.
   Fixed on both sides: prompts now show trigger-reason names under a
   `trigger_` prefix as a permanently distinct namespace, with
   observed_value/threshold/ratio_to_threshold on separate labelled
   lines; evidence-checking accepts a match against either namespace.
2. **Nearest-neighbour distance normalisation exploded on sparse
   features.** Scaling by `(p90 - p50)` collapses to ~0 for Tier-1
   features that are 0 for almost all benign flows (e.g.
   `syn_without_synack_count`), turning a real observed value into a
   normalised distance in the hundreds of billions and corrupting both
   which flow gets picked as "nearest" and which features get reported
   as "most different". Fixed by scaling on `(max - p50)` instead
   (`controlplane/reference.py`), which can't collapse the same way.
3. **Contamination/flow-level filters used naive substring matching.**
   `"uri"` as a bare substring matches inside "**duri**ng" and
   "sec**uri**ty" — would have rejected good flow-level predictions for
   the wrong reason. Fixed to word-boundary regex (same fix already
   applied once to `agents/contamination.py`, then re-discovered in
   `agents/validators.py`'s separate filter).
4. **Only API-level error text was treated as retryable.** A live
   `httpx.ConnectError` (transient DNS failure) crashed a full
   20-record run instead of retrying like a 503 would. Fixed by
   catching `httpx.TransportError` by type, not just known error
   strings.

### The redesign story — three rounds on the same 20 records

**Round 1 (initial build → stop point 2):** verdict was a categorical
3-way label asserted directly by A5. Result: 0/20 records reached
`anomalous_and_unexplained` — the chain never once landed a positive
detection, even on Bot/PortScan/DDoS records. Diagnosis: A5 was
grading its own homework with no ground truth to check hypotheses
against, and nothing forced it to reckon with contradicting evidence.

**Round 2 (four structural fixes):** (1) unit-collision fix above; (2)
calibration context added to A1/A3/A4/A5 prompts (what a "threshold"
statistically means); (3) `benign_plausibility` in `[0,1]` replaces the
categorical verdict, three-way label now derived in code
(`agents.schema.derive_verdict`, swept, not asserted); (4) hypotheses
require a falsifiable `prediction` + `referenced_features`, checked
against a banned-concept filter (payload/header/user-agent/URI — this
pipeline has no application-layer data, so a prediction about any of it
is unfalsifiable in principle). Result: **PortScan separated cleanly**
(0.10–0.25 vs 0.70–0.85 for everything else) — the first real signal.
DDoS and Bot did not separate from BENIGN at all. Root cause found on
inspection: a source opening 4,614 flows got reframed by A4 as
*supporting* an "automated polling" hypothesis rather than flagged as
contradicting anything — the falsifiable-prediction mechanism had
nothing to grab onto because upstream agents simply didn't flag strong
evidence as contradicting in the first place (only 5/133 hypotheses
ever flagged any contradiction at all).

**Round 3 (two more structural fixes, still no ground truth involved):**
(1) every claim any hypothesis's own author flags as contradicting it
is collected into a `required_claim_ids` set and enforced in code — A5
must cite every one of them or the response is invalid and retries
(`agents.validators.validate_a5_addresses_all_contradictions`); (2)
predictions must reference real flow-level features, mechanically
rejected otherwise (`FLOW_LEVEL_FILTER` in `agents/validators.py`).
Result: enforcement worked perfectly (0/20 first-attempt failures) —
but `benign_plausibility` for BENIGN/Bot/DDoS all moved *up*, not down.
Diagnosis: agents had no data-grounded sense of what "normal" looks
like on this network, so a hypothesis could sound plausible and cite
real claims while still being empirically nonsense.

**Round 4 (empirical grounding — the redesign that actually worked):**
built `controlplane/reference.py` (Monday benign reference distribution
— p50/p90/p99/p99.5/max/n per Tier-1 feature, computed from the
already-cached full-population Monday PCAP parquet, 566,864 real benign
flows — **deliberately scoped to the Tier-1/per-source namespace only**,
not the 77-feature CICFlowMeter space, since building that reference
would mean running full per-flow CICFlowMeter extraction against
566,864 flows, a multi-day job not attempted) and `agents/grounding.py`
(mechanical hypothesis testing: query real benign flows against each
hypothesis's `predicted_feature_profile` *before* A5 sees it, attach
match count + how many matches are within 2x of this flow's own
observed values; nearest-10-benign-flow grounding in normalised feature
space; a hard code-enforced cap — `benign_plausibility` cannot exceed
0.3 when the credited hypothesis has zero matching benign flows,
applied after the model responds, looked up against the real computed
support, never the model's own restatement of it). Result:

| class | benign_plausibility (5 records each) | mean |
|---|---|---|
| BENIGN | 0.85, 0.85, 0.85, 0.85, 0.95 | 0.87 |
| Bot | 0.85, 0.85, 0.85, 0.85, 0.95 | 0.87 |
| DDoS | 0.15, 0.15, 0.35, 0.45, 0.85 | 0.39 |
| PortScan | 0.00, 0.00, 0.10, 0.10, 0.25 | 0.09 |

PortScan: still perfect separation. **DDoS now partially separates**
(4/5 records ≤0.45 — a real change from complete non-separation in
round 3). **Bot still doesn't separate from BENIGN at all** — this is
now the *third* independent confirmation (alongside the PCAP-vs-CSV
sweep and the IAT-regularity experiment above) that Bot is a genuine
ceiling for this Tier-1 feature set, not a pipeline defect. Zero
code-enforced clamps fired on this run — the empirical numbers in the
prompt already nudged A5 enough on their own; the cap remains as a
verified-working backstop, not the primary mechanism. Contamination:
A4 dropped to 0/20 (was 3/20); A3 rose slightly to 4/20 (`denial of
service`, `malicious`, `port scan`, `botnet`, `ddos`, `attack` — now
reasoning more confidently about genuinely extreme statistics and
reaching for category words to explain why).

Results for all four rounds on the same 20 records:
`results/agent_pipeline_20.jsonl` (round 1) /
`agent_pipeline_20_v2.jsonl` (round 2) / `agent_pipeline_20_v3.jsonl`
(round 3) / `agent_pipeline_20_v4.jsonl` (round 4, current). Current
prompt versions in use: `a1_evidence_v4`, `a2_behaviour_v1` (never
needed a revision — no trigger reasons or hypotheses in its scope),
`a3_hypotheses_v4`, `a4_replication_v4`, `a5_verdict_v4`.

## Zero-cost ablation before scaling to 200

Before spending any of the n=200 budget, checked whether the round-4
agent pipeline's separation is actually coming from the agents, or just
from the mechanical `nn_dist` grounding signal every agent prompt
already gets handed (`controlplane/reference.py`). Pure re-analysis of
the existing `results/agent_pipeline_20_v4.jsonl` — no new API calls.
Script: `eval/ablation_v4.py` (`python -m eval.ablation_v4`); full
tables and reasoning: `results/ablation_v4_report.md`.

**Per-class recall at 0% benign false-positive rate (n=5/class, so this
is the only fully clean operating point available):**

| detector | Bot | DDoS | PortScan |
|---|---|---|---|
| `nn_dist` alone | 0% | 40% | 100% |
| `close_to_observed_count` alone | 40% | 20% | 80% |
| combined (`nn_dist` + `close_to_observed_count`) | 0% | 60% | 100% |
| full agent pipeline (`benign_plausibility`) | 0% | 80% | 100% |

**PortScan: `nn_dist` alone matches the full pipeline exactly (100%
both) — the agents add nothing measurable here.** PortScan's `nn_dist`
values (6.6-27.8) sit far outside the benign cluster (max 1.52), so a
mechanical distance check is already sufficient.

**DDoS: `nn_dist` alone does not come close (40% vs. 80%), and it's
structural, not a threshold-tuning problem.** DDoS's `nn_dist` values
cluster at 1.31-1.63, and one real BENIGN flow (`db6f4a10be5e...`,
`nn_dist=1.52`, `bp=0.85`) lands inside that same cluster — no threshold
on `nn_dist` alone can separate DDoS from benign without either missing
most DDoS or flagging that benign flow. The full pipeline correctly
calls `db6f4a10` benign (its empirical grounding found 6,820 matching
benign flows for the credited hypothesis) where raw distance alone
cannot. This is the strongest evidence in this ablation that the agents
earn their keep, and it's class-specific: true for DDoS, not true for
PortScan.

**Disagreement check (`nn_dist > 1.0` vs. `bp < 0.5`): only 2/20 records
disagree, both in the same direction (agents override distance toward
"benign"), but they split 1-1 on correctness** — `db6f4a10` (BENIGN) is
a correctly-filtered distance false positive, but `ecac22cfd0d6`
(**DDoS**) is an *incorrectly* overridden distance true positive, i.e. a
miss. At n=20 this is too small to support "the agents are a
false-positive filter, not a detector" as a general claim — that split
(FP-corrections vs. induced misses) needs the n=200 run to actually
check, not just to observe that both cases exist. The DDoS recall gap
above is the better-supported claim to lead with.

**`chain_vs_independent` for this run (not in any prior summary):**
mean **+0.060**, positive (chain underperforms A4's blind pass) in
**15/20 records (75%)**. Consistent with A2's low corroboration rate
(`V` mean 0.267, min 0.000, max 0.800) already flagged in the redesign
notes above — on this run, the sequential A1->A2->A3 chain costs
accuracy more often than it adds it, by the project's own metric for
that comparison. Belongs in the writeup as a real finding, not a
footnote.

**Bot: closed, a fourth independent confirmation, not chased further.**
Every detector tested here — `nn_dist`, `close_to_observed_count`, their
combination, and the full pipeline — fails to separate Bot from BENIGN
at 0% FPR. Same already-characterised mechanism as the IAT-regularity
experiment above: ~75% of Bot flows are too short-lived for per-flow
timing statistics to be defined at all. Reported as a characterised
limitation; no further Bot-specific experiments planned.

## Open questions

- **Is Bot/Web-Attack detection genuinely near-impossible from Tier-1 +
  per-source flow statistics, or is it another artifact? RESOLVED for
  Bot, for PCAP-derived data: genuine ceiling, not an artifact.** Two
  independent lines of evidence now agree: (1) the PCAP-vs-CSV sweep
  comparison (see "PCAP-based fitting, sweep, and Friday session
  labeling" above) showed Bot recall flat at ~2-3% on *both* paths even
  though PortScan and DDoS recovered dramatically once coarse timing and
  the bounded-fitting bug were both fixed, on a *larger* real-PCAP Bot
  sample (2,901 vs. 1,966 flows) than the CSV path had; (2) the
  IAT-regularity experiment directly above specifically targeted Bot's
  C2-beaconing pattern and found no signal, in the wrong direction, with
  a mechanical explanation (Bot flows are too short-lived). Still open
  for Web-Attack classes specifically, which haven't been re-checked
  against the fixed pipeline (CSV-only classes, no PCAP session for
  them — Thursday's PCAP was never part of this project's brief).
- **Low-confidence IAT ablation (slowloris/Slowhttptest specifically):**
  not available yet — the re-run that would answer this (coarse-timestamp
  IAT features included vs excluded from fitting, per-class recall on
  the two slow-attack classes that plausibly depend most on IAT) was
  still in progress when this file was written. `eval/run_all.py`
  already prints and saves this comparison
  (`results/low_confidence_ablation_delta.csv`) — just needs the sweep
  to finish.
- **Does the agent pipeline's DDoS separation and PortScan separation
  hold at n=200, or is n=5/class too small to trust?** Round 4's numbers
  above are the whole basis for believing empirical grounding works;
  stop point 3 (full 200-record run) is what would actually confirm it.
  Also open: whether the single-LLM baseline (`agents/baseline.py`,
  built, never run live) does anywhere near as well without the
  five-agent structure and empirical grounding — without that
  comparison there's no basis yet for claiming the architecture earns
  its cost, which was flagged as the first thing a reviewer would ask
  back when the baseline was originally specced.
- **Bot stays unseparated in the agent pipeline too** (mean
  `benign_plausibility` 0.87, identical to BENIGN's 0.87) — consistent
  with, not independent from, the data-plane's own Bot ceiling above
  (the agent pipeline only ever sees flows the selector already
  escalated, so if Tier-1 features can't distinguish Bot from benign at
  the selector stage, no amount of downstream reasoning can recover
  that signal). Not chased further for the same reason as above.

## 200-record run, re-stratified (in progress)

Regenerated the escalation sample at 70 BENIGN / 50 DDoS / 50 PortScan /
30 Bot (`eval/generate_escalation_sample.py` -- the original 50-each
generator was a throwaway script, rebuilt from scratch reusing the
existing PCAP-sweep/labels/extractor pieces). Same operating point
*specification* as the original sample (Monday PCAP fit-half, 5%
target held-out benign rate -> percentile 98.3762, k_of_n(k=2),
exclude_low_confidence=False) -- the refitted config_hash
(`a7ade29536b5ef9d`) does not match the original's recorded
`9581bdb7e3859338`, and none of the 20 swept percentiles reproduce it
either, so the original (lost, uncommitted) generator's exact threshold
values can't be exactly reconstructed. Old 50-each sample backed up as
`results/escalation_records_sample.v1_50each.jsonl` (+ manifest), not
deleted.

`eval/run_agent_pipeline.py` had a real resumability bug, found while
resuming this run: re-running the same command re-appended every
already-cached record to `--out` as a duplicate line (caching prevented
re-paying for the call, but not the duplicate write). Fixed in both
`run_agent_pipeline.py` and `run_baseline_pipeline.py` by checking
`--out` for already-written record_ids up front and skipping them
entirely. Also added a catch-all exception handler around the per-record
loop (a sustained network outage can still exhaust `base.py`'s own
retry budget and escape as a raw `httpx.ConnectError` -- hit this live,
mid-run) so any unexpected failure stops the run cleanly with a
"rerun to resume" message instead of a traceback.

Progress: **139/200** the first time this section was written, **158/200
and climbing** as of this update (DDoS 50/50, BENIGN 70/70, PortScan
38/50, Bot 0/30) -- the run is live in the background right now (resumed
after a UTC quota reset; daily count 94+/400 this cycle), run in DDoS ->
BENIGN -> PortScan -> Bot order per instruction. One BENIGN record that
failed A3 schema validation on the first pass (see prior paragraph)
succeeded on a later retry, hence 70/70 not 69/70. Daily quota
(`gemini-3.5-flash-lite`, 400/day) is the binding constraint;
`eval/run_agent_pipeline.py` and `eval/run_baseline_pipeline.py` (the
still-outstanding single-LLM baseline, `agents/baseline.py` /
`BaselineResponse` / `baseline_v2` prompt -- now includes the same
empirical grounding block A1/A3/A4/A5 get, asks directly for a
continuous `benign_plausibility` instead of a categorical verdict, so
it's head-to-head comparable to A5) both resume for free via cache --
rerun the same command after each UTC quota reset until both are done,
then re-run `eval/ablation_v200.py` (the n=200 successor to
`ablation_v4.py`: adds the baseline comparison, a full operating-point
sweep instead of just 0%-FPR, and per-class trust/chain_vs_independent
breakdowns) -- see the caught-and-fixed bug in the next section, numbers
computed with `ablation_v200.py` before that fix are wrong.

## Self-reported trust triad + fault-injection poster study (built, not yet run)

For a professor-specified poster deliverable, separate from the
detection-accuracy work above: three figures on how trust degrades
along the agent chain under deliberate fault injection, not a detection
result.

**Self-reported trust (agents/schema.py, prompt versions bumped to
a1_evidence_v5 / a2_behaviour_v2 / a3_hypotheses_v5 / a4_replication_v5
/ a5_verdict_v5):** every agent's response now carries `confidence`,
`evidence_support`, `verification` (each self-rated [0,1], shared
wording in `agents/prompts/render.py::SELF_REPORT_INSTRUCTIONS` /
`_A5`) alongside the existing code-computed C/E/V (`agents/trust.py`,
unchanged). A1-A4 are answering a genuine blind guess for
`verification` (they never see A4/the chain); A5 is shown the real
mechanical corroboration first, so its self-report has something to be
grounded in. Old cache entries for the pre-bump prompt versions are
simply orphaned (never looked up again), not invalidated in place --
same pattern this project has used for every prior prompt-version bump.
**Not backfilled onto the 139 already-done 200-record-run records** (that
would cost ~700 extra calls for no benefit to the detection-ablation
work, which doesn't use these fields) -- they'll simply lack
self-reported scores; only the still-outstanding portion of that run
and the fault-injection study (below) get them.

**Fault injection (`eval/fault_injection.py`, 25 unit tests in
`tests/test_fault_injection.py`):** four deterministic corruptions of
already-cached CLEAN agent output, applied in code, no API call for the
corruption itself --

- `missing_evidence` (A1): drop a random 50% of claims, seeded per
  record (`random.Random(f"{record_id}:missing_evidence")`).
- `incorrect_behavior` (A2): invert `greater_than_typical` <->
  `less_than_typical` on every referenced feature; statements/confidence
  untouched.
- `hallucinated_hypothesis` (A3): append one fabricated hypothesis
  (prior_plausibility 0.9) whose predicted_feature_profile names
  `trigger_process_creation_rate` -- deliberately host-telemetry-shaped
  and real nowhere in this project's vocabulary, since the professor's
  original prompts assumed process/command/auth-event telemetry this
  pipeline was never given (flow statistics only) -- see the "Notes"
  in the originating instruction for why that wording is avoided
  everywhere else in this project's real prompts.
- `faulty_verification` (A4->A5 link): flips corroborated <->
  contradicted in the mechanical chain-vs-A4 corroboration summary A5
  is shown, WITHOUT touching A1-A4's own output at all -- this needed a
  new `verification_override` param on `agents.a5_verdict.run` (default
  `None`, preserves every other caller's real computation).

`agents/base.py`'s cache key gained a `fault_condition` parameter,
backward-compatible by construction: only appended to the hashed string
when it's not `"clean"` (`CLEAN_FAULT_CONDITION`), so every call made
before fault injection existed still resolves to the exact same key
(verified in `tests/test_base.py`, including an end-to-end
"clean and faulted never collide" test through `call_structured`
itself, not just `_cache_key`).

`agents/pipeline.py` was refactored (behavior-preserving, same
435->464-passing suite before/after) to expose `compute_grounding`,
`compute_hypothesis_support`, `compute_trust_and_decay`,
`compute_contamination` as standalone functions, so
`eval/fault_injection.py`'s four condition-specific rerunners
(`run_missing_evidence` / `run_incorrect_behavior` /
`run_hallucinated_hypothesis` / `run_faulty_verification`) reuse the
exact same wiring `run_record` uses instead of duplicating it.

`eval/run_fault_injection.py`: the runner. Uses the SAME 20 flow_ids as
the original v4 run (pulled from `results/escalation_records_sample
.v1_50each.jsonl`, cross-referenced against `agent_pipeline_20_v4.jsonl`
-- confirmed all 20 present) for comparability. Real cost is **15
calls/record** (5 for a fresh CLEAN run under the new prompt versions +
10 for the four fault reruns' `4+3+2+1` downstream agents), i.e. **300
calls total**, not the ~200 originally estimated for the fault reruns
alone -- that estimate didn't count the clean baseline, which can't be
skipped or reused from the old v4 cache (v4 predates the self-reported
schema entirely). Resumable the same way as the other two runners
(checks `--out` for already-written (record, condition) pairs).

**Poster figures (`eval/poster_figures.py`, 7 tests in
`tests/test_poster_figures.py`):** reads the combined
`results/fault_injection_20.jsonl` (once it exists) and produces, at
300 dpi with poster-sized fonts:

- Fig 1: self-reported trust score (weighted C/E/V, same
  `DEFAULT_WEIGHTS` as `agents/trust.py`) vs agent stage, one line per
  condition (Clean + the four fault conditions).
- Fig 2: grouped bars, x = degraded agent (A1-A4, from
  `CONDITION_TO_DEGRADED_AGENT`), three bars (decision error rate /
  false escalation rate / missed detection rate -- each a
  clean-vs-faulted *transition*, e.g. decision error = correct-under-clean
  AND incorrect-under-fault, not just "incorrect under fault").
- Fig 3: confidence-verification self-reported gap (pooled across every
  agent and every condition, clean included -- clean anchors the
  low-gap/low-error end, faults populate the high-gap end) binned into
  9 bins, scatter + OLS fit, R² reported and flagged as weak (dashed
  line) below 0.3 rather than presented as a confident fit either way.
- Fig 4 (bonus, Part 1's own request: "probably deserves a fourth
  figure"): self-reported vs code-computed evidence_support/
  verification per agent, plus a CSV of correlation + mean-absolute-
  difference per (agent, metric) pair. A5 structurally has no
  code-computed E/V to compare against (it produces no claims to check
  mechanically) -- reported as n=0, not silently dropped.

Every figure's underlying numbers are also written to CSV in the same
output directory (`results/figures/` by default) so they can be
replotted without re-running anything.

**Tested against a synthetic placeholder dataset** built from the real
`agent_pipeline_20_v4.jsonl` (`tests/test_poster_figures.py`'s
`placeholder_dataset` fixture: v4's own code-computed C/E/V copied into
synthetic self-report fields with a deterministic per-condition offset,
since v4 predates the self-reported schema and has only the clean
condition) -- this validates the plotting/statistics code paths run
end to end and produce sane shapes, NOT that any number in it means
anything; confirmed visually (all three figures render correctly,
legends/axes/fit line all present) but not a stand-in for the real run.

**Not run yet** (no API calls made for any of this): `eval
.run_fault_injection.py` itself. Per the standing cost-priority note,
if quota runs short the figures matter more for the poster deadline
than finishing Bot's last 30 records in the 200-record run above --
Bot is already a characterised limitation from four independent
confirmations (PCAP-vs-CSV sweep, IAT-regularity experiment, round-4
agent pipeline, and this ablation's own mechanical-detector check).

## Presentation figures + summary (built, zero API calls)

Separate, time-boxed deliverable for an upcoming presentation (2-4 day
deadline): `eval/presentation_figures.py` produces five 300dpi PNGs into
`results/figures/` plus `results/PRESENTATION_SUMMARY.md` (one section
per finding, headline number + n + plain language each), entirely from
already-written results -- no new API calls, run alongside the still-
live 200-record pipeline run in the background.

**Fig 1 (data-plane recall vs. escalation) required regenerating the
step-1 sweep first.** The CSVs already on disk
(`results/recall_curve.csv`, `results/summary.md`, etc., dated Sep 9)
predate the two fitting bugfixes in "Bugs just fixed" above and read
PortScan recall as ~0.02-0.14% -- the bug's own signature, not a
finding, and already flagged as stale in this file. Regenerated via
`eval.run_all_extractions` (all 8 CICIDS2017 days x 2 key modes, current
`FEATURE_SCHEMA_VERSION="4"` -- forces fresh extraction since the CSV-
path cache only had features2/features3 on disk, ~35 min total, pure
data-plane compute) then `eval.run_all` (fit+sweep+write, fast once
cached). **Cross-checked against this file's own already-validated
PCAP-vs-CSV comparison table (the "PCAP-based fitting..." section
above) and matches exactly** -- Bot 2.7%/3.0%/3.3% and PortScan
0.14%/0.47%/0.77% at 1%/5%/10% benign rate, confirming the regeneration
is correct, not just different. PortScan's low CSV-path recall is real
and already-documented (CSV-synthesized timing can't carry the fine-
grained signal PortScan needs -- the identical selector gets ~99.7% on
real PCAP captures per that same section), not a new problem.

**A second real bug, caught while building Fig 3:**
`eval/ablation_v200.py::at_matched_benign_fpr` picked the wrong end of
the threshold sweep for "low-side" detectors (`close_to_observed_count`,
and `benign_plausibility` for both the full pipeline and the baseline)
-- it always chose the *smallest* threshold in the sweep regardless of
which direction that detector's escalation rule actually moves in,
which for these three detectors is exactly the dataset's own minimum
observed value: at that point nothing is strictly below it, so BENIGN
and every attack class simultaneously read 0% recall, independent of
what was actually achievable within the FPR budget. `nn_dist`/`combined`
(the other direction) were unaffected and already correct. Fixed by
making the function direction-aware (`direction="high"|"low"`, picking
`candidates[0]` vs `candidates[-1]` respectively) -- documented in the
function's own docstring since this is exactly the kind of bug that's
invisible unless you already know to check. Any number quoting
`close_to_observed_count`/`benign_plausibility` recall from
`ablation_v200.py` before this fix is wrong; `ablation_v4_report.md`
(the earlier, hand-inspected n=20 report) was never affected since it
didn't use this function.

Figures 2-5 read `results/agent_pipeline_200.jsonl` directly and are
current as of whenever last run -- **159/200** as this section is
written (BENIGN 70/70, DDoS 50/50, PortScan 39/50, Bot 0/30 -- Bot not
shown in any figure, nothing to show yet). Every input path is a CLI
flag (`--pipeline`, `--manifest`, `--recall-curve`,
`--per-class-benign-rate`, `--out-dir`); the defaults already point at
the same files the pipeline run appends to in place, so re-running with
no arguments at all picks up however many records have landed by then --
`python -m eval.presentation_figures` is the full regeneration command,
safe to run again once the remaining ~41 records (11 PortScan + 30 Bot)
land. The run stopped for today at 159/200 on a LIVE 429 even though
this process's own tracked count was only 104/400 for the (UTC) day --
the 429 body named the real cap explicitly, quotaValue 500, and the
mismatch points at a quota-window reset boundary that isn't UTC
midnight the way this project assumed; see `agents/base.py`'s
`DAILY_QUOTA_BY_MODEL` comment (now 500, was a 400 placeholder) for the
full note. Not investigated further tonight -- rerunning immediately
would likely just repeat the same live 429, since the server-side
window is evidently still the one that's exhausted.

Headline findings (see `PRESENTATION_SUMMARY.md` for the full plain-
language writeup): DDoS is the one class the data-plane selector alone
recalls well (~62-64%) with no agent involvement; PortScan and DDoS both
show real `benign_plausibility` overlap with BENIGN rather than clean
separation; the full agent pipeline beats every purely-mechanical
detector combination at a realistic (5%) benign-FPR budget, and at
`nn_dist`'s own 0%-FPR threshold it catches 36 real attacks (31 DDoS +
5 PortScan) that raw distance misses entirely, at the cost of reading 23
true-benign records as more suspicious than distance does; A2's
corroboration score (V=0.19) is the weakest link of any agent, uniform
across every class; and the sequential chain underperforms A4's blind
replica in ~70% of records, again uniform across classes -- if anything
worse on PortScan (79%) than DDoS/BENIGN. None of this is new signal
inconsistent with earlier n=20 findings; it's the same story confirmed
at a larger, still-growing n.

## Architecture doc, self-report coverage gap, PCAP counts, threshold sweep

Four more zero-API-call deliverables since the above, plus one blocked
real-cost run. Main pipeline run is still parked at **159/200** (DDoS
50/50, BENIGN 70/70, PortScan 39/50, Bot 0/30) -- not resumed in any of
this work.

**`results/ARCHITECTURE.md`**: full as-implemented description of the
five-agent pipeline, read fresh from the current code (not from memory
or this file) at the user's explicit request -- agent topology (ASCII
diagram), all five system prompts verbatim at current version
(`a1_evidence_v5`/`a2_behaviour_v2`/`a3_hypotheses_v5`/
`a4_replication_v5`/`a5_verdict_v5`), exactly what context each agent
receives (A2 gets no trigger reasons/grounding block/CALIBRATION_NOTE at
all -- the narrowest-scoped agent; A5 never sees the raw grounding block,
only a per-hypothesis derivative of it), output schemas, the trust
formula with weights, verdict-threshold derivation, and eight
in-code-documented divergences from an earlier design (A5's verdict was
originally categorical; hypotheses originally used bare
`referenced_features` instead of checkable ranges; empirical grounding
against real benign traffic didn't originally exist; etc. -- each
citation sourced from an explicit "Fix N"/"redesign spec"/"replaces the
earlier X" comment already in the code, not inferred). **One finding
worth carrying into the poster/presentation directly: the new
self-reported confidence/evidence_support/verification triad is NOT
wired into `agents/trust.py`'s own C/E/V/T formula at all** (except A5's
`confidence`) -- two parallel trust systems currently coexist without
touching each other.

**Self-report coverage in the 159-record run is much thinner than it
looked.** Checked directly: only **20/159 records** (indices 139-158,
i.e. records processed after the schema bump mid-run) have the
self-reported triad on all five agents; the original 139 have it on
none, all-or-nothing per record (the cache key changed with the prompt
version, so there's no partial state). Worse, those 20 are **19
PortScan + 1 BENIGN, zero DDoS, zero Bot** -- an artifact of DDoS/BENIGN
having already finished before the resume that happened to land after
the schema bump. Added two figures to `eval/presentation_figures.py`
anyway (fig6: self-reported trust vs. agent stage, single line since no
fault-injection conditions exist yet; fig7: self-reported
confidence-verification gap vs. decision error) and ran them --
**fig7 came back degenerate**: all 20 records got the correct verdict
(0 decision errors), so the fitted line is exactly `y=0, R²=0`. Not a
bug -- it's the direct consequence of this pipeline handling PortScan
almost perfectly (mean `benign_plausibility` 0.10, tight spread, see
Finding 2 in `PRESENTATION_SUMMARY.md`), so a PortScan-only slice has no
errors for the gap to explain. Recommended dropping fig7 from the
deck until DDoS records exist in the self-reported subset (DDoS is
where the pipeline is sometimes actually wrong).

**`eval/pcap_escalation_counts.py`**: exact escalation counts (not just
rates) from the PCAP-path sweep, at 1%/5%/10%-benign-rate-anchored
operating points, per STATUS's already-fitted config. Cross-checked
against the earlier PCAP-vs-CSV comparison table and matches. Headline
numbers (full breakdown, including precision and per-class miss counts,
in the chat record and `results/pcap_escalation_counts.json`): at the 5%
operating point, Friday escalates 291,490/677,526 flows, 81.9%
precision, PortScan/DDoS recall ~98.7-99.8%, Bot recall 2.1% (62/2,901);
combined with Monday's held-out half, 98.26% attack recall at 9.70% FPR.

**`eval/threshold_sweep.py`**: swept `benign_plausibility`'s verdict cut
over all 159 records (Bot n=0, excluded and stated as such, not silently
dropped). **Result: `LOW=0.3` is not just "near-optimal" -- it ties the
empirical Youden's-J maximum (+0.515) exactly**, because no record's
`benign_plausibility` falls in (0.25, 0.30]; moving the cut to 0.325
immediately costs 8.6 points of benign FPR for +2 points of DDoS
detection. `HIGH=0.7` scores lower (J=+0.471) on the same metric, but
it's answering a different operational question ("safe to fully clear"
vs. "flag as anomalous") so a lower J there isn't evidence it's wrong,
just not directly comparable via this metric. Also reconfirmed from this
angle: even the best possible cut only gets DDoS to 36% detection at
11.4% FPR -- a real class-overlap ceiling, not a threshold-tuning
problem (consistent with Finding 2/3 in `PRESENTATION_SUMMARY.md`).

**Full ablation re-run at n=159 (was n=20), all four detectors matched
to the exact same 11.4% benign FPR (8/70) via `eval/ablation_v200.py`
-- zero API calls, pure re-analysis, checked directly in response to
the question "does nn_dist alone now match the agents' DDoS number?"**
`ablation_v200.py`'s default target-FPR list didn't previously include
0.114 itself (only coarser grid steps), so it was extended to always
include the nearest achievable point to threshold_sweep.py's own
Youden-optimal 8/70, guaranteeing every detector is compared at the
identical real operating point DDoS-36% is quoted from, not a nearby
approximation. Result, DDoS / PortScan (Bot still n=0):

| detector | DDoS | PortScan |
|---|---|---|
| `nn_dist` alone | 6% | 100% |
| `close_to_observed_count` alone | 14% | 67% |
| combined (`nn_dist` + `close_to_observed_count`) | 14% | 97% |
| full agent pipeline (`benign_plausibility`, LOW=0.3) | **36%** | 97% |

**No, `nn_dist` alone does not match or beat 36% on DDoS at n=159 --
the agent layer's justification survives, more clearly than the n=5
number suggested.** `nn_dist` alone is actually worse relative to the
pipeline at scale than the earlier framing implied: 6% vs. 36%, a 6x
gap, at an identical 11.4% benign-FPR budget (not the 0%-FPR point the
n=5 40%-vs-80% comparison used, so the two aren't the same measurement,
but both point the same direction). Even `combined` (`nn_dist` +
`close_to_observed_count`, the best purely-mechanical option available)
tops out at 14% -- the agents still add +22 points on DDoS specifically.
PortScan is the one place a mechanical detector (`nn_dist` alone, 100%)
edges out the full pipeline (97%) at this exact operating point --
consistent with PortScan's already-documented story throughout this
project (a mechanical distance check was always sufficient there; the
agents have never been the thing carrying PortScan).

**Fault injection on a DDoS-heavy selection: built, blocked on quota,
zero calls made.** Per instruction (PortScan's ~0.09 mean
`benign_plausibility` is far enough from any verdict boundary that
corruption is unlikely to ever flip it, so a PortScan-heavy sample
wastes budget), selected 13 DDoS + 5 BENIGN + 2 PortScan from the
current 200-record sample pool -- `results/
fault_injection_selection_ddos_heavy.jsonl` -- reusing
`eval/run_fault_injection.py`'s existing machinery by pointing its
`--v4-results` flag (really just "a flow_id list", despite the name) at
this new selection instead of the original v4 20. **The run hit the
identical live daily-quota 429 immediately, at 0 (record, condition)
pairs written, with this process's own tracked count at only 104/500**
-- the same server-side-window mismatch as the main pipeline's earlier
stall, confirmed a second time, not a one-off. New lead, not yet
confirmed: the failed attempt landed at 06:12 UTC, ~48 minutes before
07:00 UTC -- US Pacific midnight during PDT (UTC-7). Recorded as a live
hypothesis in `agents/base.py`'s `DAILY_QUOTA_BY_MODEL` comment; a
successful call shortly after 07:00 UTC on a day that was refused just
before it would confirm the boundary. The selection and command are
staged and idempotent (0 pairs written means a rerun repeats cleanly,
nothing to skip) -- rerun
`python -m eval.run_fault_injection --v4-results results/fault_injection_selection_ddos_heavy.jsonl --source-sample results/escalation_records_sample.jsonl --out results/fault_injection_ddos_heavy.jsonl`
once quota actually allows it. Real cost when it runs: ~300 calls (20
records x (5 clean + 10 fault-rerun) calls), ~25-30 min at the measured
12 RPM cap.

## Tight percentile x k sweep -- volume problem not solved by tuning alone

`eval/percentile_k_sweep.py`: percentile in {99.5, 99.9, 99.95, 99.99,
99.995, 99.999} x k_of_n's k in {2,3,4,5}, Monday-only fit, measured on
Monday holdout + Friday PCAP (real Bot/PortScan/DDoS labels). Zero API
calls, pure re-analysis of cached parquets. Full grid/plot in
`results/percentile_k_sweep.csv` / `results/figures/percentile_k_sweep.png`.

**Finding: reaching ~1% combined escalation while keeping attack recall
high is not achievable in this grid -- it's a cliff, not a tradeoff
curve.** Every cell at <=1% combined escalation has pooled attack recall
<=0.4% (best: p99.5/k=5, 0.125% escalation, 0.4% recall). Raising k
doesn't cut volume gracefully either: p99.5 k=2->4 barely moves
escalation (28%->19%) while DDoS recall already collapses (97.5%->58.5%);
k=4->5 then craters both escalation AND recall together (19%->0.1%,
73%->0.4%). PortScan survives k increases far better than DDoS (fires 4
correlated features together -- `flows_per_src` + `distinct_dst_ports_per_src`
+ `syn_ratio` + `syn_without_synack_count`, the classic scan signature)
but even it dies by k=5.

**The Monday-to-Friday benign-FPR gap gets relatively WORSE as the
percentile tightens, not better**, even though the raw pp gap shrinks
(both numbers head toward zero): 7.7x (p99.5) -> 47x -> 90x -> 490x ->
1344x -> **infinite at p99.999** (Monday holdout hits exactly 0 false
positives across 283,432 benign flows; Friday still leaks 1.75%
through). You can't out-tighten a distribution shift.

## Two follow-up experiments: single-day baseline vs. mechanism

Per direct instruction, tested whether Monday-only fitting was itself
the problem (re-fit on pooled Monday+Friday-benign data) and whether a
combination-based signature table generalizes better than independent
percentile thresholds. Both zero-API-call, pure re-analysis.
`eval/generalization_experiments.py`; full grid in
`results/generalization_exp1.csv`, `results/generalization_feature_dist.csv`,
`results/generalization_exp2.json`.

**Experiment 1 (pooled fit): the gap does NOT close -- confirms a real
mechanism problem, not just a single-day baseline artifact.** Fit on
Monday + half of Friday's own BENIGN flows (other half held out,
zero-day property asserted exactly as `dataplane/fitting.py` already
enforces), re-measured against the identical held-out populations used
for the Monday-only baseline. At k=2 the gap ratio is mixed-to-worse
with pooling: 13.4x->14.3x (p99.5), 88.7x->83.9x (p99.9, slightly
better), 907.2x->1991.7x (p99.99, much worse), 2455.6x->inf (p99.995).
Pooling does cut Friday's *absolute* FPR at several points (e.g. p99.99
k=2: 6.40%->4.92%; p99.999 k=2: 3.04%->1.75%) but never closes the gap
to Monday's near-zero level, and at the tightest, most decision-relevant
percentiles it's actively worse in ratio terms -- fitting on Friday's
*own* benign data still doesn't predict Friday's *other* held-out
benign half well.

**Root cause, directly diagnosed via feature distributions (Monday vs
Friday benign, full populations, medians nearly identical -- this is a
pure tail phenomenon, not a general shift):**

| feature | mon p99 | fri p99 | mon p99.5 | fri p99.5 | mon max | fri max |
|---|---|---|---|---|---|---|
| `distinct_dst_ports_per_src` | 18.7 | **202.6** (10.9x) | 31.9 | **238.1** (7.5x) | 205 | 1,019 (5x) |
| `flows_per_src` | 1,314 | 4,819 (3.7x) | 1,551 | 5,061 (3.3x) | 1,993 | **45,860** (23x) |
| `syn_without_synack_count` | 3 | 2 | 4 | 4 | 14 | **247** (17.6x) |
| `pkt_len_range` | 4,186 | 8,760 (2.1x) | 4,380 | 10,220 (2.3x) | 23,360 | 24,820 |
| `bwd_pkt_len_mean` | 1,551 | 2,361 (1.5x) | 1,764 | 2,942 (1.7x) | 3,016 | 3,909 |

`distinct_dst_ports_per_src` and `flows_per_src` shift most -- Friday
has legitimate benign hosts touching 5-11x more distinct ports and
opening up to 23x more flows than anything Monday ever samples, with
`syn_without_synack_count`'s max 17.6x higher too. Medians match almost
exactly across both days for all five features -- this is specifically
a thin, legitimately-occurring heavy tail (bursty/broad-activity benign
hosts) that a single day's worth of data can't reliably characterize at
a percentile cut, not a baseline-selection artifact.

**Experiment 2 (signature table): a real, class-specific win for DDoS,
and a diagnosable total failure for PortScan.** Five load-bearing
features binned into 6 quantile ranges (edges fixed from the pooled fit
population so both K variants differ only in which signatures they've
seen); escalate if a flow's 5-tuple bin-signature was never seen (below
a frequency floor) in training. At floor=1, `K_pooled` on
`combined_holdout`: **12.85% escalation, DDoS recall 98.5%** -- beats
every single point on the percentile x k grid at comparable-or-lower
volume (the closest percentile-grid escalation, ~16%, only gets DDoS to
0.1-16%). Confirms the hypothesis: combinations catch DDoS that marginal
per-feature thresholds miss. **But PortScan recall is 0.0% in every
floor/variant combination tested (1, 5, 20 x monday_only, pooled) --
strictly worse than percentiles, which get PortScan to 95-99%
everywhere.** Diagnosis: PortScan's extreme `flows_per_src`/
`pkt_len_range` values collapse into the same coarse top bin ("above
the 5th quantile cut") that Friday's own heavy-tailed *benign* hosts
already occupy in K -- per experiment 1's own finding, Friday benign
legitimately reaches `flows_per_src` up to 45,860. The identical
distribution-shift mechanism that breaks percentile thresholds also
breaks signatures for PortScan specifically, just via coarse binning
instead of a threshold miss. Unlike percentile pooling, growing K's
training population barely moves Friday-side numbers at all (floor=5:
`friday_eval` escalation 20.82%->20.77% monday_only->pooled, DDoS/PortScan
identical) -- K's bottleneck is bin coarseness at the tail, not sample
size.

**Net verdict: the single-day baseline was NOT the (whole) problem --
much of the redesign is still warranted.** But the signature-table
result is a genuine, reusable finding: it's a strong DDoS-specific
detector worth keeping/refining (e.g. finer top-end binning so PortScan
stops colliding with heavy-tailed benign hosts), not a replacement for
percentile thresholds outright.

## Ratio features: fixes the tail-shift problem, breaks PortScan

Per direct instruction, tested the hypothesis both experiments above
point at: the load-bearing features are absolute counts (`flows_per_src`,
`distinct_dst_ports_per_src`, `syn_without_synack_count`,
`distinct_dst_ips_per_src`), heavy-tailed and unbounded, so a percentile
fitted on one day's tail can't bound another day's. Added three
normalised replacements to `dataplane/selector.py`'s
`compute_src_features` -- `port_diversity_ratio =
distinct_dst_ports_per_src/flows_per_src`, `unanswered_syn_ratio =
syn_without_synack_count/flows_per_src`, `dst_concentration =
distinct_dst_ips_per_src/flows_per_src` -- each bounded in [0,1] by
construction (numerator is a sub-count of `flows_per_src`), registered as
`FeatureKind.BOUNDED_RATIO` in `dataplane/fitting.py`.
`FEATURE_SCHEMA_VERSION` bumped 4->5 (forces fresh PCAP extraction, ~10
min/day); `eval/sweep.py`'s `SRC_FEATURES` updated so per-flow-only
ablations still exclude all seven per-source columns. New script:
`eval/ratio_feature_experiments.py`. Zero-API-call, PCAP-path only.

**Part 1 confirms the hypothesis cleanly.** Friday/Monday ratio at
p99.5, counts vs their normalised replacements:

| feature | mon p99.5 | fri p99.5 | fri/mon | mon max | fri max |
|---|---|---|---|---|---|
| `flows_per_src` | 1,551 | 5,061 | **3.26x** | 1,993 | 45,860 |
| `distinct_dst_ports_per_src` | 31.9 | 238.1 | **7.46x** | 205 | 1,019 |
| `syn_without_synack_count` | 4 | 4 | 1.00x | 14 | **247** (17.6x) |
| `port_diversity_ratio` | 1.024 | 1.026 | **1.00x** | 1.103 | 1.107 |
| `unanswered_syn_ratio` | 0.035 | 0.043 | **1.24x** | 1.5 | 1.5 |
| `dst_concentration` | 1.004 | 1.004 | **1.00x** | 1.05 | 1.057 |

`port_diversity_ratio`/`dst_concentration` show essentially *zero*
cross-day shift at p99.5 (their raw-count counterparts shift 7.5x/3.3x),
and `unanswered_syn_ratio`'s max stays near 1.5 on both days vs.
`syn_without_synack_count`'s max exploding 17.6x on Friday. The medians
match almost exactly across both, same as the counts -- confirms this is
the same thin-tail phenomenon, just no longer able to blow past a hard
[0,1] bound the way an unbounded count can.

**Part 2 (percentile x k sweep, ratios substituted for
`flows_per_src`/`distinct_dst_ports_per_src`/`distinct_dst_ips_per_src`/
`syn_without_synack_count`): the FPR gap shrinks substantially, but
PortScan recall collapses.** At the same p99.5/k=2 operating point
(`results/percentile_k_sweep_ratios.csv` vs `results/percentile_k_sweep.csv`):

| | Monday FPR | Friday FPR | gap (pp) | gap (ratio) | PortScan recall | DDoS recall | Bot recall | pooled recall |
|---|---|---|---|---|---|---|---|---|
| counts (baseline) | 0.97% | 7.45% | 6.48pp | 7.66x | **99.72%** | 97.49% | 1.83% | 97.81% |
| ratios | 0.98% | 4.53% | 3.55pp | 4.62x | **1.04%** | 91.43% | 2.03% | 31.22% |

The FPR gap genuinely narrows (pp gap -45%, ratio 7.7x->4.6x, combined
escalation volume 28.4%->10.2%) -- normalising *does* help transfer, as
predicted. **But PortScan recall drops from 99.7% to 1.0%,** dragging
pooled attack recall from 97.8% down to 31.2%. Diagnosis: PortScan's
detectability was never "an elevated *rate*" -- a scan touches each port
once, so `port_diversity_ratio` for a scanning source looks the same
(~1.0) as any normal source that never reuses ports. The signal was
always the raw *magnitude* (thousands of flows/ports in one window), and
dividing by `flows_per_src` normalises exactly that away. This is a
sharper version of Experiment 2's PortScan finding above (there,
extreme-count collapse into a coarse top bin; here, the count's
information content is deliberately discarded) -- same root cause,
PortScan needs the absolute count, appearing a second, independent way.

**Net verdict: not a straight swap.** Ratios fix generalisation for
`port_diversity_ratio`/`dst_concentration` specifically and are worth
keeping, but not as a *replacement* for the raw counts -- PortScan needs
`flows_per_src`/`distinct_dst_ports_per_src` (or some other
magnitude-sensitive form of them) present alongside the ratios, not
instead of them. Untested next step: fit both forms together (ratios for
generalisation, counts for magnitude-sensitive classes like PortScan) and
see whether k_of_n's combination logic gets the best of both without
reopening the tail-shift gap.

## 5-feature selector: cross-day generalization test -- SUPERSEDED, see correction below

**This section's "confirmed overfit" conclusion does not hold as stated --
see "Cross-day follow-up: adapter mismatch confound" further down, which
found a same-day, same-threshold, two-order-of-magnitude CSV-vs-PCAP
crossing-rate gap on the exact per-source COUNT features this test
depends on. Left in place, not deleted, because the raw numbers below are
still accurate and the correction only holds if this section is visible
for it to correct.**

The reduced-feature line of work (`results/reduced_selector_report.md` ->
`results/final_selector_report.md` -> `results/seven_feature_selector_report.md`)
selected its final 5/7-feature set by looking only at which features fire
often on **Friday** (PortScan/DDoS/Bot). `eval/five_feature_cross_day.py`
tests that SAME, unchanged fitted config (Monday PCAP benign, chronological
split, p99.5; COUNT=`flows_per_src`/`distinct_dst_ports_per_src`/
`syn_without_synack_count`, RATIO=`syn_ratio`/`bwd_fwd_byte_ratio`, rule:
>=1 COUNT AND >=1 RATIO) against Tuesday/Wednesday/Thursday (CSV path,
cached `*__eval__adapter1__features4.parquet`, real CICIDS2017 labels
never examined during feature selection) and Friday (PCAP), no API calls,
no re-simulation. Full per-day tables in
`results/five_feature_cross_day_report.md`.

**Result: near-zero recall on every class outside PortScan/DDoS, at
sample sizes too large to be noise:**

| day | class | n | recall |
|---|---|---|---|
| Tue | FTP-Patator | 7,938 | 18.8% |
| Tue | SSH-Patator | 5,897 | 0.0% |
| Wed | DoS Hulk | 231,073 | 26.8% |
| Wed | DoS GoldenEye | 10,293 | 11.2% |
| Wed | DoS Slowhttptest | 5,499 | 11.0% |
| Wed | DoS slowloris | 5,796 | 0.0% |
| Wed | Heartbleed | 11 | 100.0% (n=11, unreliable) |
| Thu | Web Attack: Brute Force | 1,507 | 0.0% |
| Thu | Web Attack: XSS | 652 | 0.0% |
| Thu | Web Attack: SQL Injection | 21 | 0.0% (unreliable) |
| Thu | Infiltration | 36 | 0.0% (unreliable) |
| Fri | Bot | 2,901 | 0.3% |
| Fri | **PortScan** | 158,945 | **99.6%** |
| Fri | **DDoS** | 81,075 | **78.2%** |

Even `DoS Hulk` -- structurally the class closest to DDoS (a volumetric
flood) -- only reaches 26.8%, which argues against "the selector generalizes
to floods/scans broadly" and for "it's tuned to this scan's `syn_ratio`
signature and this flood's `bwd_fwd_byte_ratio` signature specifically."
Consistent with the prior finding (`results/seven_feature_selector_report.md`'s
ratio-contribution breakdown) that `syn_ratio`/`bwd_fwd_byte_ratio` never
co-fire and each carries exactly one Friday class end-to-end with zero
redundancy -- this cross-day test is the same narrowness showing up as
zero *transfer*, not just zero backup.

**Caveat, stated not used to soften the result:** Tuesday-Thursday are
CSV-adapter-derived (synthetic packet spacing) and the Monday-PCAP-fitted
thresholds are applied to them unchanged -- a cross-adapter gap layered on
the cross-day question. Doesn't explain the pattern away: `DoS Hulk`
failing alongside DDoS partially succeeding, both volumetric floods, is
hard to attribute to adapter mismatch alone.

**Verdict, in the terms the question was asked: yes, say so.** The
5-/7-/18-feature count-AND-ratio selector at p99.5 is a PortScan/DDoS
detector, not a general anomaly detector, and STATUS/README claims about
it should be scoped to those two classes rather than presented as a
general zero-day result until it's re-derived against the other attack
families (Patator, DoS-Hulk-style floods, web attacks, infiltration) --
none of which contributed to any feature or threshold choice in this
line of work.

## Cross-day follow-up: adapter mismatch confound -- the overfit conclusion above does not hold as stated

Two checks, requested specifically before trusting the section above:
(1) whether the unreduced 18-feature config also fails on Tuesday-Thursday
(isolating reduction from the count-AND-ratio approach itself), and (2)
whether Tue-Thu's near-zero recall is actually the CSV adapter failing to
produce the signal these features need, not a real generalization gap.
`eval/cross_day_followups.py`, zero API calls, re-analysis of cached
feature parquets. Full tables in `results/cross_day_followups_report.md`.

**Check 1 result: the 18-feature config does NOT cleanly succeed on
Tue-Thu either -- it fails in the opposite direction.** Recall looks
high (FTP-Patator 99.8%, DoS Slowhttptest 99.5%, DoS slowloris 98.4%,
Web XSS 93.6%, Infiltration 94.4%), but the BENIGN false-alarm rate on
those same days is **62.2% (Tue) / 64.9% (Wed) / 63.0% (Thu)** -- vs.
3.4% on Friday. That's not a working selector; it's escalating roughly
two-thirds of ALL traffic, benign and attack alike. High recall bought
by flagging almost everything isn't evidence of generalization. So
neither original branch ("18 fails too, reduction innocent" / "18
succeeds, revert reduction") is quite right: **both configs fail on
Tue-Thu, in opposite directions** (5-feature: recall-collapse with a
clean FPR; 18-feature: FPR-collapse with inflated-looking recall) --
reduction changed *which* failure mode shows up (a narrower RATIO-side
OR is incidentally more robust to whatever is driving the FPR blowup)
but didn't cause the underlying problem, which check 2 identifies.

**Check 2 result: confirmed, decisively -- this is adapter mismatch, not
generalization, and it's large.** Same Monday, same fitted thresholds
(from Monday PCAP), only the adapter differs:

| feature | Monday CSV benign crossing rate | Monday PCAP benign crossing rate |
|---|---|---|
| `flows_per_src` | 67.08% | 0.60% |
| `distinct_dst_ports_per_src` | 61.91% | 0.27% |
| `syn_without_synack_count` | 60.39% | 0.11% |
| `syn_ratio` | 3.85% | 0.56% |
| `bwd_fwd_byte_ratio` | 1.62% | 0.54% |

The three per-source COUNT features cross **100-600x more often on CSV
than on PCAP, on the identical day's benign traffic.** This directly
falsifies the specific mechanism hypothesized before running this check
(that `syn_ratio`/`syn_without_synack_count` might structurally never
fire on CSV data because the adapter reconstructs flags rather than
observing them) -- `syn_without_synack_count` doesn't fail to fire, it
fires 500x *too often*, the opposite direction. The real mechanism is
almost certainly the CSV adapter's per-source accounting itself:
`adapters/csv_flow_adapter.py`'s known lossy packet synthesis (evenly-
spaced synthetic timing, no true concurrency -- see README's "Known
Limitations") feeds a `SrcTable` whose per-source flow/port counts
accumulate completely differently than they would from real per-packet
arrival. The RATIO features are much closer between adapters (`syn_ratio`
3.85% vs 0.56%, ~7x, not ~100x) -- the distortion is concentrated in the
per-source COUNT features specifically, consistent with Tue/Wed's own
BENIGN crossing rates (`flows_per_src` 63-69%, `distinct_dst_ports_per_src`
64-69%) landing in the same wildly-inflated range as Monday CSV, while
`syn_ratio`/`bwd_fwd_byte_ratio` stay in the low single digits on every
CSV day, PCAP-comparable.

(Side observation, not chased further: Thursday's `flows_per_src`/
`distinct_dst_ports_per_src` crossing rates are anomalously low, 0.004%/
0.16%, unlike Tuesday/Wednesday's 60%+ -- almost certainly because
Thursday is cached as two SEPARATE CSV files, `thursday_morning_webattacks`
and `thursday_afternoon_infiltration`, each simulated with its own fresh
`SrcTable` per `eval/simulate.py`'s per-day-file caching, so per-source
counts never accumulate across a full working day the way Monday's/
Tuesday's/Wednesday's single-file simulations do. A cache-structure
artifact, not a new finding about Thursday's traffic.)

**Corrected verdict: the cross-day test as run cannot separate
"feature-selection overfit to PortScan/DDoS" from "CSV/PCAP adapter
mismatch," because the confound is real, large (two orders of magnitude
on the count features), and demonstrated on the SAME day with the SAME
thresholds -- not inferred. The "confirmed overfit" language in the
section above should be read as unconfirmed pending a clean test:
re-fit thresholds on Monday CSV (adapter1, matching Tue-Thu's own
adapter) rather than reusing Monday-PCAP-fitted thresholds, then re-run
the cross-day comparison. Not yet done. The one claim that still stands
untouched by this confound is the original PortScan/DDoS PCAP-only
result (`results/final_selector_report.md`, `results/
seven_feature_selector_report.md`) -- those never crossed the adapter
boundary.

## Clean cross-day test, adapter confound removed -- DEFINITIVE: narrowness is real, not the adapter, not the reduction

Both configs refit from scratch on Monday **CSV** (adapter1) benign,
chronologically split, p99.5 -- same adapter as Tuesday-Thursday, so the
confound identified in the section above can't be in play.
`eval/csv_refit_cross_day.py`, zero API calls, re-analysis of cached
feature parquets. Full per-day tables in
`results/csv_refit_cross_day_report.md`. (18-feature config runs at 17
features here -- the CSV cache is schema4, predating the 4->5 bump that
added `port_diversity_ratio`/`unanswered_syn_ratio`/`dst_concentration`;
stated in the report, not silently dropped.)

**Result: both configs now have sane, low FPR everywhere -- and both get
essentially zero recall on every non-Friday class.**

| day | class | n | 5-feature recall | 18-feature (17-feat) recall |
|---|---|---|---|---|
| Tue | FTP-Patator | 7,938 | 0.0% | 0.0% |
| Tue | SSH-Patator | 5,897 | 0.0% | 0.0% |
| Wed | DoS Hulk | 231,073 | 0.0% | 0.4% |
| Wed | DoS GoldenEye | 10,293 | 0.0% | 0.0% |
| Wed | DoS Slowhttptest | 5,499 | 0.0% | 0.0% |
| Wed | DoS slowloris | 5,796 | 0.0% | 0.0% |
| Wed | Heartbleed | 11 | 0.0% | 100.0% (n=11, unreliable) |
| Thu | Web Attack: Brute Force | 1,507 | 0.0% | 0.0% |
| Thu | Web Attack: XSS | 652 | 0.0% | 0.0% |
| Thu | Web Attack: SQL Injection | 21 | 0.0% | 0.0% |
| Thu | Infiltration | 36 | 0.0% | 0.0% |

Monday CSV holdout FPR: 5-feature 0.080%, 18-feature 0.262%. Tue/Wed/Thu
FPR: 0.05-0.57% for both configs across the board -- **the earlier
62-65% FPR "success" for the 18-feature config was confirmed to be
entirely the adapter-mismatch artifact identified above, not a real
property of count-AND-ratio at p99.5.** With that confound removed, the
18-feature config doesn't secretly work either -- it drops from
"escalates two-thirds of everything" straight to "detects almost
nothing," with no usable middle ground exposed at this operating point.

**Answers both original questions cleanly:**
- *Does 18-feature still escalate ~60% of benign traffic once properly
  fit?* No -- that was 100% the adapter confound. Properly fit, its FPR
  is sane (0.26-0.57%).
- *Does 5-feature get near-zero recall because it's too narrow?* Yes --
  but so does the (nearly-)full 18-feature config, at the same operating
  point, once fairly fit. **The narrowness isn't a reduction artifact --
  count-AND-ratio at p99.5 genuinely does not carry signal for Patator,
  DoS-Hulk-style floods, web attacks, or infiltration, at 5 features or
  at 17.** Feature count was never the variable that mattered here.

**This supersedes both sections above with a confound-free answer.** The
selector (5-feature or 18-feature, count-AND-ratio, p99.5, Monday-fitted)
is confirmed to be PortScan/DDoS-specific -- not because of reduction,
not because of an adapter artifact, but as a genuine property of the
approach against these other attack families. Any claim about this
selector's zero-day/general-anomaly properties should be scoped to
PortScan/DDoS explicitly. Untested next step, if this needs to be fixed
rather than just scoped: different features/rule/operating point derived
with Tue-Thu's own attack behavior in view, not just Friday's.

## Two checks on the "narrowness is real" verdict -- refined, not reversed

Before recording the section above as final: (1) is DoS Hulk's 0.0%
(n=231,073) a feature firing-and-missing or a feature that structurally
can't fire, and are CSV-fitted thresholds inflated relative to PCAP; (2)
does recall turn on at any looser percentile (p98/p99/p99.5), or does it
stay at zero regardless. `eval/csv_refit_diagnostics.py`, zero API calls.
Full detail in `results/csv_refit_diagnostics_report.md`.

**Check 1: CSV-fitted thresholds ARE dramatically inflated vs PCAP --
22-184x higher** (`flows_per_src` 33,523 vs 1,505; `distinct_dst_ports_per_src`
1,760.76 vs 38.80; `syn_without_synack_count` 737 vs 4; `bwd_fwd_byte_ratio`
105.36 vs 33.06). For `syn_without_synack_count` specifically this is the
direct cause of Hulk's miss: Hulk's own value (348) clears the PCAP-scale
threshold (4) easily but sits below the CSV-inflated one (737). But
`flows_per_src`/`distinct_dst_ports_per_src` would miss Hulk even at
PCAP-scale thresholds -- Hulk's own values (58 flows/src, 12.29 ports/src)
are genuinely far below even the PCAP threshold (1,505 / 38.8), so part
of the miss is real narrowness, not just inflation.

**Check 2: recall stays exactly 0.0% at every percentile tested (p98,
p99, p99.5) -- confirmed, and the mechanism is now fully understood.**
Drilling into which side of the AND-gate still blocks Hulk at p98 (the
loosest point swept): the RATIO side loosens correctly (`bwd_fwd_byte_ratio`'s
threshold drops to 31.57 at p98, just below Hulk's own median of 32.03 --
most Hulk flows WOULD satisfy the ratio side there). **The COUNT side is
what's frozen: `flows_per_src`'s threshold is flat at exactly 33,523
across the entire p98-99.5 range.** Root cause, checked directly: Monday
CSV fit-half's `flows_per_src` (264,959 rows, only 191 distinct values)
has **37,692 flows (14.2%) tied at exactly 33,523, its own maximum** --
so every percentile from p90 through p100 lands on that tied maximum and
`>` can never cross it, the same structural failure as the bounded-ratio
saturation bug fixed earlier in this project (`syn_ratio`/
`no_response_flag`), now found in an *unbounded* count feature, on the
CSV adapter specifically -- almost certainly `adapters/csv_flow_adapter.py`'s
already-documented lack of true concurrency (README's "CSV-adapter-driven
runs don't reproduce realistic flow concurrency") letting a small number
of "busy" simulated sources dominate the per-source count distribution.
Reaching below that 14.2% mass would need a percentile below ~p86 --
outside anything tested or proposed.

**Refined verdict: the "narrowness is real" conclusion survives in
substance but was incomplete.** DoS Hulk's miss is a genuine mix of (a)
real narrowness -- `flows_per_src`/`distinct_dst_ports_per_src` are the
wrong shape of feature for Hulk's behavior (low flow-count, high
byte-volume) regardless of threshold placement -- and (b) a real,
previously-unknown fitting bug (`flows_per_src` percentile-saturation on
the CSV adapter) that never got a chance to loosen within the tested
range and should not be read as "no operating point could ever work."
Whether `syn_without_synack_count` + `bwd_fwd_byte_ratio` alone (both
free of the flat-threshold problem, both already loosening correctly by
p98) could carry Hulk detection once `flows_per_src`'s saturation bug is
fixed is untested -- flagged as the concrete next step, not assumed
either way. Patator/web-attacks/infiltration weren't re-examined by this
check (only Hulk was drilled into) -- their own 0.0% numbers should be
treated with the same "not yet ruled out" caution until similarly
checked, not silently carried over as confirmed.

## Generic saturation fix, then a valid cross-day re-run -- FINAL, supersedes both sections above

The "not yet ruled out" caution above was warranted: `flows_per_src` --
the single highest-firing feature -- was structurally frozen on the CSV
path (14.2% of Monday CSV's fit-half tied at its own maximum, saturated
at every percentile from p90 through p100), and the whole cross-day test
ran with it silently disabled. Fixed generically, not case-by-case, and
re-run properly. Four-part task, `dataplane/fitting.py` +
`dataplane/selector.py` + `eval/sweep.py` changed; 4 new regression tests
(475 passing, up from 471). No API calls throughout.

**Part 1 -- generic saturation detection (`dataplane/fitting.py`).**
Every feature (not just the hand-curated BOUNDED_RATIO/BOOLEAN list) now
goes through one saturation-aware fit: a threshold is unreachable if it
lands exactly on the feature's own observed extreme, OR the mass tied AT
that extreme exceeds the percentile's own false-positive budget (with a
`n_distinct > 1` guard so a genuinely constant feature, or the single
unique top value of a small sample, doesn't false-positive). On
saturation: BOOLEAN features are always excluded (closed 2-value domain,
a rarity fit can never be informative there); everything else gets a
rarity fit (escalate on values covering under the percentile's own
budget of training mass -- including any value never seen in training at
all) when cardinality is low enough (`RARITY_MAX_DISTINCT_ABS`/`_RATIO`,
calibrated against this project's own cases), otherwise excluded. Every
fit now emits a full per-feature report (`action`, threshold value(s),
`feature_max`/`min`, `tied_mass`, `n_distinct`, `rarity_coverage`) --
this should surface automatically from now on, not require a
investigation each time it's hit. `FeatureThreshold` gained
`common_values` (a rarity fit, mutually exclusive with `high`/`low`);
`Selector.evaluate_features` and `eval/sweep.py::compute_crossings` both
handle it. This is a global fitting-code change -- it also applies to
every PCAP fit in this project's prior reports
(`results/final_selector_report.md`,
`results/seven_feature_selector_report.md`, etc.), which were generated
before this fix and are not guaranteed bit-identical if re-run today;
not re-run as part of this task, flagged for whoever revisits them.

**Part 2 -- clean cross-day re-run, saturation fixed, percentile swept
(p98/p99/p99.5), both configs fit on Monday CSV.** Full tables in
`results/saturation_fixed_cross_day_report.md`. Headline change from the
invalid test: real, non-saturation-artifact detection now appears for
multiple non-PortScan/DDoS classes AT p98, cliffing off by p99 (same
"cliff not dial" percentile pattern found throughout this project):

| class (n) | p98 5-feat / 18-feat | p99 5-feat / 18-feat | p99.5 5-feat / 18-feat |
|---|---|---|---|
| FTP-Patator (7,938) | 30.9% / **80.8%** | 30.9% / 30.9% | 0.5% / 0.5% |
| SSH-Patator (5,897) | 0.2% / 35.2% | 0.2% / 0.2% | 0.2% / 0.2% |
| DoS Hulk (231,073) | **40.1% / 41.8%** | 0.0% / 0.7% | 0.0% / 0.7% |
| DoS GoldenEye (10,293) | 14.7% / 14.7% | 0.0% / 0.0% | 0.0% / 0.0% |
| DoS Slowhttptest (5,499) | 5.4% / 11.9% | 5.4% / 9.1% | 0.1% / 3.8% |
| DoS slowloris (5,796) | 22.7% / 23.9% | 22.7% / 22.7% | 17.9% / 17.9% |
| Web Attacks (all 3) | 0.0% / 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |

Monday CSV holdout FPR at p98: 2.29% (5-feat) / 5.15% (18-feat) -- Tue/Wed/Thu
FPRs land in the same 4-7% range, a real, usable operating point, not a
saturated one. **This directly overturns the "narrowness is real"
conclusion for Patator and the DoS-flood variants** -- they were never
undetectable, they were measured with a broken feature.

**Part 3 -- does `syn_without_synack_count` + `bwd_fwd_byte_ratio` alone
carry Hulk? No -- and this corrects Part 2's own initial read, not just
the pre-fix baseline.** Isolated 2-feature test:
**0.0% Hulk recall at every percentile (p98/p99/p99.5)** --
`syn_without_synack_count`'s Monday-CSV-fitted threshold (557) sits
above Hulk's own uniform value (348), unlike the PCAP-scale comparison
that motivated testing this pair (PCAP threshold was only 4 -- CSV/PCAP
threshold inflation, already documented, applies here too). **The real
mechanism carrying Hulk's 40% recall in the 5-/18-feature configs is
`flows_per_src`'s newly-fixed rarity fit** -- crossing rate 90.6% on ALL
Wednesday flows (Hulk's uniform value of 58 isn't in Monday's common
set), paired with `bwd_fwd_byte_ratio`. The fix didn't turn
`flows_per_src` into a clean magnitude-based flood detector; it turned
it into a broad rarity/novelty detector that happens to catch Hulk
(and fires on 90% of Wednesday in general) while the ratio side does the
real discriminating. Full detail: `results/two_feature_hulk_report.md`.

**Part 4 -- failure attribution, three distinct findings, not collapsed
into one.** Full detail: `results/failure_attribution_report.md`.

- **Wrong shape (genuine limit): Web Attack Brute Force/XSS/SQL
  Injection, 0.0% at every config and every percentile.** Exact values:
  `flows_per_src`=5 (uniform, actually NOT in the rarity common set --
  the COUNT side fires), `distinct_dst_ports_per_src`=1.002 (uniform,
  threshold 820.76), `syn_without_synack_count`=0 (uniform, threshold
  557 -- a clean handshake, not a scan), `syn_ratio`=0 (IS in the common
  set -- ordinary), `bwd_fwd_byte_ratio` undefined for 90-96% of flows
  and far under threshold (31.57) where defined. The RATIO side is what
  blocks it, for a structural reason: these are application-layer
  attacks (malicious HTTP payload) riding on completely ordinary-looking
  single-request flows -- no flow-statistics feature observes payload
  content, so no percentile or combination within this family fixes it.
- **Below threshold (tuning, not a hard limit): DoS Hulk/GoldenEye/
  Slowhttptest/slowloris.** Real signal at p98, suppressed by p99+ as
  `bwd_fwd_byte_ratio`'s threshold rises past the attacks' own values
  (31.57->63.68->105.36 vs. Hulk's median 32.03/p99 39.57) -- already
  diagnosed in the prior section's follow-up.
- **Inconclusive (n too small): Infiltration (36), Heartbleed (11).**
  Not attributed either way.
- **Config-dependent, not shape/threshold: SSH-Patator** (0.2% at
  5-feature, 35.2% at 18-feature -- needs the fuller feature set).
- **Out of scope: Bot** (no CSV-path data; PCAP-path finding, a genuine
  per-flow-too-short limit, already independently established and
  unaffected by this fix).

**Realistic-outcome check against the task's own prediction: partially
right, with a real correction.** PortScan/DDoS (PCAP, unaffected) and now
Patator/Hulk/GoldenEye (CSV, p98) do get detected, as predicted -- but
NOT via "flood-shaped features" as hypothesized; Hulk specifically is
carried by the fixed rarity mechanism, not magnitude. Web attacks and
Bot fail for genuine, now precisely diagnosed reasons, as predicted.
Slow DoS (Slowhttptest/slowloris) is NOT a clean failure -- it's
partially detected and threshold-sensitive, more nuanced than "failing
by design." No number was tuned toward a target; every table above is
what the measurements produced.

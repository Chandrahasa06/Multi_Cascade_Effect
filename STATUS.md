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
- `agents/` — the five-agent zero-day review pipeline (A1 evidence → A2 behaviour → A3 hypotheses, A4 blind replica, A5 verdict), plus `controlplane/reference.py` (Monday benign reference distribution) and `eval/run_agent_pipeline.py` (cost-controlled runner). See "Five-agent zero-day review pipeline" below — stop points 1-2 done, stop point 3 (the full 200-record run) not started.
- `tests/` — 435 tests, all passing, across all of the above.

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

Progress as of last run: **139/200 unique records** (DDoS 50/50,
BENIGN 69/70 -- one record failing A3 schema validation on retry,
PortScan 20/50, Bot 0/30), run in DDoS -> BENIGN -> PortScan -> Bot
order per instruction. Daily quota (`gemini-3.5-flash-lite`, 400/day)
is the binding constraint; `eval/run_agent_pipeline.py` and
`eval/run_baseline_pipeline.py` (the still-outstanding single-LLM
baseline, `agents/baseline.py` / `BaselineResponse` / `baseline_v2`
prompt -- now includes the same empirical grounding block A1/A3/A4/A5
get, asks directly for a continuous `benign_plausibility` instead of a
categorical verdict, so it's head-to-head comparable to A5) both resume
for free via cache -- rerun the same command after each UTC quota
reset until both are done, then re-run `eval/ablation_v200.py`
(the n=200 successor to `ablation_v4.py`: adds the baseline comparison,
a full operating-point sweep instead of just 0%-FPR, and per-class
trust/chain_vs_independent breakdowns).

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

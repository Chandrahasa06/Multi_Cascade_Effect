# Multi-Cascade Effect — Data Plane + Zero-Day-Safe Selector

An SDN intrusion-detection pipeline for zero-day attacks, built in two
stages (this repo covers both, not the downstream agent pipeline):

1. **Data plane** — O(1), integer-only, fixed-capacity per-flow and
   per-source counters, checked against benign-fitted thresholds by a
   selector that decides which flows to escalate.
2. **Control plane** — escalated flows get full CICFlowMeter feature
   extraction, anonymised, and packaged as a serialisable handoff record.

Zero-day safety is enforced structurally: threshold fitting only ever
sees benign traffic (`fitting.py` asserts this — see below), so the
selector can never become a signature matcher for known attacks.

## Status

Built and tested: `dataplane/` (flow table, per-flow and per-source
counters, HyperLogLog sketch, selector), the CSV adapter, and
`fitting.py`. Not yet built: `controlplane/` (CICFlowMeter extraction —
blocked on real PCAPs), `eval/` (recall/escalation-rate sweep).

## Running

```
pip install -r requirements.txt
python -m pytest tests/ -q
python run_csv_stress_test.py   # stress-test the flow table on real CSV data
```

`dataplane/fitting.py` is built and tested (`fit_thresholds`, `save_thresholds`/
`load_thresholds`) but not yet wired to a CLI entry point — that lands with
the sweep script.

## Module layout

```
dataplane/
  flow_state.py      per-flow counter struct (Packet, FlowState), O(1) update
  flow_table.py       fixed-capacity table, keyed, eviction, KeyMode (fidelity/eval)
  src_table.py        per-source sliding-window counters + HyperLogLog sketches
  hyperloglog.py       from-scratch HLL cardinality sketch
  selector.py          Tier-1/per-source feature computation, threshold comparison
  fitting.py            benign-only threshold fitting
adapters/
  csv_flow_adapter.py  CICIDS2017 TrafficLabelling CSV -> Packet stream
tests/
```

## Known limitations

**Ground truth is defined at CICFlowMeter's flow granularity, and the
data plane doesn't necessarily share that definition — this is an
irreducible join ambiguity, not a bug to fix.** CICFlowMeter decides
where one flow record ends and the next begins (its own idle/active
timeouts, its own handling of retransmits and reordering). Our flow
table has its own, independently-implemented notion of the same thing.
The two will not always agree on where a real conversation's boundaries
are, so "what is this simulated flow's ground-truth label" doesn't
always have a single correct answer — a simulated flow can legitimately
span, or be spanned by, more than one CICFlowMeter-labeled row.

We do not solve this; we sidestep it. `adapters/csv_flow_adapter.py`
exposes `FlowTable.KeyMode`:

- `KeyMode.FIDELITY` (plain 5-tuple) lets the table's own merge/timeout/
  eviction behavior run and be measured honestly — this is the right
  mode for reporting on the data plane's own flow definition. On real
  data this mode *does* merge rows that CICFlowMeter had kept separate:
  checked on CICIDS2017 Wednesday (a DoS day), 5.4% of the resulting
  merge-groups — 32.7% of all rows — combined more than one ground-truth
  label under one 5-tuple.
- `KeyMode.EVAL` (5-tuple + source CSV row index) forces one row to map
  to exactly one simulated flow, guaranteeing an unambiguous label per
  flow. This is the default for anything that reports label-derived
  numbers (escalation rate, recall, the fitting/sweep pipeline). It
  removes the ambiguity by construction, not by resolving it — a real
  deployment reading real packets has no "row index" to key on, so this
  mode is specific to replaying pre-aggregated CSV ground truth and
  doesn't generalize to the PCAP path.

Every synthesized `Packet` carries `source_row_id` regardless of mode,
and `FlowState.source_row_ids` accumulates all of them, so even a
fidelity-mode flow that merged several rows can be traced back to
exactly which ones.

Keying alone isn't quite enough for an exact 1:1 guarantee in EVAL mode:
some rows have a long real duration but very few packets, and this
adapter's even-spaced packet synthesis can reproduce that as a synthetic
inter-packet gap longer than the idle timeout even though it's one
continuous CICFlowMeter-recorded flow — idle timeout would then split it
regardless of the key. `KeyMode.EVAL` therefore also disables idle/active
timeout enforcement outright (capacity eviction still applies). Confirmed
on the real Monday file: 529,918 rows -> exactly 529,918 flows, 0 idle
timeouts, 0 active timeouts. The cost: with timeout-based cleanup gone,
capacity eviction becomes the only way a flow leaves the table short of
a FIN/RST, and it now fires far more often — 464,310 times at the
default 65,536 capacity on that same run. EVAL mode's eviction count is
not a realistic resource-pressure estimate either; it's the price of the
label-accuracy guarantee, not a data-plane finding.

**The CSV adapter's packet synthesis is necessarily lossy.** CICIDS2017's
TrafficLabelling CSVs are CICFlowMeter's finished, aggregated flow
records, not packet captures — there is no way to recover the true
per-packet sequence from packet-count/byte-sum/min/max/flag-count
summaries. See the module docstring in `adapters/csv_flow_adapter.py`
for the specific approximations (packet-length reconstruction, even
timestamp spacing instead of real IAT distribution, flag placement by
count only) and what each one costs. The practical consequence: the
selector's every-8-packets check cadence cannot be honestly validated
against CSV-derived data — that needs a real PCAP.

**CICIDS2017's own Timestamp column resolution varies by day.** Monday
has second resolution; Tuesday–Friday only have minute resolution (no
seconds field at all). `Packet.timestamp_resolution_us` /
`FlowState.timestamp_resolution_us` carry this through, and
`selector.py` flags IAT/duration/rate features `low_confidence=True`
when a flow's coarsest contributing packet is coarser than 1 second —
this rides through to `TriggerReason.low_confidence` so the agent
pipeline sees it too, and must be respected by `fitting.py` (excluded
or weighted, not silently trusted as exact).

**Distinct-port/distinct-IP counts are HyperLogLog estimates, not exact
counts.** Default precision (256 registers) gives ~6.5% standard error.
Fine for telling a scanner touching 30 ports from a benign host touching
3; not exact, and reported as such via `SrcTable.sketch_standard_error`.

**The flow table's and source table's LRU eviction is a simulation
convenience, not hardware-accurate.** Both use touch-recency order to
decide what to evict under capacity pressure. Real P4 register arrays
have no such concept — they evict on hash collision, which would change
*which* entry gets evicted (and plausibly the eviction rate) versus what
this simulator reports. See the hardware-fidelity notes in
`flow_table.py` and `src_table.py`.

**The source table's capacity is independent of the flow table's, on
purpose.** A naive "one source table slot per flow table slot" sizing
(65,536) would cost roughly 203MB of register state at the default HLL
precision — more than a P4 pipeline stage's on-chip SRAM budget in
practice. The default of 8,192 concurrently-tracked sources (~25MB)
reflects that sources are hosts, not flows, and need a much smaller
independent budget.

**CSV-adapter-driven runs don't reproduce realistic flow concurrency.**
Rows are fed to the table in file order, each row's synthesized packets
run to completion before the next row starts. This preserves realistic
*ordering* (CICIDS2017 rows are close to chronological) but not
realistic *concurrency* — flows that really overlapped in time aren't
interleaved in the packet stream, which understates how many flows are
open at once and therefore understates capacity-eviction pressure at
realistic table sizes. Eviction *mechanism* correctness is still
demonstrated separately with an artificially small capacity; the
eviction *rate* measured this way should not be quoted as realistic.

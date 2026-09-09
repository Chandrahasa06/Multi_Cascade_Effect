# Task: Data-plane flow simulator + zero-day-safe selector + feature extraction (CICIDS2017)

## Context

I'm building an SDN-based intrusion detection pipeline for **zero-day attack detection**. The full system has three stages:

1. **Data plane** — cheap per-flow counters on every flow at line rate. A *selector* applies thresholds; flows that cross get escalated.
2. **Control plane** — escalated flows get full CICFlowMeter feature extraction (78 features), anonymised and packaged for the agents.
3. **Agent pipeline** — a 5-agent LLM chain analyses the extracted features and decides whether the flow is a novel attack.

**You are building stages 1 and 2 only.** Do not build the agent pipeline. Do not call any LLM. If you find yourself writing prompts or API calls, stop — that's out of scope. The deliverable ends at a clean, serialisable record ready to hand to the agents.

## The core design constraint

The selector must be **zero-day safe**. This is the single most important property of this code.

Thresholds are fitted on **benign traffic only**. The fitting code must never see an attack label. If the selector is tuned against known attacks, it becomes a signature matcher and the whole zero-day claim collapses. Enforce this in code: the threshold-fitting function accepts only benign flows and `assert`s that no row carries an attack label.

Attack labels are used for **evaluation only**, never for fitting.

## Hard constraints on the data plane

Everything in the flow table must be implementable on a real P4 switch:

- **Integer arithmetic only.** No floats in the stored state.
- **No variance, no standard deviation, no sqrt, no percentiles at runtime.** If you want std-dev, that feature belongs in the control plane.
- **O(1) update per packet.** No loops over packet history.
- **Fixed-capacity flow table** (default 65,536 entries) with eviction. Real switches have finite register arrays. Log the eviction rate — it's a real limitation worth reporting.
- Derived quantities (rates, means, ratios) are computed **at check time** from stored counters, never stored.

Mean packet length is fine (`byte_sum / pkt_count`). Variance is not.

## Module structure

```
dataplane/
  flow_table.py      # fixed-capacity table, 5-tuple keyed, eviction
  flow_state.py      # per-flow counter struct + O(1) update
  src_table.py       # per-source-IP sliding-window counters + sketches
  selector.py        # threshold rules, escalation decision, trigger reasons
  fitting.py         # benign-only threshold fitting
controlplane/
  extractor.py       # packet window -> CICFlowMeter -> 78 features
  anonymise.py       # strip identity leaks
  record.py          # EscalationRecord dataclass (interface to stage 3)
eval/
  labels.py          # join flows to CICIDS2017 ground truth
  sweep.py           # recall vs escalation-rate curve
  report.py          # per-class results table
tests/
```

## Per-flow state

Key on the canonicalised 5-tuple (src IP, dst IP, src port, dst port, protocol) so A→B and B→A map to one flow, with a direction bit per packet. Store:

- `fwd_pkt_count`, `bwd_pkt_count`
- `fwd_byte_sum`, `bwd_byte_sum`
- `fwd_pkt_len_min`, `fwd_pkt_len_max`, `bwd_pkt_len_min`, `bwd_pkt_len_max`
- `first_ts`, `last_ts`, `prev_pkt_ts`
- `iat_sum`, `iat_min`, `iat_max` (microseconds, integer)
- `syn_count`, `fin_count`, `rst_count`, `psh_count`, `ack_count`, `urg_count`
- `init_win_bytes_fwd`, `init_win_bytes_bwd`

## Tier 1 — selector features (computed at check time)

| Feature | Derivation |
|---|---|
| `flow_duration` | `last_ts − first_ts` |
| `flow_bytes_per_sec` | `(fwd_bytes + bwd_bytes) / duration` |
| `flow_pkts_per_sec` | `(fwd_pkts + bwd_pkts) / duration` |
| `fwd_pkt_len_mean` | `fwd_byte_sum / fwd_pkt_count` |
| `bwd_pkt_len_mean` | `bwd_byte_sum / bwd_pkt_count` |
| `pkt_len_range` | `pkt_len_max − pkt_len_min` |
| `down_up_pkt_ratio` | `bwd_pkt_count / fwd_pkt_count` |
| `bwd_fwd_byte_ratio` | `bwd_byte_sum / fwd_byte_sum` |
| `flow_iat_mean` | `iat_sum / (total_pkts − 1)` |
| `flow_iat_max`, `flow_iat_min` | stored directly |
| `syn_ratio` | `syn_count / total_pkts` |
| `rst_ratio` | `rst_count / total_pkts` |
| `no_response_flag` | `bwd_pkt_count == 0` |
| `init_win_bytes_fwd`, `init_win_bytes_bwd` | stored directly |

## Per-source register array (`src_table.py`)

**Build this from the start, not as a later addition.** A single brute-force or bot flow looks normal in isolation; the anomaly lives in the pattern across flows from one host. Without these features the selector does well on DoS and PortScan and badly on Patator, Bot, and Infiltration.

Keyed on source IP, sliding window (default 60s):

- `flows_per_src`
- `distinct_dst_ports_per_src` — approximate with a Bloom filter or HyperLogLog sketch (both are P4-implementable; exact sets are not)
- `distinct_dst_ips_per_src`
- `syn_without_synack_count`

These feed the selector alongside the per-flow features and are fitted the same benign-only way.

## Flow lifecycle

- **Terminate** on FIN handshake or RST.
- **Active timeout**: 120s (configurable, matches CICFlowMeter default).
- **Idle timeout**: 15s (configurable).
- **Selector runs every N packets** (default N=8) *and* at flow close. Do not defer the decision to flow end — a long-lived attack flow should escalate early.

## Threshold fitting (`fitting.py`)

Input: benign flows only. CICIDS2017 Monday is benign-only and is the natural fitting set.

For each selector feature, compute the p-th percentile of the benign distribution (default p=99.5). Some features are anomalous when *low* as well as high (e.g. mean packet length for scan traffic), so support two-sided thresholds — take both the p-th and (100−p)-th percentile where it makes sense.

Escalation rule, configurable:
- `any` — escalate if any single feature crosses (high recall, high escalation rate)
- `k_of_n` — escalate if at least k features cross (fewer false escalations)

Persist fitted thresholds to JSON so runs are reproducible.

## Trigger reasons

When the selector fires, record **which** features crossed and **by how much**, as a multiple of the benign threshold. For example: `flow_pkts_per_sec = 8400 (12.3× benign p99.5)`.

This is not just for debugging. It gets passed forward to the agent pipeline, whose first stage turns raw measurement into interpretable evidence — "here is what looked wrong at line rate" is the strongest starting evidence it can receive. Make it a structured list, not a formatted string.

## Tier 2 — control plane feature extraction

When the selector fires, pull that flow's packet window from the PCAP and run it through CICFlowMeter to produce the full feature record.

Use the `cicflowmeter` Python package if it works cleanly; fall back to shelling out to the original Java CICFlowMeter if the Python port produces feature drift. Note which you used.

**Send all 78 features. Do not prune the feature set.** The usual way people derive a "minimum feature set" on CICIDS2017 is feature importance from a supervised model trained on the labels — which is the contamination problem in another form. It selects the features that best separate *known* attacks, exactly the wrong prior for a novel one. Pruning throws away the features a zero-day would show up in. Token cost is trivial next to five agent calls.

## Anonymisation (`anonymise.py`) — mandatory

Two fields must be stripped before anything reaches the agents:

- **Source and destination IP.** In CICIDS2017 the attacker is a fixed host and the victim range is a known subnet. An agent that sees IPs can learn "this address is the attacker" and the detection rate becomes meaningless. Replace with flow-local pseudonymous IDs.
- **Wall-clock timestamp.** The attack windows are at documented times of day. Keep relative duration; drop absolute time.

Destination **port** stays — it's a genuine observable, not an identity leak.

Write this as an explicit allow-list of fields, not a deny-list, so new fields can't leak in silently. Unit-test that no output record contains an IP-shaped string or an absolute timestamp.

## The handoff record (`record.py`)

The interface to stage 3. A serialisable dataclass:

```
EscalationRecord:
    flow_id: str              # pseudonymous
    trigger_reasons: list     # feature, observed value, threshold, ratio
    features: dict            # 78 CICFlowMeter features, anonymised
    packet_window: tuple      # index range, for reproducibility
    selector_config_hash: str # which thresholds produced this
```

Ground truth label is stored **separately**, in the evaluation harness, never inside this record.

## Data

CICIDS2017: five days of PCAPs (Monday–Friday) plus the labelled `MachineLearningCSV` flow records.

Joining flows to ground-truth labels is fiddly — match on 5-tuple plus timestamp proximity, and expect imperfect coverage. Report the join rate; if it's below ~95%, tell me before proceeding rather than silently dropping flows.

Two known issues: the original CSVs have documented labelling errors and CICFlowMeter bugs (Engelen et al., 2021 — a corrected version of the dataset exists). And several attack classes are tiny: Heartbleed ~11 flows, SQL Injection ~21, Infiltration ~36. Report them but flag the sample size rather than drawing conclusions.

## Evaluation

The headline result is a **recall vs escalation-rate curve**. Sweep the fitting percentile from 90.0 to 99.99 and at each point measure:

- **Escalation rate** — % of all flows sent to the control plane. This is the cost axis; every escalated flow costs a CICFlowMeter run plus, later, a five-agent chain.
- **Per-attack-class recall** — of flows labelled attack class C, what fraction escalated. This is the ceiling on the whole system's detection rate for that class; anything the selector drops is invisible forever.
- **Overall attack recall**, benign escalation rate, flow-table eviction rate.

Produce a table of per-class recall at escalation rates of 1%, 5%, and 10%, plus the full curve as a plot.

Report separately how much the per-source features contribute: run the sweep with and without them. I expect the difference to be large for Patator, Bot, and Infiltration, and I want that quantified.

Also report simulator throughput (packets/sec, flows/sec).

## Deliverables

1. The modules above, with type hints and docstrings.
2. A CLI: fit thresholds from benign traffic, run the simulator over a PCAP, extract features for escalated flows, produce the evaluation report.
3. Unit tests for flow keying (bidirectional canonicalisation), counter updates, timeout handling, eviction, percentile fitting, and the anonymisation allow-list.
4. A `results/` directory with the sweep table and curve.
5. A short README covering how to reproduce.

## Before you start

Ask me about anything genuinely ambiguous rather than guessing — in particular the PCAP file paths and whether the label CSVs are the original or corrected version. Don't invent a data layout.

Build incrementally. Get the flow table and counters correct and tested first, and show me them working on a small PCAP slice before building anything downstream. Then per-source counters, then fitting, then extraction, then the sweep.

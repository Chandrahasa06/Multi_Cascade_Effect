# How the neighbourhood count is computed, as implemented

Written for: a supervisor who has not read the code. Read-only: no code, results or experiments were changed, and no model calls were made. The numbers below were recomputed from the saved outputs and the cached benign pool. Quotes are from the files at the line numbers shown.

Short version: the pipeline uses **two different benign populations** in the same sentence it gives A5, and **the flow's own benign copy is counted** when the flow is benign. Both are detailed below.

## 1. Which benign flows are the reference set

There are two paths. The reports cite both numbers.

### 1a. The 566,864 population (the Tier-1 profile count and `grounding.benign_df`)

- **Days:** Monday only. The file is named `monday_pcap__...` and its docstring says it is "Monday's benign-only PCAP simulation".
- **Filter:** the loader asserts every row is BENIGN; it does not select rows.
- **Chronology:** no split. It is the whole Monday file.
- **Where assembled:** a cached parquet, not built from the CSVs at run time.

`controlplane/reference.py` lines 43-43

```python
DEFAULT_REFERENCE_PARQUET = Path("results/cache/monday_pcap__fidelity__adapter2__features4.parquet")
```

(line numbers: 43)

`controlplane/reference.py` lines 113-122

```python
def load_benign_dataframe(parquet_path: Path = DEFAULT_REFERENCE_PARQUET) -> pd.DataFrame:
    """Cached in-process (lru_cache, keyed on path) -- a 566,864-row
    parquet has no business being re-read from disk on every one of the
    ~100+ calls a single pipeline run makes into this module."""
    df = pd.read_parquet(parquet_path)
    assert (df["label"] == "BENIGN").all(), (
        "reference distribution must be built from benign-only data -- "
        "found a non-BENIGN label in the source parquet"
    )
    return df
```

(line numbers: 113, 114, 115, 116, 117, 118, 119, 120, 121, 122)

`agents/pipeline.py` lines 156-159

```python
def compute_grounding(record: EscalationRecord) -> Grounding:
    reference = get_reference_distribution()
    benign_df = load_benign_dataframe()
    nn_result = compute_nearest_neighbours(record, benign_df, reference)
```

(line numbers: 156, 157, 158, 159)


### 1b. The 2,273,097 population (the repaired neighbourhood path)

- **Days:** all eight capture days, Monday through Friday, including the Tuesday to Friday days that contain the attacks.
- **Filter:** `Label == BENIGN` only. Attack rows are dropped.
- **Chronology:** no split. Whole pool, all days pooled.
- **Where assembled:** `load_pool()` builds the pool from the eight CSVs (cached as parquet); the benign rows are selected in `load_benign_escalation_reference`.

`eval/escalation_data.py` lines 34-43

```python
DAY_FILES: List[Tuple[str, str, int]] = [
    ("monday", "Monday-WorkingHours.pcap_ISCX.csv", 0),
    ("tuesday", "Tuesday-WorkingHours.pcap_ISCX.csv", 1),
    ("wednesday", "Wednesday-workingHours.pcap_ISCX.csv", 2),
    ("thursday_webattacks", "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv", 3),
    ("thursday_infiltration", "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv", 3),
    ("friday_morning", "Friday-WorkingHours-Morning.pcap_ISCX.csv", 4),
    ("friday_portscan", "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv", 4),
    ("friday_ddos", "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", 4),
]
```

(line numbers: 34, 35, 36, 37, 38, 39, 40, 41, 42, 43)

`eval/escalation_data.py` lines 175-179

```python
def load_pool(*, use_cache: bool = True, csv_dir: Path = CSV_DIR) -> pd.DataFrame:
    if use_cache and CACHE_PATH.exists():
        return pd.read_parquet(CACHE_PATH)

    frames = []
```

(line numbers: 175, 176, 177, 178, 179)

`agents/escalation_grounding.py` lines 86-99

```python
def load_benign_escalation_reference(force_reload: bool = False) -> pd.DataFrame:
    """Real BENIGN rows from the same pooled CICIDS2017 CSVs the
    escalation policy itself uses, restricted to the 10 escalation
    features. Cached in-process (this project's pool is 2.83M rows;
    re-filtering it on every one of a run's ~20 records is wasted work)."""
    global _benign_reference_cache
    if _benign_reference_cache is not None and not force_reload:
        return _benign_reference_cache
    from eval.escalation_data import BENIGN_LABEL, load_pool  # lazy: this module has no hard dependency on the CSV pool otherwise

    pool = load_pool()
    benign = pool.loc[pool["Label"] == BENIGN_LABEL, list(ESCALATION_FEATURES)].reset_index(drop=True)
    _benign_reference_cache = benign
    return benign
```

(line numbers: 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99)


**Does the reference include benign traffic from the same days as the attacks being evaluated?** For the 2,273,097 population: **yes.** `DAY_FILES` includes Tuesday, Wednesday, Thursday and Friday, and the benign rows from those days are in the reference. For the 566,864 population: **no.** It is Monday only.

What this means for the result: the reference is in-sample for the test flows' own days. A deployed system would only have earlier traffic. So the closeness results describe "does this flow have benign twins somewhere in the same week", not "does this flow look unusual against earlier normal traffic". The result is not a forecast of deployment behaviour.

## 2. What counts as "close"

The closeness test has three parts.

1. **Band per feature.** A benign flow is within a feature's band if its value lies in `[0.5 x observed, 2.0 x observed]`. The band is multiplicative, so it is not symmetric around the observed value. If the observed value is 0, the benign value must be exactly 0.
2. **Joint AND.** A benign flow is a neighbour only if it is in the band on **all ten** escalation features at once. One feature failing excludes the flow.
3. **Raw values.** The comparison is on raw feature values. There is no normalisation, percentile, or binning.

The ten features (`ESCALATION_FEATURES`) are CICFlowMeter features, not the Tier-1 features the profile check uses. That mismatch matters; see Divergences.

`agents/escalation_grounding.py` lines 49-60

```python
ESCALATION_FEATURES = (
    "Bwd Packets/s",
    "Flow IAT Mean",
    "Fwd Packet Length Max",
    "Init_Win_bytes_backward",
    "Init_Win_bytes_forward",
    "Packet Length Mean",
    "Subflow Fwd Packets",
    "Total Backward Packets",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
)
```

(line numbers: 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60)

`agents/escalation_grounding.py` lines 67-68

```python
BAND_LOW_MULT = 0.5
BAND_HIGH_MULT = 2.0
```

(line numbers: 67, 68)

`agents/escalation_grounding.py` lines 137-170

```python
def compute_feature_neighbourhood(
    observed: Dict[str, float],
    benign_df: Optional[pd.DataFrame] = None,
    min_neighbours: int = MIN_NEIGHBOURHOOD_SIZE,
) -> FeatureNeighbourhood:
    """Benign flows within [0.5x, 2x] of this flow's own value (or ==0
    when the flow's own value is 0), jointly across every escalation
    feature both sides have -- drawn from `benign_df` (real BENIGN rows
    only; the caller is responsible for that filter, and
    load_benign_escalation_reference's default already applies it)."""
    benign_df = benign_df if benign_df is not None else load_benign_escalation_reference()
    features = [f for f in ESCALATION_FEATURES if f in observed and f in benign_df.columns]

    mask = pd.Series(True, index=benign_df.index)
    for f in features:
        val = observed[f]
        col = benign_df[f]
        if val == 0:
            mask &= col == 0
        else:
            lo, hi = sorted([val * BAND_LOW_MULT, val * BAND_HIGH_MULT])
            mask &= col.between(lo, hi)

    size = int(mask.sum()) if features else 0
    return FeatureNeighbourhood(
        features_used=features,
        band_description=(
            f"[{BAND_LOW_MULT}x, {BAND_HIGH_MULT}x] of this flow's own observed value per feature "
            f"(or ==0 when observed is 0), joint AND across all {len(features)} feature(s) used"
        ),
        neighbourhood_size=size,
        benign_population_size=int(len(benign_df)),
        ungrounded=(size < min_neighbours),
        min_required=min_neighbours,
```

(line numbers: 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170)


### Worked example A: a flow with zero neighbours (DoS Hulk, record `6846713a`)

Its neighbourhood count is **0** of 2,273,097. It is ungrounded (count below 30). The walk applies the bands in the code's order, one feature at a time, to the same 2,273,097-row pool. The `marginal` column is the count if that band were applied alone; the `cumulative` column is the count after all bands so far.

| feature | this flow's value | band (benign values accepted) | benign flows in this band alone | still in all bands so far |
|---|---|---|---|---|
| Bwd Packets/s | 0.081321 | [0.0406605, 0.162642] | 104,683 | 104,683 |
| Flow IAT Mean | 8.19797e+06 | [4.09898e+06, 1.63959e+07] | 54,462 | 42,649 |
| Fwd Packet Length Max | 372 | [186, 744] | 372,249 | 9,262 |
| Init_Win_bytes_backward | 235 | [117.5, 470] | 390,742 | 5,759 |
| Init_Win_bytes_forward | 0 | [0, 0] | 40,810 | 0 |
| Packet Length Mean | 856.071 | [428.036, 1712.14] | 114,913 | 0 |
| Subflow Fwd Packets | 5 | [2.5, 10] | 451,965 | 0 |
| Total Backward Packets | 8 | [4, 16] | 380,691 | 0 |
| act_data_pkt_fwd | 2 | [1, 4] | 1,220,694 | 0 |
| min_seg_size_forward | 20 | [10, 40] | 2,262,812 | 0 |

No benign flow qualifies because the joint count reaches 0 at `Init_Win_bytes_forward`. The flow's value there is 0, and none of the 5,759 benign flows still in the band had exactly 0 on it. The later features cannot recover what the earlier ones removed.

### Worked example B: a non-zero neighbourhood (PortScan, record `b32f2676`, count 117)

This is one of the six grounded attacks in the reports. Count **117**, not ungrounded.

| feature | this flow's value | band (benign values accepted) | benign flows in this band alone | still in all bands so far |
|---|---|---|---|---|
| Bwd Packets/s | 100000 | [50000, 200000] | 21,596 | 21,596 |
| Flow IAT Mean | 10 | [5, 20] | 39,515 | 21,026 |
| Fwd Packet Length Max | 2 | [1, 4] | 64,822 | 1,698 |
| Init_Win_bytes_backward | 0 | [0, 0] | 109,021 | 1,688 |
| Init_Win_bytes_forward | 1024 | [512, 2048] | 146,940 | 1,688 |
| Packet Length Mean | 3.33333 | [1.66667, 6.66667] | 337,157 | 1,688 |
| Subflow Fwd Packets | 1 | [0.5, 2] | 1,537,368 | 1,688 |
| Total Backward Packets | 1 | [0.5, 2] | 1,384,938 | 1,688 |
| act_data_pkt_fwd | 0 | [0, 0] | 719,652 | 117 |
| min_seg_size_forward | 24 | [12, 48] | 2,270,918 | 117 |

The count falls to 117 at `act_data_pkt_fwd`, where the flow's value is 0. Two of those 117 benign flows are exact copies of this flow's feature vector on all ten features. They are real benign flows with the same values, not the flow itself, which is why the count is not zero.

## 3. The two counts, and what A5 receives

There are two numbers. Both reach A5 in the same sentence.

**(i) The profile count** (`matching_profile_count`). For each hypothesis, how many benign flows fall in its predicted feature ranges. It uses Tier-1 feature names and the **566,864** Monday population.

`agents/grounding.py` lines 93-140

```python
def evaluate_hypothesis_support(
    hypothesis: Hypothesis,
    trigger_reason_values: Dict[str, float],
    benign_df: pd.DataFrame,
) -> HypothesisSupport:
    """``trigger_reason_values``: TRIGGER_FEATURE_PREFIX-prefixed feature
    name -> this flow's own observed value (from its trigger_reasons --
    the only Tier-1 values this flow actually has; see
    controlplane/reference.py::NearestNeighbourResult's docstring for why
    that's the full extent of what's available)."""
    profile = hypothesis.predicted_feature_profile
    mask = pd.Series(True, index=benign_df.index)
    any_feature_present = False
    for p in profile:
        bare = _strip_prefix(p.feature)
        if bare not in benign_df.columns:
            continue
        any_feature_present = True
        col = benign_df[bare]
        col_mask = col.notna()
        if p.expected_min is not None:
            col_mask &= col >= p.expected_min
        if p.expected_max is not None:
            col_mask &= col <= p.expected_max
        mask &= col_mask

    if not any_feature_present:
        return HypothesisSupport(hypothesis.hypothesis_id, 0, None, [], len(benign_df))

    matching = benign_df[mask]
    matching_count = int(len(matching))

    checked_features: List[str] = []
    close_mask = pd.Series(True, index=matching.index)
    for p in profile:
        bare = _strip_prefix(p.feature)
        observed = trigger_reason_values.get(p.feature)
        if observed is None or bare not in matching.columns:
            continue
        checked_features.append(p.feature)
        col = matching[bare]
        if observed == 0:
            close_mask &= col == 0
        else:
            lo, hi = sorted([observed / 2.0, observed * 2.0])
            close_mask &= col.between(lo, hi)

    close_count = int(close_mask.sum()) if checked_features else None
```

(line numbers: 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140)

**(ii) The closeness count** (`close_to_observed_count`). For the same flow, how many benign flows are within the band on the ten escalation features. The v3 pipeline computes this from the **2,273,097** pool and writes it into every hypothesis's support, overwriting the Tier-1 value (which is `None` here, because `trigger_reasons` is empty for escalated flows).

`agents/escalation_grounding.py` lines 174-186

```python
def patch_hypothesis_support(
    supports: Dict[str, HypothesisSupport], neighbourhood: FeatureNeighbourhood,
) -> Dict[str, HypothesisSupport]:
    """Replaces every hypothesis's close_to_observed_count/checked_features
    for this record with the SAME record-level neighbourhood -- "how many
    real benign flows resemble this flow" is a property of the flow, not
    of any one hypothesis, so every hypothesis on the same record gets
    the same, now-always-DEFINED number (never None again for this
    reason). `matching_profile_count` (the existing Tier-1 profile check)
    is left untouched -- this only repairs the closeness refinement,
    it does not replace the generic check."""
    close_count = 0 if neighbourhood.ungrounded else neighbourhood.neighbourhood_size
    return {
```

(line numbers: 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186)

**Divergence:** the profile count's denominator (566,864) and the closeness count's denominator (2,273,097) come from different populations, shown together to A5 as one sentence. The stored v3 record for hypothesis `a3_h1` on the zero-count flow shows the mismatch:

```json
{
 "hypothesis_id": "a3_h1",
 "matching_profile_count": 33656,
 "close_to_observed_count": 0,
 "checked_features": [
  "Bwd Packets/s",
  "Flow IAT Mean",
  "Fwd Packet Length Max",
  "Init_Win_bytes_backward",
  "Init_Win_bytes_forward",
  "Packet Length Mean",
  "Subflow Fwd Packets",
  "Total Backward Packets",
  "act_data_pkt_fwd",
  "min_seg_size_forward"
 ],
 "benign_population_size": 566864
}
```

`benign_population_size` is 566,864 in that record, while the flow's escalation neighbourhood reports `benign_population_size` 2,273,097.

The sentence A5 receives is built here:

`agents/grounding.py` lines 201-215

```python
def render_hypothesis_support(support: HypothesisSupport) -> str:
    if support.matching_profile_count == 0:
        verdict = (
            f"EMPIRICALLY UNSUPPORTED -- zero of {support.benign_population_size} benign flows "
            "observed on this network match this predicted profile"
        )
    else:
        verdict = f"{support.matching_profile_count} benign flow(s) match this predicted profile"
        if support.checked_features:
            verdict += (
                f", of which {support.close_to_observed_count} are within 2x of this flow's own "
                f"observed values on {', '.join(support.checked_features)}"
            )
        else:
            verdict += (
```

(line numbers: 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215)

**The EMPIRICALLY UNSUPPORTED branch fires only when the profile count is 0.** In the zero-count example the profile counts are large (see below), so A5 never sees "unsupported". It sees "many match, of which 0 are within 2x". That is the mechanism behind the six misses.

Sample support lines from the verbatim block:

```
Every candidate explanation below now carries its empirical support: how many real benign flows observed on this network actually match its predicted feature profile, and how many of those are close (within 2x) to this flow's own observed values. This was computed by querying real benign traffic, not estimated by either reviewer. A hypothesis marked EMPIRICALLY UNSUPPORTED matched zero real benign flows -- treat that as a fact about this network, not a technicality. A hypothesis with many matching flows but none close to what this flow actually shows is weaker than its raw match count suggests: it means flows like that exist, but not flows that look like THIS one.
      empirical support: 33656 benign flow(s) match this predicted profile, of which 0 are within 2x of this flow's own observed values on Bwd Packets/s, Flow IAT Mean, Fwd Packet Length Max, Init_Win_bytes_backward, Init_Win_bytes_forward, Packet Length Mean, Subflow Fwd Packets, Total Backward Packets, act_data_pkt_fwd, min_seg_size_forward
      empirical support: 29549 benign flow(s) match this predicted profile, of which 0 are within 2x of this flow's own observed values on Bwd Packets/s, Flow IAT Mean, Fwd Packet Length Max, Init_Win_bytes_backward, Init_Win_bytes_forward, Packet Length Mean, Subflow Fwd Packets, Total Backward Packets, act_data_pkt_fwd, min_seg_size_forward
```

**The verbatim A5 input, for record `6846713a` (DoS Hulk, count 0).** This is the full prompt A5 received in the v3 run, rebuilt from the saved upstream outputs with the v6 prompt builder. No model call was made to produce it.

```text
Context: this flow was selected for review by an automated system using purely statistical thresholds fitted on ordinary traffic from this network. Nothing about being selected implies a conclusion -- treat this as one flow's evidence, to be examined on its own terms.

Calibration context: every "typical-range threshold" you see (here and in any claim using greater_than_typical/less_than_typical) was fit using ONLY ordinary (benign) traffic from this network -- never from the kind of traffic this review process exists to catch, so a threshold reflects what is statistically rare for ordinary use, not a guess. This record's flagged measurements were selected so that, combined, only about 5% of ordinary traffic would ever cross two or more such thresholds at once (this project's default rule); crossing a single feature's own threshold alone is markedly rarer than that for most features. A "ratio" of 1.5x means the observed value is 1.5 times the fitted threshold itself -- values near 1.0x are common right at the boundary and not remarkable on their own; larger ratios represent increasingly rare deviations from ordinary traffic on this specific network.

You are the final reviewer for one network flow. You are not being asked to judge this traffic's intent. Your question is: given the benign traffic actually observed on this network, how many benign flows resemble this one, and does any surviving hypothesis have empirical support?

Below you'll see the combined output of a three-stage review chain (evidence, behaviour, candidate explanations) and a separate, fully independent review of the same flow performed without seeing the chain's output. You'll also see a mechanical comparison of which claims from the two sides corroborate, contradict, or simply don't overlap.

Every candidate explanation below now carries its empirical support: how many real benign flows observed on this network actually match its predicted feature profile, and how many of those are close (within 2x) to this flow's own observed values. This was computed by querying real benign traffic, not estimated by either reviewer. A hypothesis marked EMPIRICALLY UNSUPPORTED matched zero real benign flows -- treat that as a fact about this network, not a technicality. A hypothesis with many matching flows but none close to what this flow actually shows is weaker than its raw match count suggests: it means flows like that exist, but not flows that look like THIS one.

Before crediting any benign explanation, check both its empirical support above and whether its stated prediction holds against the INDEPENDENT review's claims (the side that did NOT generate that explanation). A benign explanation with strong empirical support AND a confirmed prediction is real corroboration. A benign explanation that merely sounds plausible, with weak or zero empirical support, is not --  do not credit it just because no one has disproven it in words.

MANDATORY: every claim_id listed below under "claims you must address" was flagged by SOME hypothesis's own generating reviewer as contradicting that hypothesis. You must cite every single one of them in cited_claim_ids -- this is checked mechanically, and a response missing even one will be rejected and you will be asked to try again. Citing a claim_id there does not mean you have to agree it's fatal: for each one, either explain in your rationale why it doesn't actually undermine the hypothesis you're crediting, or let it lower your benign_plausibility estimate accordingly. What you may not do is simply not mention it.

Report:
- credited_hypothesis_id: the single hypothesis_id (from either side) your benign_plausibility estimate actually rests on, or null if no hypothesis survives well enough to credit at all. NOTE: if the hypothesis you name here has zero empirically-matching benign flows, your benign_plausibility will be capped at 0.3 regardless of what you report -- this is enforced in code after you respond, not something you need to self-apply, but your own estimate should already reflect it: do not report a high plausibility for a hypothesis the data does not support.
- benign_plausibility, from 0 to 1 -- your estimate of how plausible the credited hypothesis is, once you have weighed its empirical support and checked its prediction against the independent evidence. 0 means nothing survives; 1 means a benign explanation is fully confirmed by both real benign traffic and independent evidence, with nothing left unexplained. Do not round to a convenient category -- report your actual estimate, including values close to 0.5 if that's genuinely where the evidence leaves you.

Do not try to name what kind of activity this is or how severe it is; that is out of scope here.

Cite the claim_ids (from either side, including every claim_id listed under "claims you must address" above) that most influenced your estimate, and give your own confidence (0 to 1) in the estimate itself. In your rationale, cite the empirical support numbers (match count, close-to-observed count) for the hypothesis you credited, explain whether its prediction was confirmed or contradicted by the independent evidence, and how you addressed each contradicting claim you were required to cite.

Also report three overall self-assessments for this verdict, each from 0 to 1 -- distinct from cited_claim_ids and the rationale, which explain your reasoning, not rate it:
- confidence: your overall confidence in this benign_plausibility estimate.
- evidence_support: your own estimate of how well the evidence you cited above would hold up if checked directly against the record's actual data.
- verification: your own estimate of how well this verdict is actually supported by the mechanical corroboration between the chain and the independent review shown to you above -- unlike A1-A4, you have already been shown that comparison, so base this on it, not a blind guess.
Report your real estimate for each of the three, even where you expect them to diverge from each other.


Full feature set for this flow:
  ACK Flag Count = 0
  Active Max = 11963
  Active Mean = 11963
  Active Min = 11963
  Active Std = 0
  Average Packet Size = 921.923
  Avg Bwd Segment Size = 1449.38
  Avg Fwd Segment Size = 76.8
  Bwd Avg Bulk Rate = 0
  Bwd Avg Bytes/Bulk = 0
  Bwd Avg Packets/Bulk = 0
  Bwd Header Length = 264
  Bwd IAT Max = 9.84e+07
  Bwd IAT Mean = 1.41e+07
  Bwd IAT Min = 2
  Bwd IAT Std = 3.72e+07
  Bwd IAT Total = 9.84e+07
  Bwd PSH Flags = 0
  Bwd Packet Length Max = 5792
  Bwd Packet Length Mean = 1449.38
  Bwd Packet Length Min = 0
  Bwd Packet Length Std = 2047.78
  Bwd Packets/s = 0.081321
  Bwd URG Flags = 0
  CWE Flag Count = 0
  Destination Port = 80
  Down/Up Ratio = 1
  ECE Flag Count = 0
  FIN Flag Count = 1
  Flow Bytes/s = 121.768
  Flow Duration = 9.83756e+07
  Flow IAT Max = 9.84e+07
  Flow IAT Mean = 8.19797e+06
  Flow IAT Min = 2
  Flow IAT Std = 2.84e+07
  Flow Packets/s = 0.132147
  Fwd Avg Bulk Rate = 0
  Fwd Avg Bytes/Bulk = 0
  Fwd Avg Packets/Bulk = 0
  Fwd Header Length = 144
  Fwd Header Length.1 = 144
  Fwd IAT Max = 9.84e+07
  Fwd IAT Mean = 2.46e+07
  Fwd IAT Min = 553
  Fwd IAT Std = 4.92e+07
  Fwd IAT Total = 9.84e+07
  Fwd PSH Flags = 0
  Fwd Packet Length Max = 372
  Fwd Packet Length Mean = 76.8
  Fwd Packet Length Min = 0
  Fwd Packet Length Std = 165.049
  Fwd Packets/s = 0.0508256
  Fwd URG Flags = 0
  Idle Max = 9.84e+07
  Idle Mean = 9.84e+07
  Idle Min = 9.84e+07
  Idle Std = 0
  Init_Win_bytes_backward = 235
  Init_Win_bytes_forward = 0
  Max Packet Length = 5792
  Min Packet Length = 0
  PSH Flag Count = 0
  Packet Length Mean = 856.071
  Packet Length Std = 1664.98
  Packet Length Variance = 2.77215e+06
  RST Flag Count = 0
  SYN Flag Count = 0
  Subflow Bwd Bytes = 11595
  Subflow Bwd Packets = 8
  Subflow Fwd Bytes = 384
  Subflow Fwd Packets = 5
  Total Backward Packets = 8
  Total Fwd Packets = 5
  Total Length of Bwd Packets = 11595
  Total Length of Fwd Packets = 384
  URG Flag Count = 0
  act_data_pkt_fwd = 2
  min_seg_size_forward = 20

Multi-signal benign evidence for this flow:
Benign-only evidence for this flow. Every figure comes from real benign traffic on this network. No rule for combining these signals is given.

1. Percentile of each feature within benign traffic (0 = lowest benign value, 100 = highest):
   - Bwd Packets/s: percentile 16.65 (flow value 0.081321)
   - Flow IAT Mean: percentile 97.98 (flow value 8.19797e+06)
   - Fwd Packet Length Max: percentile 83.90 (flow value 372)
   - Init_Win_bytes_backward: percentile 75.29 (flow value 235)
   - Init_Win_bytes_forward: percentile 45.84 (flow value 0)
   - Packet Length Mean: percentile 98.28 (flow value 856.071)
   - Subflow Fwd Packets: percentile 79.23 (flow value 5)
   - Total Backward Packets: percentile 86.76 (flow value 8)
   - act_data_pkt_fwd: percentile 76.13 (flow value 2)
   - min_seg_size_forward: percentile 51.50 (flow value 20)

2. Direction against benign:
   - Bwd Packets/s: below benign median
   - Flow IAT Mean: above benign median
   - Fwd Packet Length Max: above benign median
   - Init_Win_bytes_backward: above benign median
   - Init_Win_bytes_forward: below benign median
   - Packet Length Mean: above benign median
   - Subflow Fwd Packets: above benign median
   - Total Backward Packets: above benign median
   - act_data_pkt_fwd: above benign median
   - min_seg_size_forward: at benign median

3. Nearest benign flows (distance in standardised feature units; the flow's own reference copy is removed if present). This flow's values first, then each neighbour's:
   this flow: Bwd Packets/s=0.081321, Flow IAT Mean=8.19797e+06, Fwd Packet Length Max=372, Init_Win_bytes_backward=235, Init_Win_bytes_forward=0, Packet Length Mean=856.071, Subflow Fwd Packets=5, Total Backward Packets=8, act_data_pkt_fwd=2, min_seg_size_forward=20
   neighbour 1 (distance 1.430): Bwd Packets/s=0.0533449, Flow IAT Mean=1.04144e+07, Fwd Packet Length Max=1429, Init_Win_bytes_backward=62, Init_Win_bytes_forward=62, Packet Length Mean=524.182, Subflow Fwd Packets=5, Total Backward Packets=5, act_data_pkt_fwd=4, min_seg_size_forward=20
   neighbour 2 (distance 1.432): Bwd Packets/s=0.0502641, Flow IAT Mean=1.10527e+07, Fwd Packet Length Max=1429, Init_Win_bytes_backward=62, Init_Win_bytes_forward=62, Packet Length Mean=513.455, Subflow Fwd Packets=5, Total Backward Packets=5, act_data_pkt_fwd=4, min_seg_size_forward=20
   neighbour 3 (distance 1.472): Bwd Packets/s=0.0791148, Flow IAT Mean=7.77837e+06, Fwd Packet Length Max=1429, Init_Win_bytes_backward=62, Init_Win_bytes_forward=62, Packet Length Mean=579.4, Subflow Fwd Packets=6, Total Backward Packets=8, act_data_pkt_fwd=5, min_seg_size_forward=20
   neighbour 4 (distance 1.492): Bwd Packets/s=0.0555153, Flow IAT Mean=1.00072e+07, Fwd Packet Length Max=630, Init_Win_bytes_backward=305, Init_Win_bytes_forward=256, Packet Length Mean=489.909, Subflow Fwd Packets=5, Total Backward Packets=5, act_data_pkt_fwd=4, min_seg_size_forward=20
   neighbour 5 (distance 1.502): Bwd Packets/s=0.0854771, Flow IAT Mean=6.15739e+06, Fwd Packet Length Max=281, Init_Win_bytes_backward=123, Init_Win_bytes_forward=257, Packet Length Mean=510.238, Subflow Fwd Packets=10, Total Backward Packets=10, act_data_pkt_fwd=2, min_seg_size_forward=20

4. Neighbourhood count within [0.5x, 2x] on all ten features: 0 benign flows; 0 after leave-one-out (no identical benign flow present, none removed).

5. Source history (benign flows from this source IP, earlier buckets only), 5-minute buckets before this flow, most recent first:
   -5 to 0 min: 63 flows, 63 distinct destination ports, 126 packets
   -10 to -5 min: 59 flows, 59 distinct destination ports, 118 packets
   -15 to -10 min: 82 flows, 82 distinct destination ports, 164 packets
   -20 to -15 min: 100 flows, 100 distinct destination ports, 200 packets
   -25 to -20 min: 84 flows, 84 distinct destination ports, 171 packets
   -30 to -25 min: 83 flows, 83 distinct destination ports, 166 packets


Chain review -- combined claims (evidence + behaviour + hypothesis stages):
  [a1_c1] (confidence 0.95) Flow IAT Mean is at the 97.98th percentile of benign traffic, indicating a relatively high inter-arrival time mean. -- refs: Flow IAT Mean greater_than_typical (asserted 8.19797e+06)
  [a1_c2] (confidence 0.95) Packet Length Mean is at the 98.28th percentile of benign traffic, representing an elevated mean packet length. -- refs: Packet Length Mean greater_than_typical (asserted 856.071)
  [a1_c3] (confidence 0.90) Bwd Packets/s is at the 16.65th percentile of benign traffic, which is below the benign median. -- refs: Bwd Packets/s less_than_typical (asserted 0.081321)
  [a1_c4] (confidence 0.90) Init_Win_bytes_forward is at the 45.84th percentile of benign traffic with a value of zero. -- refs: Init_Win_bytes_forward less_than_typical (asserted 0)
  [a2_c1] (confidence 0.95) The backward direction sends significantly more data and larger payloads compared to the forward direction, as shown by the stark difference in average segment and packet sizes. -- refs: Avg Bwd Segment Size greater_than_typical (asserted 1449.38); Avg Fwd Segment Size less_than_typical (asserted 76.8)
  [a2_c2] (confidence 0.90) The connection remains active over an extended duration with very low packet rates, indicating a persistent but highly intermittent exchange rather than a rapid transfer. -- refs: Flow Duration greater_than_typical (asserted 9.83756e+07); Flow Packets/s less_than_typical (asserted 0.132147)
  [a2_c3] (confidence 0.95) The communication concludes normally with a finish flag, indicating that the session terminated through standard connection teardown procedures rather than being abruptly cut short. -- refs: FIN Flag Count present (asserted 1)
  [a2_c4] (confidence 0.85) The long idle interval dominates the entire flow duration, suggesting this is a single isolated exchange featuring a prolonged pause before completion. -- refs: Idle Mean greater_than_typical (asserted 9.84e+07)
  [a3_c1] (confidence 0.95) The destination port is 80, suggesting standard unencrypted HTTP traffic. -- refs: Destination Port equals (asserted 80)

Chain review -- candidate explanations (with empirical support):
  [a3_h1] (benign, prior plausibility 0.80) An automated monitoring system or health-check tool periodically polling an HTTP endpoint over a long interval, causing very low packet rates and extended idle times.
      prediction: The flow duration should be long with very few packets per second and long inter-arrival times reflecting scheduled polling intervals.
      predicted_feature_profile: trigger_flow_duration [1e+07, 2e+08]; trigger_flow_pkts_per_sec <= 2; trigger_flow_iat_mean >= 1e+06
      supporting: a1_c1, a2_c2, a2_c4, a3_c1
      contradicting: (none)
      empirical support: 33656 benign flow(s) match this predicted profile, of which 0 are within 2x of this flow's own observed values on Bwd Packets/s, Flow IAT Mean, Fwd Packet Length Max, Init_Win_bytes_backward, Init_Win_bytes_forward, Packet Length Mean, Subflow Fwd Packets, Total Backward Packets, act_data_pkt_fwd, min_seg_size_forward
  [a3_h2] (benign, prior plausibility 0.75) A client application downloading a small configuration file or a small web resource over HTTP where the server responds with a chunked or larger payload size relative to the request.
      prediction: The backward packet length mean and backward total bytes will significantly exceed the forward totals, reflecting asymmetric web retrieval.
      predicted_feature_profile: trigger_bwd_pkt_len_mean [500, 6000]; trigger_bwd_fwd_byte_ratio >= 5
      supporting: a1_c2, a2_c1, a3_c1
      contradicting: (none)
      empirical support: 29549 benign flow(s) match this predicted profile, of which 0 are within 2x of this flow's own observed values on Bwd Packets/s, Flow IAT Mean, Fwd Packet Length Max, Init_Win_bytes_backward, Init_Win_bytes_forward, Packet Length Mean, Subflow Fwd Packets, Total Backward Packets, act_data_pkt_fwd, min_seg_size_forward
  [a3_h3] (benign, prior plausibility 0.60) A misconfigured client or backup script performing intermittent requests with long pauses between connection retries or data exchanges.
      prediction: The flow will exhibit extended idle periods and irregular inter-arrival times with low overall packet counts.
      predicted_feature_profile: trigger_flow_iat_max >= 1e+07; trigger_flows_per_src <= 200
      supporting: a2_c2, a2_c4
      contradicting: (none)
      empirical support: 25230 benign flow(s) match this predicted profile, of which 0 are within 2x of this flow's own observed values on Bwd Packets/s, Flow IAT Mean, Fwd Packet Length Max, Init_Win_bytes_backward, Init_Win_bytes_forward, Packet Length Mean, Subflow Fwd Packets, Total Backward Packets, act_data_pkt_fwd, min_seg_size_forward

Independent review -- claims:
  [a4_c1] (confidence 0.99) The flow duration is approximately 98.38 seconds. -- refs: Flow Duration equals (asserted 9.83756e+07)
  [a4_c2] (confidence 0.99) Total forward packets equal 5. -- refs: Total Fwd Packets equals (asserted 5)
  [a4_c3] (confidence 0.99) Total backward packets equal 8. -- refs: Total Backward Packets equals (asserted 8)
  [a4_c4] (confidence 0.99) Total length of forward packets is 384 bytes. -- refs: Total Length of Fwd Packets equals (asserted 384)
  [a4_c5] (confidence 0.99) Total length of backward packets is 11595 bytes. -- refs: Total Length of Bwd Packets equals (asserted 11595)
  [a4_c6] (confidence 0.99) Destination port is 80, indicating standard HTTP traffic. -- refs: Destination Port equals (asserted 80)
  [a4_c7] (confidence 0.99) FIN flag count is 1, indicating connection termination. -- refs: FIN Flag Count equals (asserted 1)

Independent review -- candidate explanations (with empirical support):
  [a4_h1] (benign, prior plausibility 0.80) A client sending a short query or request payload and receiving a larger data response over a persistent or idle connection.
      prediction: The flow duration is extended with low overall packet rate and asymmetric data transfer where backward bytes vastly exceed forward bytes.
      predicted_feature_profile: trigger_flow_duration [1e+07, 1e+08]
      supporting: a4_c1, a4_c4, a4_c5
      contradicting: (none)
      empirical support: 26967 benign flow(s) match this predicted profile, of which 0 are within 2x of this flow's own observed values on Bwd Packets/s, Flow IAT Mean, Fwd Packet Length Max, Init_Win_bytes_backward, Init_Win_bytes_forward, Packet Length Mean, Subflow Fwd Packets, Total Backward Packets, act_data_pkt_fwd, min_seg_size_forward
  [a4_h2] (benign, prior plausibility 0.70) An automated health check or monitoring service polling an endpoint with minimal payload and receiving a status response.
      prediction: Low packet counts with intermittent activity spanning a long idle duration.
      predicted_feature_profile: trigger_flow_duration [1e+06, 2e+08]
      supporting: a4_c2, a4_c3
      contradicting: (none)
      empirical support: 95692 benign flow(s) match this predicted profile, of which 0 are within 2x of this flow's own observed values on Bwd Packets/s, Flow IAT Mean, Fwd Packet Length Max, Init_Win_bytes_backward, Init_Win_bytes_forward, Packet Length Mean, Subflow Fwd Packets, Total Backward Packets, act_data_pkt_fwd, min_seg_size_forward
  [a4_h3] (non-benign, prior plausibility 0.30) A slow-loris style or delayed transmission client-side behavior holding the connection open.
      prediction: Prolonged flow duration accompanied by low flow bytes per second and irregular packet inter-arrival times.
      predicted_feature_profile: trigger_flow_bytes_per_sec [0, 500]
      supporting: a4_c1
      contradicting: (none)
      empirical support: 62673 benign flow(s) match this predicted profile, of which 0 are within 2x of this flow's own observed values on Bwd Packets/s, Flow IAT Mean, Fwd Packet Length Max, Init_Win_bytes_backward, Init_Win_bytes_forward, Packet Length Mean, Subflow Fwd Packets, Total Backward Packets, act_data_pkt_fwd, min_seg_size_forward

Mechanical agreement between chain and independent review (per chain stage: corroborated / contradicted / not addressed):
  a1: 0 corroborated, 0 contradicted, 4 not addressed by the other side (of 4 checkable claims)
  a2: 0 corroborated, 1 contradicted, 5 not addressed by the other side (of 6 checkable claims)
  a3: 1 corroborated, 0 contradicted, 0 not addressed by the other side (of 1 checkable claims)

Claims you must address (cite every one of these in cited_claim_ids -- see MANDATORY above):
  (none -- no hypothesis below has flagged any claim as contradicting it)

```

## 4. Thresholds

**Is 30 the real cutoff?** Yes, in the code:

`agents/escalation_grounding.py` lines 72-72

```python
MIN_NEIGHBOURHOOD_SIZE = 30
```

(line numbers: 72)

**Where else it is used.** The cutoff only sets the `ungrounded` flag (`size < min_neighbours`). Three things depend on that flag:

1. `patch_hypothesis_support` sets the closeness count to **0** when ungrounded (quoted in section 3). A5 sees "of which 0 are within 2x" for every hypothesis.
2. `apply_ungrounded_neighbourhood_cap` clamps `benign_plausibility` when the flag is set. The check `if not neighbourhood.ungrounded:` is in the same file; the clamp is at the cap value quoted below.
3. The stored `ungrounded` boolean, used in the reports.

**Does A5 get a numeric rule?** No. The v6 prompt (and v5 on this point) says the closeness numbers should be weighed, that EMPIRICALLY UNSUPPORTED means zero matches, and that many matches with none close is weaker. It never gives a neighbour count below which a flow is suspicious, and it does not mention 30. A5 is not told the word "ungrounded".

**The cap and its value.** The clamp value is 0.29 in the code. The prompt still says 0.3:

`agents/escalation_grounding.py` lines 81-81

```python
UNGROUNDED_PLAUSIBILITY_CAP = 0.29
```

(line numbers: 81)

`agents/grounding.py` lines 39-39

```python
ZERO_SUPPORT_PLAUSIBILITY_CAP = 0.29
```

(line numbers: 39)

`agents/prompts/a5_verdict_v6.py` lines 86-89

```python
hypothesis survives well enough to credit at all. NOTE: if the \
hypothesis you name here has zero empirically-matching benign flows, \
your benign_plausibility will be capped at 0.3 regardless of what you \
report -- this is enforced in code after you respond, not something you \
```

(line numbers: 86, 87, 88, 89)

**The mismatch is still present.** The prompt states 0.3 and the code clamps at 0.29. A5 can report 0.30, and the code then lowers it to 0.29.

## 5. Distribution of the count, over the 44 scored records

Attacks (29), sorted by count. The leave-one-out column removes the one identical benign copy per flow, as the new evidence block does. The pipeline's own count does not.

| record | true label | neighbourhood count (as the pipeline computes it) | leave-one-out count |
|---|---|---|---|
| 6846713a33 | DoS Hulk | 0 | 0 |
| 2948aa4982 | DoS Hulk | 0 | 0 |
| 405408a974 | FTP-Patator | 0 | 0 |
| 7437f3e358 | DDoS | 0 | 0 |
| 3aedc4e3b0 | DDoS | 0 | 0 |
| 421c39412b | DDoS | 0 | 0 |
| dd7d54642f | Bot | 0 | 0 |
| aa23704f6b | DoS Hulk | 0 | 0 |
| 49bcc3b140 | DoS Slowhttptest | 1 | 1 |
| 56d0f28196 | DoS Slowhttptest | 1 | 1 |
| 25323bd708 | PortScan | 2 | 2 |
| a2123a0401 | DoS Hulk | 4 | 4 |
| 681c501fd3 | DoS Hulk | 9 | 9 |
| fb0ed3f72f | DoS GoldenEye | 17 | 17 |
| d67a88f41a | SSH-Patator | 30 | 30 |
| 70448e2e53 | DDoS | 62 | 62 |
| d07e7212eb | DDoS | 76 | 76 |
| 2c27e52883 | DDoS | 88 | 88 |
| 5b39770724 | DoS GoldenEye | 114 | 114 |
| b32f2676f5 | PortScan | 117 | 116 |
| 145e00165f | DoS GoldenEye | 512 | 512 |
| 7991405abc | DoS Slowhttptest | 552 | 551 |
| cd5ffd67ca | DoS Slowhttptest | 552 | 551 |
| 4d33af2908 | Bot | 1,611 | 1,611 |
| fb4f6abaa8 | Bot | 1,611 | 1,611 |
| 3b893a5989 | Bot | 1,611 | 1,611 |
| 2b7954cb1b | Bot | 1,614 | 1,614 |
| 9721796b2e | DoS Hulk | 12,902 | 12,901 |
| 3da627f972 | SSH-Patator | 26,855 | 26,855 |

Benign (15), sorted by count.

| record | true label | neighbourhood count (as the pipeline computes it) | leave-one-out count |
|---|---|---|---|
| 2031095dbd | BENIGN | 2,069 | 2,068 |
| 5725bf68e6 | BENIGN | 2,807 | 2,806 |
| ce2915d6ca | BENIGN | 3,710 | 3,709 |
| b740cddeb3 | BENIGN | 4,095 | 4,094 |
| b38f27e379 | BENIGN | 4,137 | 4,136 |
| 445a062b00 | BENIGN | 6,445 | 6,444 |
| 03c6f14dc5 | BENIGN | 11,979 | 11,978 |
| f56d600786 | BENIGN | 76,097 | 76,096 |
| 216bba5c6c | BENIGN | 100,695 | 100,694 |
| e9324f3833 | BENIGN | 135,668 | 135,667 |
| 5db4f0dca1 | BENIGN | 170,961 | 170,960 |
| d81abee130 | BENIGN | 171,607 | 171,606 |
| 84e78ebc09 | BENIGN | 278,844 | 278,843 |
| e209f4618e | BENIGN | 294,704 | 294,703 |
| 5658142a21 | BENIGN | 306,920 | 306,919 |

Attacks: 8 of 29 have count 0; 6 have 1 to 29; 6 have 30 to 117; 9 have more than 117. Benign: smallest 2,069, largest 306,920.

The six grounded attacks in the reports (counts 30 to 117) are d67a88f4 (30), 70448e2e (62), d07e7212 (76), 2c27e528 (88), 5b397707 (114) and b32f2676 (117).

## Divergences from the reports

Each divergence is stated with its code location.

1. **Two benign populations in one sentence.** The reports present the 2.27M benign reference as the single grounding population. The profile count uses the 566,864 Monday parquet: `agents/pipeline.py` line 310 passes `grounding.benign_df`, which `compute_grounding` (line 158) fills with `load_benign_dataframe()` from `controlplane/reference.py`. The closeness count uses the 2.27M pool. A5 reads both in one sentence.
2. **The flow's own copy is counted for benign flows.** The pool contains benign flows, so a benign flow matches itself on all ten features. For all 15 benign records in the 44, the leave-one-out count is exactly one lower. The fix is in the new evidence signals only (`agents/evidence_signals.py`). The cap, the ungrounded flag and the closeness numbers A5 reads still include the copy. This does not change any cutoff result on these 44 records, because no benign count is near a cutoff. It is still a divergence from "a neighbourhood of other flows".
3. **The reference shares days with the test attacks.** Stated in section 1. The reports do not say this.
4. **The profile count and the closeness count use different feature spaces.** The profile count uses Tier-1 features. The closeness count uses CICFlowMeter escalation features. The sentence A5 reads pairs them, and its phrasing suggests one feature space. They are two.
5. **The cutoff is not a rule A5 can read.** 30 appears only in code and as the cap's trigger. A5 sees zeros, not the word ungrounded.
6. **The cap is 0.29 in code and 0.3 in the prompt.** Still present.
7. **Two different extraction paths.** The 566,864 population is a PCAP-simulation parquet (`monday_pcap__fidelity__adapter2__features4`). The 2,273,097 pool is built from the CICIDS2017 CSVs. The reports do not distinguish them, and feature values for the same flow may not match exactly between them.
8. **Not re-verified here.** The reports' figure of 258,258 of 566,864 (45.6%) for one hypothesis comes from an earlier run whose files are not in the current results set, so I could not re-check it. The code path that produces that kind of number is the profile count in section 3(i).

"""Cost-controlled runner for the five-agent pipeline.

Selection can use the ground-truth manifest (to build a stratified
sample) -- that's a decision made by this script, outside the pipeline,
and never something any agent sees. Nothing here passes a label into a
prompt; see tests/test_pipeline.py::test_no_ground_truth_label_reaches_any_prompt
for the structural check.

Examples::

    # stop point 1: 2 records, full raw output
    python -m eval.run_agent_pipeline --n 2 --print-raw \\
        --out results/agent_pipeline_2.jsonl

    # stop point 2: 20 records, 5 per true class, full raw output
    python -m eval.run_agent_pipeline --per-class 5 --print-raw \\
        --out results/agent_pipeline_20.jsonl

    # stop point 3: all 200 records
    python -m eval.run_agent_pipeline --records results/escalation_records_sample.jsonl \\
        --out results/agent_pipeline_200.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from agents import base, pipeline
from agents.pipeline import RecordResult
from agents.schema import derive_verdict
from controlplane.record import EscalationRecord


def already_written_record_ids(out_path: Optional[str]) -> set:
    """flow_ids already present in ``--out`` from a previous (interrupted)
    run. agents/base.py's disk cache makes re-running a previously-done
    record free in API cost, but ``pipeline.append_result_jsonl`` just
    appends -- without this check, resuming a killed run re-appends a
    duplicate line for every record that was already finished, silently
    double-counting it in any downstream analysis. Checked once up front
    rather than per-record so a resumed run doesn't re-run (even for
    free) the 5 cached agent calls for records it's going to skip anyway."""
    if not out_path or not Path(out_path).exists():
        return set()
    seen = set()
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                seen.add(json.loads(line)["record_id"])
    return seen


def load_manifest(path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[row["flow_id"]] = row["label"]
    return out


def stratified_sample(
    records: List[EscalationRecord], manifest: Dict[str, str], per_class: int
) -> List[EscalationRecord]:
    by_class: Dict[str, List[EscalationRecord]] = {}
    for r in records:
        label = manifest.get(r.flow_id, "UNKNOWN")
        by_class.setdefault(label, []).append(r)
    out: List[EscalationRecord] = []
    for label in sorted(by_class):
        out.extend(by_class[label][:per_class])
    return out


def _render_claims_block(claims) -> str:
    lines = []
    for c in claims:
        refs = "; ".join(
            f"{r.name} {r.relation.value}" + (f"={r.asserted_value:g}" if r.asserted_value is not None else "")
            for r in c.referenced_features
        )
        lines.append(
            f"    [{c.claim_id}] (conf {c.confidence:.2f}) {c.statement}" + (f"  <{refs}>" if refs else "")
        )
    return "\n".join(lines) if lines else "    (none)"


def _render_predicted_profile(profile) -> str:
    parts = []
    for p in profile:
        if p.expected_min is not None and p.expected_max is not None:
            bounds = f"[{p.expected_min:g}, {p.expected_max:g}]"
        elif p.expected_min is not None:
            bounds = f">= {p.expected_min:g}"
        else:
            bounds = f"<= {p.expected_max:g}"
        parts.append(f"{p.feature} {bounds}")
    return "; ".join(parts) or "(none)"


def _render_hypotheses_block(hyps, support_by_id=None) -> str:
    support_by_id = support_by_id or {}
    lines = []
    for h in hyps:
        tag = "BENIGN" if h.benign else "non-benign"
        block = (
            f"    [{h.hypothesis_id}] ({tag}, prior {h.prior_plausibility:.2f}) {h.description}\n"
            f"        prediction: {h.prediction}\n"
            f"        predicted_feature_profile: {_render_predicted_profile(h.predicted_feature_profile)}\n"
            f"        + {', '.join(h.supporting_claim_ids) or '(none)'}   "
            f"- {', '.join(h.contradicting_claim_ids) or '(none)'}"
        )
        support = support_by_id.get(h.hypothesis_id)
        if support is not None:
            block += (
                f"\n        empirical: matching={support.matching_profile_count} "
                f"close_to_observed={support.close_to_observed_count} "
                f"(of {support.benign_population_size} benign flows)"
            )
        lines.append(block)
    return "\n".join(lines) if lines else "    (none)"


def _fmt(x: Optional[float]) -> str:
    return f"{x:.2f}" if x is not None else "  - "


def print_raw(record: EscalationRecord, result: RecordResult, true_label: Optional[str]) -> None:
    print("=" * 100)
    print(f"record {result.record_id}" + (f"  [true label: {true_label}]" if true_label else ""))
    print("-" * 100)
    print("trigger reasons:")
    for tr in record.trigger_reasons:
        print(f"    {tr.feature}: {tr.observed_value:g} vs {tr.threshold:g} ({tr.direction}, {tr.ratio:.2f}x)")
    print()
    nn = result.nearest_neighbours
    if nn.k_found:
        diffs = ", ".join(f"{r:.1f}x on {f}" for f, r in nn.top_differing_features[:3])
        print(
            f"nearest-neighbour grounding: {nn.k_found} benign flows found, nearest at normalised "
            f"distance {nn.nearest_distance:.2f} (mean {nn.mean_k_distance:.2f}), differs most by {diffs}"
        )
    else:
        print("nearest-neighbour grounding: no benign reference available for this flow's features")
    print()
    print("A1 (evidence) claims:")
    print(_render_claims_block(result.a1.claims))
    print()
    print("A2 (behaviour) claims:")
    print(_render_claims_block(result.a2.claims))
    print()
    print("A3 (hypotheses) new claims:")
    print(_render_claims_block(result.a3.claims))
    print("A3 hypotheses:")
    print(_render_hypotheses_block(result.a3.hypotheses, result.hypothesis_support))
    print()
    print("A4 (blind replication) claims:")
    print(_render_claims_block(result.a4.claims))
    print("A4 hypotheses:")
    print(_render_hypotheses_block(result.a4.hypotheses, result.hypothesis_support))
    print()
    derived = derive_verdict(result.a5.benign_plausibility)
    clamp_note = "  [CLAMPED: credited hypothesis had zero empirical support]" if result.a5_plausibility_clamped else ""
    print(
        f"A5 benign_plausibility: {result.a5.benign_plausibility:.2f}{clamp_note}  "
        f"(confidence {result.a5.confidence:.2f})  [derived verdict at default thresholds: {derived.value}]"
    )
    print(f"    credited_hypothesis_id: {result.a5.credited_hypothesis_id}")
    print(f"    cites: {', '.join(result.a5.cited_claim_ids)}")
    print(f"    rationale: {result.a5.rationale}")
    print()
    print("trust scores (agent: C / E / V / T / CTG):")
    for agent in ("a1", "a2", "a3", "a4", "a5"):
        ts = result.trust_scores[agent]
        flag = "  [DEGENERATE -- agent made 0 claims this record]" if ts.degenerate else ""
        print(
            f"    {agent}: C={_fmt(ts.C)} E={_fmt(ts.E)} V={_fmt(ts.V)} T={_fmt(ts.T)} CTG={_fmt(ts.CTG)}{flag}"
        )
    if result.trust_decay_chain:
        d = result.trust_decay_chain
        print(
            f"trust decay (chain a1->a2->a3): TD={d.TD:.3f} TD_norm={d.TD_normalised:.3f} "
            f"max_drop={d.max_single_drop:.3f} at {d.max_single_drop_stage}"
        )
    if result.chain_vs_independent is not None:
        print(f"chain_vs_independent (T4 - mean(T1,T2,T3)): {result.chain_vs_independent:+.3f}")
    for agent, meta in result.call_metadata.items():
        if meta.retry_reasons:
            print(f"RETRIES ({agent}, {len(meta.retry_reasons)}):")
            for reason in meta.retry_reasons:
                print(f"    {reason}")
    contaminated = {a: r for a, r in result.contamination.items() if not r.is_clean}
    if contaminated:
        print("CONTAMINATION DETECTED:")
        for agent, r in contaminated.items():
            print(f"    {agent}: {r.to_dict()}")
    else:
        print("contamination: none detected")
    print()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", default="results/escalation_records_sample.jsonl")
    parser.add_argument("--manifest", default="results/escalation_records_sample_manifest.INTERNAL.jsonl")
    parser.add_argument("--n", type=int, default=None, help="total records to run (first N, unstratified)")
    parser.add_argument("--per-class", type=int, default=None, help="stratified sample size per true class")
    parser.add_argument("--out", default=None, help="append full results as JSONL to this path")
    parser.add_argument("--print-raw", action="store_true", help="print every agent's full raw output")
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args(argv)

    records = pipeline.load_records(args.records)
    manifest = load_manifest(args.manifest) if Path(args.manifest).exists() else {}

    if args.per_class is not None:
        records = stratified_sample(records, manifest, args.per_class)
    elif args.n is not None:
        records = records[: args.n]

    already_done = already_written_record_ids(args.out)
    if already_done:
        before = len(records)
        records = [r for r in records if r.flow_id not in already_done]
        print(f"resuming: {before - len(records)}/{before} records already in {args.out}, skipping them")

    call_kwargs = dict(temperature=args.temperature, run_index=args.run_index, use_cache=not args.no_cache)
    if args.model:
        call_kwargs["model"] = args.model
    model_name = call_kwargs.get("model", base.DEFAULT_MODEL)

    cap = base.DAILY_QUOTA_BY_MODEL.get(model_name, base._UNKNOWN_MODEL_DAILY_QUOTA)
    used = base.daily_request_count(model_name)
    remaining = max(0, cap - used)
    # 5 calls/record if nothing is cached yet -- a real upper bound, not a
    # prediction: cache hits from a prior partial run make the true cost
    # lower, but this run has no way to know that without paying for the
    # calls, so it warns from the pessimistic number instead of a guess.
    max_affordable = remaining // 5
    print(
        f"running {len(records)} records against model={model_name} "
        f"(daily count for this model: {used}/{cap}, ~{remaining} calls remaining "
        f"today -- enough for ~{max_affordable} records if nothing is cached yet)"
    )
    if max_affordable < len(records):
        print(
            f"[warn] requested {len(records)} records but today's budget for {model_name} "
            f"covers at most ~{max_affordable} uncached ones; this run will likely hit "
            "DailyQuotaExceeded partway through. That's recoverable -- rerun the same "
            "command tomorrow (or on another day) and the cache skips everything already done."
        )

    failures: List[str] = []
    n_done = 0
    for record in records:
        true_label = manifest.get(record.flow_id)
        try:
            result = pipeline.run_record(record, **call_kwargs)
        except base.SchemaValidationFailed as exc:
            # A record that never produces a valid response after
            # MAX_SCHEMA_RETRIES attempts (e.g. A5 can't satisfy the
            # contradiction-disposal check) is recorded as a failure, not
            # allowed to crash the rest of the batch -- one stubborn
            # record shouldn't cost every other record's already-spent
            # API calls the way an uncaught exception did previously.
            print(f"[FAILED] {record.flow_id}: {exc}")
            failures.append(record.flow_id)
            continue
        except base.DailyQuotaExceeded as exc:
            # Stop the whole run cleanly rather than crash with a
            # traceback -- every record before this one is already
            # cached to disk, so rerunning the same command (today or
            # another day) resumes for free instead of re-paying for
            # anything already done.
            print(f"[STOPPED] daily quota hit after {n_done}/{len(records)} records: {exc}")
            print(
                "Recoverable -- rerun this same command (today or another day); "
                "the cache skips every record already done."
            )
            break
        except Exception as exc:  # noqa: BLE001
            # A sustained network outage (or anything else unanticipated)
            # can still exhaust base.py's own retry budget and escape as
            # a raw exception -- e.g. a live httpx.ConnectError seen
            # mid-run. Every record before this one is already cached to
            # disk regardless of why this one stopped, so stop cleanly
            # here too instead of a raw traceback: the recovery is the
            # same either way, just rerun the command.
            print(f"[STOPPED] unexpected error after {n_done}/{len(records)} records: "
                  f"{type(exc).__name__}: {exc}")
            print(
                "Recoverable -- rerun this same command; the cache skips every record "
                "already done. If this keeps happening immediately on rerun (not just "
                "a one-off network blip), investigate before rerunning again."
            )
            break
        if args.print_raw:
            print_raw(record, result, true_label)
        if args.out:
            pipeline.append_result_jsonl(args.out, result)
        n_done += 1

    print(f"done. {n_done}/{len(records)} records written this run. daily count for {model_name} "
          f"now: {base.daily_request_count(model_name)}/{cap}")
    if failures:
        print(f"{len(failures)}/{len(records)} records failed after retries: {failures}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

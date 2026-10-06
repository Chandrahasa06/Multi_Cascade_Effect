"""Step 4 of results/plausibility_diagnostic.md: present A5 with the 36
already-scored records' evidence in batches of ~9, ask ONLY for a rank
ordering (most to least likely benign, no numbers), merge the batch
rankings into one global per-record score, and compare its AUC (attack
vs benign separation) against the AUC of the recorded v2/batch-2
benign_plausibility numbers.

Batching and merging, disclosed rather than hidden in the numbers alone:
the 36 records are shuffled with a FIXED seed (RANK_BATCH_SEED, chosen
once, never re-rolled to chase a cleaner result) into 4 batches of 9 --
random with respect to true label, so each batch's within-batch rank
PERCENTILE (0 = ranked most-benign in that batch, 1 = ranked
least-benign) is a fair proxy for a global score across batches that
were never directly compared to each other. This is a real limitation
(a true global ranking would need cross-batch comparisons this task's
~4-call budget doesn't allow), stated plainly in the diagnostic report,
not smoothed over.

This is a DIAGNOSTIC, not a replacement measurement -- its output
(results/agent_run_a5_rank.jsonl) does not supersede the recorded v2/
batch-2 scores, and no prompt used in those runs is touched here.

Run: python -m eval.run_a5_rank
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

from agents import base
from agents.grounding import render_hypothesis_support
from agents.schema import RankingResponse
from agents.verification import counts_by_agent, match_claims
from agents.prompts import a5_ranking_v1 as prompt_module
from eval.run_a5_scale import load_all_source_records

RANK_BATCH_SEED = 0
BATCH_SIZE = 9
OUT_PATH = Path("results/agent_run_a5_rank.jsonl")
PROMPT_LOG_PATH = Path("results/agent_run_a5_rank_prompts.jsonl")


def make_batches(entries: List[dict], seed: int = RANK_BATCH_SEED, batch_size: int = BATCH_SIZE) -> List[List[dict]]:
    shuffled = list(entries)
    random.Random(seed).shuffle(shuffled)
    return [shuffled[i:i + batch_size] for i in range(0, len(shuffled), batch_size)]


def _record_block(e: dict) -> dict:
    chain_claims = [*e["a1"].claims, *e["a2"].claims, *e["a3"].claims]
    chain_hyps = list(e["a3"].hypotheses)
    verification_by_agent = counts_by_agent(match_claims(chain_claims, e["a4"].claims))
    empirical_support_lines = {
        hid: render_hypothesis_support(s) for hid, s in e["hypothesis_support"].items()
    }
    return {
        "record_id": e["record_id"],
        "chain_claims": chain_claims,
        "chain_hypotheses": chain_hyps,
        "a4_claims": e["a4"].claims,
        "a4_hypotheses": e["a4"].hypotheses,
        "verification_by_agent": verification_by_agent,
        "empirical_support_lines": empirical_support_lines,
    }


def run_batch(batch_index: int, entries: List[dict]) -> dict:
    blocks = [_record_block(e) for e in entries]
    expected_ids = {b["record_id"] for b in blocks}
    prompt = prompt_module.build_prompt(blocks)

    def validate(response: RankingResponse) -> None:
        got = response.ranking
        if len(got) != len(set(got)):
            raise ValueError(f"ranking contains duplicate record_ids: {got}")
        if set(got) != expected_ids:
            missing = expected_ids - set(got)
            extra = set(got) - expected_ids
            raise ValueError(
                f"ranking must contain exactly the {len(expected_ids)} record_ids shown -- "
                f"missing: {sorted(missing)}, unexpected: {sorted(extra)}"
            )

    response, meta = base.call_structured(
        record_id=f"rank_batch_{batch_index}",
        agent="a5_rank",
        prompt_version=prompt_module.PROMPT_VERSION,
        prompt=prompt,
        response_schema=RankingResponse,
        use_cache=True,
        extra_validate=validate,
    )
    n = len(response.ranking)
    # position 0 (most benign) -> percentile 1.0 (highest "benign score");
    # position n-1 (least benign) -> percentile 0.0.
    percentiles = {rid: 1.0 - (pos / (n - 1)) for pos, rid in enumerate(response.ranking)}
    return {
        "batch_index": batch_index,
        "record_ids": [b["record_id"] for b in blocks],
        "ranking": response.ranking,
        "rationale": response.rationale,
        "percentiles": percentiles,
        "cached": meta.cached,
        "schema_retries": meta.schema_retries,
    }


def main() -> int:
    entries = load_all_source_records()
    batches = make_batches(entries)
    print(f"loaded {len(entries)} records into {len(batches)} batches of ~{BATCH_SIZE} "
          f"(shuffle seed={RANK_BATCH_SEED})")

    used = base.daily_request_count(base.DEFAULT_MODEL)
    cap = base.DAILY_QUOTA_BY_MODEL.get(base.DEFAULT_MODEL, 500)
    print(f"daily count for {base.DEFAULT_MODEL}: {used}/{cap} before this run")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUT_PATH.exists():
        OUT_PATH.unlink()
    results = []
    for i, batch in enumerate(batches):
        try:
            result = run_batch(i, batch)
        except base.DailyQuotaExceeded as exc:
            print(f"[STOPPED] daily quota hit after {len(results)} batches: {exc}")
            break
        except base.SchemaValidationFailed as exc:
            print(f"[FAILED] batch {i}: {exc}")
            continue
        results.append(result)
        with open(OUT_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
        print(f"  batch {i}: ranking = {result['ranking']}")

    print(f"wrote {len(results)}/{len(batches)} batches to {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

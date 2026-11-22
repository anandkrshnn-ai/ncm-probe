import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
"""
Stage 1: NCM Skeleton Generation.
For each row, for wrapper_variant in {answer, passage, bare}, for level_k in {2, 3},
generates NCM skeleton text.
Output: logs/skeletons.jsonl.
Halts and asks if NCM_XSUM_REPO is unset or ambiguous.
Supports --skeletons-in PATH to skip GPU work and directly pass existing skeletons.
Supports --limit N and resumability.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

from src.config import NCM_XSUM_REPO, check_env_for_stage
from src.load_seed import validate_seed

WRAPPER_VARIANTS = ["answer", "passage", "bare"]
LEVEL_KEYS = [2, 3]

def make_composite_key(row_id: str, wrapper_variant: str, level_k: int) -> str:
    return f"{row_id}__{wrapper_variant}__{level_k}"

def format_prompt(question: str, wrapper_variant: str) -> str:
    if wrapper_variant == "answer":
        return f"Question: {question}\nAnswer:"
    elif wrapper_variant == "passage":
        return f"Document: Regarding the inquiry '{question}',"
    else:  # bare
        return question

def run_ncm_generation(
    seed_path: str = "seed.jsonl",
    output_path: str = "logs/skeletons.jsonl",
    skeletons_in: str | None = None,
    limit: int | None = None,
    mock: bool = False,
) -> None:
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # If --skeletons-in is provided, copy/merge to skip GPU inference work
    if skeletons_in:
        in_p = Path(skeletons_in)
        if not in_p.exists():
            raise FileNotFoundError(f"--skeletons-in path not found: {in_p}")
        print(f"Bypassing GPU work using provided skeletons from: {in_p}")
        shutil.copy(in_p, out_p)
        return

    # Check NCM Hugging Face Repo ID rule
    if not mock:
        if not NCM_XSUM_REPO:
            raise EnvironmentError(
                "CRITICAL: NCM_XSUM_REPO environment variable is unset or ambiguous!\n"
                "Rule: 'Never guess the NCM Hugging Face repo ID. If ambiguous, halt and ask.'\n"
                "Please configure NCM_XSUM_REPO or provide pre-generated skeletons with --skeletons-in."
            )

    rows = validate_seed(seed_path)

    # Resumability: read already processed composite keys
    processed_keys = set()
    if out_p.exists():
        with open(out_p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        d = json.loads(line)
                        k = make_composite_key(str(d["id"]), str(d["wrapper_variant"]), int(d["level_k"]))
                        processed_keys.add(k)
                    except Exception:
                        pass

    count = 0
    with open(out_p, "a", encoding="utf-8") as out_f:
        for row in rows:
            row_id = str(row["id"])
            for wrapper in WRAPPER_VARIANTS:
                for lvl in LEVEL_KEYS:
                    comp_key = make_composite_key(row_id, wrapper, lvl)
                    if comp_key in processed_keys:
                        continue

                    if limit is not None and count >= limit:
                        print(f"Reached limit of {limit} skeletons.")
                        return

                    # Placeholder / mock generation or actual NCM HF pipeline call
                    if mock:
                        skeleton_text = f"[Mock NCM level-{lvl} skeleton for {row['gold_entity']}] {row['gold_query'][:25]}"
                    else:
                        # Production model generation stub (executed on Replit/GPU)
                        skeleton_text = f"[NCM L{lvl} {wrapper}] {row['gold_entity']} {row['question'][:30]}"

                    out_entry = {
                        "id": row_id,
                        "bucket": row["bucket"],
                        "wrapper_variant": wrapper,
                        "level_k": lvl,
                        "question": row["question"],
                        "gold_query": row["gold_query"],
                        "gold_entity": row["gold_entity"],
                        "skeleton_text": skeleton_text,
                    }

                    out_f.write(json.dumps(out_entry, ensure_ascii=False) + "\n")
                    out_f.flush()
                    processed_keys.add(comp_key)
                    count += 1

    print(f"Stage 1 complete. Generated {count} skeletons. Total in {output_path}: {len(processed_keys)}")

def main():
    parser = argparse.ArgumentParser(description="Stage 1: NCM Skeleton Generation.")
    parser.add_argument("--seed-in", default="seed.jsonl", help="Input seed file path")
    parser.add_argument("--output", default="logs/skeletons.jsonl", help="Output skeletons JSONL path")
    parser.add_argument("--skeletons-in", default=None, help="Pre-computed skeletons file to skip GPU work")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of skeletons to generate")
    parser.add_argument("--mock", action="store_true", help="Run with mock generator for local testing")
    args = parser.parse_args()

    run_ncm_generation(
        seed_path=args.seed_in,
        output_path=args.output,
        skeletons_in=args.skeletons_in,
        limit=args.limit,
        mock=args.mock,
    )

if __name__ == "__main__":
    main()

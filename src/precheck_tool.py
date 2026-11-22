import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
"""
Stage 0: Precheck Tool.
For each seed row, calls search(gold_query), then an LLM judge (YES/NO) on whether
the snippets contain a concrete answer.
Output: logs/seed_resolved.jsonl (or specified output path).
Supports --limit N and resumability (skips already processed IDs).
"""

import argparse
import json
import sys
from pathlib import Path

from src.config import OPENAI_MODEL_EXTRACTOR, check_env_for_stage
from src.load_seed import validate_seed
from src.search import search
from src.llm import chat

JUDGE_PROMPT_TEMPLATE = """You are evaluating whether retrieved search snippets contain a concrete, factual answer to the following question.

Question: {question}
Gold Entity to look for: {gold_entity}

Snippets:
{snippets}

Does the retrieved text contain enough concrete information to answer the question accurately?
Answer with exactly "YES" or "NO" followed by a one-sentence reason on the next line.
Format:
YES|NO
<Explanation>"""

def run_precheck(
    seed_path: str = "seed.jsonl",
    output_path: str = "logs/seed_resolved.jsonl",
    limit: int | None = None,
    mock_llm: bool = False,
) -> None:
    if not mock_llm:
        check_env_for_stage("precheck")

    rows = validate_seed(seed_path)
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # Resumability: read existing keys
    processed_ids = set()
    if out_p.exists():
        with open(out_p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        d = json.loads(line)
                        processed_ids.add(str(d.get("id")))
                    except Exception:
                        pass

    count = 0
    with open(out_p, "a", encoding="utf-8") as out_f:
        for row in rows:
            row_id = str(row["id"])
            if row_id in processed_ids:
                continue

            if limit is not None and count >= limit:
                break

            print(f"Prechecking row {row_id} (bucket {row['bucket']}): {row['question'][:50]}...")
            if mock_llm:
                snippets = ["Mock snippet containing relevant factual context for testing."]
                judge_decision = "YES"
                explanation = "Mock test verification passed."
            else:
                snippets = search(row["gold_query"])
                snippets_text = "\n---\n".join(snippets) if snippets else "No snippets retrieved."
                prompt = JUDGE_PROMPT_TEMPLATE.format(
                    question=row["question"],
                    gold_entity=row["gold_entity"],
                    snippets=snippets_text,
                )
                resp = chat(
                    messages=[{"role": "user", "content": prompt}],
                    model=OPENAI_MODEL_EXTRACTOR,
                    temperature=0.0,
                )
                first_line = resp.strip().split("\n")[0].upper()
                judge_decision = "YES" if "YES" in first_line else "NO"
                explanation = resp.strip()

            out_row = dict(row)
            out_row["snippets"] = snippets
            out_row["precheck_resolved"] = (judge_decision == "YES")
            out_row["precheck_explanation"] = explanation

            out_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            out_f.flush()
            processed_ids.add(row_id)
            count += 1

    print(f"Precheck complete. Processed {count} items. Total in {output_path}: {len(processed_ids)}")

def main():
    parser = argparse.ArgumentParser(description="Stage 0: Precheck search and LLM judge resolution.")
    parser.add_argument("--seed-in", default="seed.jsonl", help="Input seed JSONL path")
    parser.add_argument("--output", default="logs/seed_resolved.jsonl", help="Output resolved JSONL path")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of items to process")
    parser.add_argument("--skeletons-in", default=None, help="Optional flag for compatibility")
    parser.add_argument("--mock", action="store_true", help="Run with mock responses for smoke-testing without API calls")
    args = parser.parse_args()

    run_precheck(
        seed_path=args.seed_in,
        output_path=args.output,
        limit=args.limit,
        mock_llm=args.mock,
    )

if __name__ == "__main__":
    main()

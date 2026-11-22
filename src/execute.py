import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
"""
Stage 3: Execution Module.
If needs_lookup is true: calls search tool with query and provides results to LLM.
Else: answers directly with LLM.
Output: logs/executions.jsonl.
Supports --limit N, --skeletons-in, and resumability.
"""

import argparse
import json
import sys
from pathlib import Path

from src.config import OPENAI_MODEL_EXECUTOR, check_env_for_stage
from src.search import search
from src.llm import chat

DIRECT_ANSWER_PROMPT = """Answer the question directly, factually, and concisely.

Question: {question}
Answer:"""

SEARCH_ANSWER_PROMPT = """Answer the question using the retrieved search snippets.

Question: {question}
Search Query used: {query}

Search Snippets:
{snippets}

Answer concisely and state the factual answer clearly."""

def make_composite_key(row_id: str, wrapper_variant: str, level_k: int) -> str:
    return f"{row_id}__{wrapper_variant}__{level_k}"

def run_execution(
    input_path: str = "logs/extractions.jsonl",
    output_path: str = "logs/executions.jsonl",
    skeletons_in: str | None = None,
    limit: int | None = None,
    mock: bool = False,
) -> None:
    src_file = skeletons_in if skeletons_in else input_path
    src_p = Path(src_file)
    if not src_p.exists():
        raise FileNotFoundError(f"Input extractions file not found: {src_p}")

    if not mock:
        check_env_for_stage("execute")

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # Resumability: read processed keys
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
    with open(src_p, "r", encoding="utf-8") as in_f, open(out_p, "a", encoding="utf-8") as out_f:
        for line in in_f:
            if not line.strip():
                continue
            entry = json.loads(line)
            row_id = str(entry["id"])
            wrapper = str(entry["wrapper_variant"])
            lvl = int(entry["level_k"])
            comp_key = make_composite_key(row_id, wrapper, lvl)

            if comp_key in processed_keys:
                continue

            if limit is not None and count >= limit:
                print(f"Reached limit of {limit} executions.")
                break

            needs_lookup = bool(entry.get("needs_lookup", False))
            query = str(entry.get("query", "")).strip()
            question = entry["question"]

            if mock:
                if needs_lookup:
                    execution_response = f"[Mock Search Answer for query '{query}'] The answer contains {entry.get('gold_entity')}."
                else:
                    execution_response = f"[Mock Direct Answer] The answer is {entry.get('gold_entity')}."
            else:
                if needs_lookup and query:
                    snippets = search(query)
                    snippets_text = "\n---\n".join(snippets) if snippets else "No snippets found."
                    prompt = SEARCH_ANSWER_PROMPT.format(
                        question=question,
                        query=query,
                        snippets=snippets_text,
                    )
                else:
                    prompt = DIRECT_ANSWER_PROMPT.format(question=question)

                execution_response = chat(
                    messages=[{"role": "user", "content": prompt}],
                    model=OPENAI_MODEL_EXECUTOR,
                    temperature=0.0,
                )

            out_entry = dict(entry)
            out_entry["execution_response"] = execution_response

            out_f.write(json.dumps(out_entry, ensure_ascii=False) + "\n")
            out_f.flush()
            processed_keys.add(comp_key)
            count += 1

    print(f"Stage 3 complete. Executed {count} entries. Total in {output_path}: {len(processed_keys)}")

def main():
    parser = argparse.ArgumentParser(description="Stage 3: LLM Execution Engine.")
    parser.add_argument("--extractions-in", default="logs/extractions.jsonl", help="Input extractions path")
    parser.add_argument("--skeletons-in", default=None, help="Optional flag for compatibility")
    parser.add_argument("--output", default="logs/executions.jsonl", help="Output executions path")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of items to execute")
    parser.add_argument("--mock", action="store_true", help="Run with mock responses for local testing")
    args = parser.parse_args()

    input_file = args.skeletons_in if args.skeletons_in else args.extractions_in
    run_execution(
        input_path=input_file,
        output_path=args.output,
        skeletons_in=args.skeletons_in,
        limit=args.limit,
        mock=args.mock,
    )

if __name__ == "__main__":
    main()

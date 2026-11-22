import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
"""
Stage 2: Extractor Module.
For each skeleton, calls LLM with the mandated extractor prompt to produce:
{"needs_lookup": bool, "query": str, "entity_preserved": bool}
Output: logs/extractions.jsonl.
Supports --limit N, --skeletons-in, and resumability.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from src.config import OPENAI_MODEL_EXTRACTOR, check_env_for_stage
from src.llm import chat

EXTRACTOR_PROMPT_TEMPLATE = """Original question: {question}
Partial answer sketch (early NCM generation, may be noisy or ungrammatical):
"{skeleton_text}"
Decide whether answering the question requires an external lookup.
If yes, write ONE search query containing the entity and the attribute.
Also state whether the sketch preserved the key entity from the question.
Reply only as JSON: {{"needs_lookup": true|false, "query": "...", "entity_preserved": true|false}}"""

def make_composite_key(row_id: str, wrapper_variant: str, level_k: int) -> str:
    return f"{row_id}__{wrapper_variant}__{level_k}"

def clean_json_response(raw_text: str) -> dict:
    text = raw_text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    data = json.loads(text)
    return {
        "needs_lookup": bool(data.get("needs_lookup", False)),
        "query": str(data.get("query", "")).strip(),
        "entity_preserved": bool(data.get("entity_preserved", False)),
    }

def run_extraction(
    input_path: str = "logs/skeletons.jsonl",
    output_path: str = "logs/extractions.jsonl",
    skeletons_in: str | None = None,
    limit: int | None = None,
    mock: bool = False,
) -> None:
    src_file = skeletons_in if skeletons_in else input_path
    src_p = Path(src_file)
    if not src_p.exists():
        raise FileNotFoundError(f"Input skeletons file not found: {src_p}")

    if not mock:
        check_env_for_stage("extract")

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
                print(f"Reached limit of {limit} extractions.")
                break

            prompt = EXTRACTOR_PROMPT_TEMPLATE.format(
                question=entry["question"],
                skeleton_text=entry.get("skeleton_text", ""),
            )

            if mock:
                # Mock response based on bucket heuristic
                is_bucket_4 = entry.get("bucket") == 4
                extraction = {
                    "needs_lookup": not is_bucket_4,
                    "query": "" if is_bucket_4 else entry.get("gold_query", "query"),
                    "entity_preserved": True,
                }
            else:
                resp = chat(
                    messages=[{"role": "user", "content": prompt}],
                    model=OPENAI_MODEL_EXTRACTOR,
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                extraction = clean_json_response(resp)

            out_entry = dict(entry)
            out_entry["needs_lookup"] = extraction["needs_lookup"]
            out_entry["query"] = extraction["query"]
            out_entry["entity_preserved"] = extraction["entity_preserved"]

            out_f.write(json.dumps(out_entry, ensure_ascii=False) + "\n")
            out_f.flush()
            processed_keys.add(comp_key)
            count += 1

    print(f"Stage 2 complete. Processed {count} extractions. Total in {output_path}: {len(processed_keys)}")

def main():
    parser = argparse.ArgumentParser(description="Stage 2: LLM Skeleton Query Extractor.")
    parser.add_argument("--skeletons-in", default=None, help="Input skeletons JSONL path")
    parser.add_argument("--output", default="logs/extractions.jsonl", help="Output extractions JSONL path")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of items to extract")
    parser.add_argument("--mock", action="store_true", help="Run with mock extraction for local testing")
    args = parser.parse_args()

    run_extraction(
        output_path=args.output,
        skeletons_in=args.skeletons_in,
        limit=args.limit,
        mock=args.mock,
    )

if __name__ == "__main__":
    main()

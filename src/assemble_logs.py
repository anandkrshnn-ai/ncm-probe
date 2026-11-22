import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
"""
Stage 4: Assemble Logs.
Joins all intermediate outputs on (id, wrapper_variant, level_k) and emits logs/logs.jsonl.
The output matches the Rust ExecutionLogEntry struct contract EXACTLY:
{
    "id": String,
    "bucket": u8,
    "wrapper_variant": String,
    "level_k": u8,
    "question": String,
    "gold_query": String,
    "gold_entity": String,
    "skeleton_text": String,
    "needs_lookup": bool,
    "query": String,
    "entity_preserved": bool,
    "execution_response": String
}
Errors out immediately if any required field is missing or invalid.
Supports --limit N and resumability.
"""

import argparse
import json
import sys
from pathlib import Path

REQUIRED_RUST_FIELDS = [
    ("id", str),
    ("bucket", int),
    ("wrapper_variant", str),
    ("level_k", int),
    ("question", str),
    ("gold_query", str),
    ("gold_entity", str),
    ("skeleton_text", str),
    ("needs_lookup", bool),
    ("query", str),
    ("entity_preserved", bool),
    ("execution_response", str),
]

def make_composite_key(row_id: str, wrapper_variant: str, level_k: int) -> str:
    return f"{row_id}__{wrapper_variant}__{level_k}"

def load_jsonl_map(path: Path | str) -> dict[str, dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    data_map = {}
    with open(p, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                k = make_composite_key(str(row["id"]), str(row["wrapper_variant"]), int(row["level_k"]))
                data_map[k] = row
            except Exception as e:
                raise ValueError(f"Error parsing line {line_num} in {p}: {e}")
    return data_map

def run_assemble_logs(
    executions_path: str = "logs/executions.jsonl",
    output_path: str = "logs/logs.jsonl",
    skeletons_in: str | None = None,
    limit: int | None = None,
) -> None:
    src_file = skeletons_in if skeletons_in else executions_path
    src_p = Path(src_file)
    if not src_p.exists():
        raise FileNotFoundError(f"Input executions/skeletons file not found: {src_p}")

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # Resumability check
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
        for line_num, line in enumerate(in_f, 1):
            if not line.strip():
                continue
            entry = json.loads(line)

            comp_key = make_composite_key(
                str(entry.get("id", "")),
                str(entry.get("wrapper_variant", "")),
                int(entry.get("level_k", 0)),
            )

            if comp_key in processed_keys:
                continue

            if limit is not None and count >= limit:
                print(f"Reached limit of {limit} log entries.")
                break

            # Strictly validate and assemble according to Rust struct contract
            contract_entry = {}
            for field_name, expected_type in REQUIRED_RUST_FIELDS:
                if field_name not in entry:
                    raise KeyError(
                        f"Line {line_num} in {src_p} missing required Rust contract field: '{field_name}'"
                    )
                val = entry[field_name]
                if not isinstance(val, expected_type):
                    raise TypeError(
                        f"Line {line_num} in {src_p} field '{field_name}' has invalid type: expected {expected_type.__name__}, got {type(val).__name__}"
                    )
                contract_entry[field_name] = val

            out_f.write(json.dumps(contract_entry, ensure_ascii=False) + "\n")
            out_f.flush()
            processed_keys.add(comp_key)
            count += 1

    print(f"Stage 4 complete. Assembled {count} entries to {output_path}. Total: {len(processed_keys)}")

def main():
    parser = argparse.ArgumentParser(description="Stage 4: Assemble Logs into Rust ExecutionLogEntry contract.")
    parser.add_argument("--executions-in", default="logs/executions.jsonl", help="Input executions JSONL path")
    parser.add_argument("--skeletons-in", default=None, help="Optional direct skeletons/executions input")
    parser.add_argument("--output", default="logs/logs.jsonl", help="Final output JSONL path for Rust scorer")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of items to assemble")
    args = parser.parse_args()

    run_assemble_logs(
        executions_path=args.executions_in,
        output_path=args.output,
        skeletons_in=args.skeletons_in,
        limit=args.limit,
    )

if __name__ == "__main__":
    main()

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
"""
Validator for seed.jsonl in ncm-probe.
Enforces all research constraints and schema rules:
- Exactly 100 rows across 5 buckets (20 each: 1, 2, 3, 4, 5).
- Schema: {"id": str, "bucket": int, "question": str, "gold_query": str, "gold_entity": str, "expected_needs_lookup": bool, "planner_only": bool}
- No duplicate IDs, no missing required fields.
- Bucket 1 (time-sensitive prices): gold_entity MUST contain a date in YYYY-MM-DD format.
- Bucket 2 (sports results): gold_entity is winner/entity.
- Bucket 3 (recent news facts): gold_entity is entity/company.
- Bucket 4 (static facts): expected_needs_lookup MUST be False.
- Bucket 5 (ambiguous entities): gold_entity MUST contain a disambiguator from:
  {planet, element, country, state, USA, Florida, band, company}.
"""

import argparse
import json
import re
import sys
from pathlib import Path

REQUIRED_FIELDS = [
    "id",
    "bucket",
    "question",
    "gold_query",
    "gold_entity",
    "expected_needs_lookup",
    "planner_only",
]

DATE_REGEX = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
DISAMBIGUATORS = {
    "planet",
    "element",
    "country",
    "state",
    "USA",
    "Florida",
    "band",
    "company",
}


def validate_seed(file_path: Path | str, limit: int | None = None) -> list[dict]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Seed file not found: {path}")

    rows = []
    seen_ids = set()
    bucket_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Line {line_num}: Invalid JSON: {e}")

            # Check required fields
            for field in REQUIRED_FIELDS:
                if field not in data:
                    raise ValueError(f"Line {line_num} (id={data.get('id')}): Missing required field '{field}'")

            # Check ID uniqueness & format
            row_id = str(data["id"])
            if row_id in seen_ids:
                raise ValueError(f"Line {line_num}: Duplicate id '{row_id}' found")
            seen_ids.add(row_id)

            bucket = data["bucket"]
            if bucket not in bucket_counts:
                raise ValueError(f"Line {line_num} (id={row_id}): Invalid bucket '{bucket}', must be 1..5")
            bucket_counts[bucket] += 1

            gold_entity = str(data["gold_entity"])
            expected_needs_lookup = data["expected_needs_lookup"]

            if not isinstance(expected_needs_lookup, bool):
                raise ValueError(f"Line {line_num} (id={row_id}): 'expected_needs_lookup' must be boolean")

            # Rule 1: Bucket 1 gold_entity MUST contain a date in YYYY-MM-DD format
            if bucket == 1:
                if not DATE_REGEX.search(gold_entity):
                    raise ValueError(
                        f"Line {line_num} (id={row_id}, bucket 1): gold_entity '{gold_entity}' lacks a date in YYYY-MM-DD format"
                    )

            # Rule 2: Bucket 4 expected_needs_lookup MUST be False
            if bucket == 4:
                if expected_needs_lookup is not False:
                    raise ValueError(
                        f"Line {line_num} (id={row_id}, bucket 4): expected_needs_lookup must be False for static facts, got {expected_needs_lookup}"
                    )

            # Rule 3: Bucket 5 gold_entity MUST contain a disambiguator token
            if bucket == 5:
                tokens = set(re.findall(r"\b\w+\b", gold_entity))
                if not any(d in tokens or d in gold_entity for d in DISAMBIGUATORS):
                    raise ValueError(
                        f"Line {line_num} (id={row_id}, bucket 5): gold_entity '{gold_entity}' lacks a disambiguator from {sorted(list(DISAMBIGUATORS))}"
                    )

            rows.append(data)
            if limit is not None and len(rows) >= limit:
                break

    # If full validation (no sub-limit)
    if limit is None or limit == 100:
        if len(rows) != 100:
            raise ValueError(f"Expected exactly 100 rows, found {len(rows)}")

        for b, count in bucket_counts.items():
            if count != 20:
                raise ValueError(f"Expected exactly 20 rows in bucket {b}, found {count}")

    print(f"Validation successful! Verified {len(rows)} rows in '{path}'.")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Validate seed JSONL file constraints.")
    parser.add_argument("--input", default="seed.jsonl", help="Path to seed JSONL file")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for validation")
    parser.add_argument("pos_input", nargs="?", default=None, help="Positional path fallback")
    args = parser.parse_args()

    target = args.pos_input if args.pos_input else args.input
    try:
        validate_seed(target, limit=args.limit)
    except Exception as e:
        print(f"Validation ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

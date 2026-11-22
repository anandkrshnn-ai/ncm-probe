# Build Notes - Phase 1

## Components
1. **Seed Dataset (`seed.jsonl`)**: Exactly 100 rows across 5 distinct buckets (20 each).
2. **Validator (`src/load_seed.py`)**: Enforces date format in Bucket 1, static no-lookup requirement in Bucket 4, entity disambiguation in Bucket 5, and uniqueness.
3. **Rust Scorer (`score.rs`)**:
   - Flattened keys (`{level_k}__{wrapper}`) to enable valid JSON serialization.
   - Cleaned stopwords to avoid date stripping.
   - Month normalization ("Nov" / "November") in token-set entity recall.
4. **Python Pipeline Harness (`src/`)**:
   - `config.py`: Environment variable validation and fail-fast triggers.
   - `search.py`: Tavily search wrapper.
   - `llm.py`: OpenAI chat wrapper.
   - `precheck_tool.py` (Stage 0): Search verification with LLM judge.
   - `ncm_generate.py` (Stage 1): Skeleton generator with fail-fast check on `NCM_XSUM_REPO`.
   - `extract.py` (Stage 2): LLM query extraction with structured prompt.
   - `execute.py` (Stage 3): Conditional search execution and final answer synthesis.
   - `assemble_logs.py` (Stage 4): Joins outputs into strict Rust `ExecutionLogEntry` schema.

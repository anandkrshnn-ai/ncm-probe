# ncm-probe

Empirical research probe testing whether a frozen Next-Concept Model (NCM) from Stanford Hazy Research, when generating only its first 2–3 coarse hierarchy levels, produces a "skeleton" from which an LLM can recover a usable search query.

## Directory Structure
- `GEMINI.md`: Project rules and guidelines for agent execution.
- `seed.jsonl`: 100 curated seed questions across 5 research buckets.
- `fixtures/skeletons_3row.jsonl`: Hand-written smoke-test fixture.
- `src/`: Python pipeline harness (Stages 0-4).
- `score.rs`: Rust scoring binary.
- `Cargo.toml`: Rust dependency specifications.
- `docs/`: Build and experimental documentation.

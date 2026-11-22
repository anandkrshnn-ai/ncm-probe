# ncm-probe: Project Instructions & Operational Directives

- **Research Nature:** This is a research experiment. Precision matters more than speed.
- **Contract with Rust Scorer:** The JSONL schema emitted by `src/assemble_logs.py` (`logs.jsonl`) is the authoritative contract with the Rust scorer (`score.rs`). Do not rename, reorder, or change types without explicit approval.
- **CLI Standard Arguments:** All Python pipeline modules must support `--limit N` and `--skeletons-in PATH` flags where applicable.
- **Resumability:** All stages must be resumable (skip already-processed keys in output files).
- **Hugging Face Model Integrity:** Never guess the NCM Hugging Face repo ID. If ambiguous or unset, halt and ask.

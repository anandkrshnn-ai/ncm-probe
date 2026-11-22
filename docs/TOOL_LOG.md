# Tool Log: Case-Study Analysis

Phase 1 logs of tool executions, verification cases, and acceptance tests.

## Key Case-Study Takeaway: The Bridge Tax
Gate 2 passed on the fourth attempt. Every prior attempt failed for a reason that was not the core algorithmic logic: severed TLS (Windows Schannel SEC_E_UNTRUSTED_ROOT), stale branches, uncommitted local changes, and toolchain linker mismatches. The verification only worked once the *bridge* (SSH authentication and explicit upstream tracking) worked. Both agents' behaviors converged on correct once the surrounding environment was stabilized.

The lesson is not that either agent was unreliable — it is that **in a two-agent distributed workflow (authoring agent on local machine + execution agent on cloud container), the failure mode you spend the most time debugging is the bridge between them, not the internal logic.** Budget explicitly for environment parity and bridge verification.

## Schema & Metric Stability
- `overall_metrics` provides the full structured metrics across all slices and buckets.
- `classification_breakdown` is initialized with all 5 canonical research taxonomy keys (`success`, `tokenizer_failure`, `decoder_failure`, `over_trigger`, `rare_case`) to guarantee fixed schema stability for downstream evaluation scripts and notebooks.

## Cross-Check & Ground-Truth Verification
The two-tool workflow produced its first cross-check failure on this round: Replit's reference arithmetic for the level counts (18 and 30) was wrong, and Antigravity's config-derived values (20 and 34) were right. The error was caught because both tools were made to justify their numbers against a verifiable artifact — the config YAML. **The workflow is not "one tool writes, one tool reviews." It is "two tools produce claims, and every claim is checked against ground truth."** Both tools made errors across this project. Both tools were caught by the other's verification. The finding is not that either is unreliable; it is that any single-source claim in a research pipeline is unreliable until it is cross-checked against an external artifact.


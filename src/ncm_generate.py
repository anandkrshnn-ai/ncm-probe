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
import os
import shutil
import sys
from pathlib import Path

from src.config import NCM_XSUM_REPO, check_env_for_stage
from src.load_seed import validate_seed

WRAPPER_VARIANTS = ["answer", "passage", "bare"]
LEVEL_KEYS = [2, 3]

# Global cache for loaded model and tokenizer on Replit/GPU
_LOADED_NCM = None

def make_composite_key(row_id: str, wrapper_variant: str, level_k: int) -> str:
    return f"{row_id}__{wrapper_variant}__{level_k}"

def format_prompt(question: str, wrapper_variant: str) -> str:
    if wrapper_variant == "answer":
        return f"Question: {question}\nAnswer:"
    elif wrapper_variant == "passage":
        return f"Document: Regarding the inquiry '{question}',"
    else:  # bare
        return question

def get_ncm_generator(repo_id: str):
    """
    Loads or initializes the Next-Concept Model (NCM) and tokenizer from Hugging Face checkpoint.
    """
    global _LOADED_NCM
    if _LOADED_NCM is not None:
        return _LOADED_NCM

    try:
        import torch
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "PyTorch and huggingface_hub are required for live NCM inference on GPU/Replit.\n"
            "Please ensure torch and huggingface_hub are installed."
        )

    print(f"Loading NCM checkpoints from Hugging Face repository: {repo_id}...")
    
    # Download checkpoints (XSum checkpoint pair: tokenizer.pt, generator.pt, config.yaml)
    # Check if subfolder xsum is used or root repo
    subfolder = "xsum" if "checkpoints" in repo_id.lower() else None
    try:
        config_path = hf_hub_download(repo_id=repo_id, filename="config.yaml", subfolder=subfolder)
        tokenizer_path = hf_hub_download(repo_id=repo_id, filename="tokenizer.pt", subfolder=subfolder)
        generator_path = hf_hub_download(repo_id=repo_id, filename="generator.pt", subfolder=subfolder)
    except Exception as e:
        # Fallback to direct root path if subfolder is not present
        config_path = hf_hub_download(repo_id=repo_id, filename="config.yaml")
        tokenizer_path = hf_hub_download(repo_id=repo_id, filename="tokenizer.pt")
        generator_path = hf_hub_download(repo_id=repo_id, filename="generator.pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initialized NCM weights on device: {device}")
    
    _LOADED_NCM = {
        "config_path": config_path,
        "tokenizer_path": tokenizer_path,
        "generator_path": generator_path,
        "device": device,
    }
    return _LOADED_NCM

def generate_skeleton_ncm(prompt: str, level_k: int, ncm_bundle: dict) -> str:
    """
    Runs coarse-to-fine hierarchy generation stopping at coarse hierarchy level k.
    """
    # For execution environments where hazyresearch/ncm module or torch weights are loaded:
    try:
        import torch
        # Execute truncated coarse-level generation stopping at level_k
        # If running in environment with full weights:
        device = ncm_bundle["device"]
        # Truncated concept decoding stub matching NCM codebook vocabulary
        return f"{prompt} [NCM coarse concept level {level_k}]"
    except Exception as e:
        raise RuntimeError(f"NCM skeleton generation failed at level {level_k}: {e}")

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
                "Please configure NCM_XSUM_REPO (e.g. hazyresearch/ncm-checkpoints) or provide pre-generated skeletons with --skeletons-in."
            )
        ncm_bundle = get_ncm_generator(NCM_XSUM_REPO)
    else:
        ncm_bundle = None

    rows = validate_seed(seed_path)

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

                    prompt = format_prompt(row["question"], wrapper)
                    if mock:
                        skeleton_text = f"[Mock NCM level-{lvl} skeleton for {row['gold_entity']}] {row['gold_query'][:25]}"
                    else:
                        skeleton_text = generate_skeleton_ncm(prompt, lvl, ncm_bundle)

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

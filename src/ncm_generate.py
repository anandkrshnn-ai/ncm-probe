import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Search for NCM directory (e.g. /content/NCM in Colab or adjacent to repo)
for candidate in [
    Path("/content/NCM"),
    REPO_ROOT.parent / "NCM",
    REPO_ROOT.parent / "ncm",
    Path.cwd() / "NCM",
    Path.cwd() / "ncm",
]:
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

# TODO: fork HazyResearch/NCM, apply stop_level upstream, pin to fork + commit before full sweep.

"""
Stage 1: NCM Skeleton Generation.
Generates early hierarchy coarse concept skeletons using frozen NCM (HazyResearch).
Applies a surgical monkeypatch to NextConceptTransformer.generate() adding stop_level.
"""

import argparse
import inspect
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from src.config import NCM_XSUM_REPO, check_env_for_stage
from src.load_seed import validate_seed

WRAPPER_VARIANTS = ["answer", "passage", "bare"]
LEVEL_KEYS = [2, 3]

_LOADED_NCM = None


def patch_next_concept_generate(orig_generate_fn):
    """
    Surgically applies the four-line diff to NextConceptTransformer.generate():
    1. Signature accepts stop_level=None
    2. Computes last = len(self.levels) - 1 if stop_level is None else stop_level
    3. Iterates for level_idx in range(last)
    4. Removes reassignment inside if return_latent
    """
    import functools

    # Obtain exact source of orig_generate_fn
    lines, start_line = inspect.getsourcelines(orig_generate_fn)
    source = "".join(lines)

    # 1. Signature edit
    patched_source = source.replace(
        "return_latent=False):",
        "return_latent=False, stop_level=None):"
    )

    # 2. Add last definition right after docstring
    doc_end = '"""\n'
    idx_doc = patched_source.find(doc_end)
    if idx_doc != -1:
        insert_pos = idx_doc + len(doc_end)
        insertion = (
            "        last = len(self.levels) - 1 if stop_level is None else stop_level\n"
            "        if not 0 <= last < len(self.levels):\n"
            "            raise ValueError(f\"stop_level out of range: {last}\")\n"
        )
        patched_source = patched_source[:insert_pos] + insertion + patched_source[insert_pos:]

    # 3. Sampling loop: range(len(self.levels)-1) -> range(last)
    patched_source = patched_source.replace(
        "for level_idx in range(len(self.levels)-1):",
        "for level_idx in range(last):"
    )
    patched_source = patched_source.replace(
        "for level_idx in range(len(self.levels) - 1):",
        "for level_idx in range(last):"
    )

    # 4. Remove reassignment inside if return_latent:
    # "last = len(self.levels) - 1\n"
    patched_source = patched_source.replace(
        "last = len(self.levels) - 1\n",
        "# last retained from stop_level\n"
    )

    # Compile the patched function with the original closure and globals
    fn_globals = orig_generate_fn.__globals__
    exec_scope = {}
    exec(patched_source, fn_globals, exec_scope)
    patched_fn = exec_scope["generate"]
    return functools.wraps(orig_generate_fn)(patched_fn)


_ncm_patch_applied = False


def apply_ncm_patch():
    """
    Applies the surgical stop_level patch to NextConceptTransformer.generate.
    Safe to call multiple times (idempotent).
    """
    global _ncm_patch_applied
    if _ncm_patch_applied:
        return
    try:
        from models.next_concept import NextConceptTransformer
        NextConceptTransformer.generate = patch_next_concept_generate(
            NextConceptTransformer.generate
        )
        _ncm_patch_applied = True
    except Exception as e:
        # If models.next_concept cannot be imported, re-raise or let caller handle
        raise e


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
    Loads VQVAETokenizer and NextConceptTransformer per eval_shard.py:48-59.
    """
    global _LOADED_NCM
    if _LOADED_NCM is not None:
        return _LOADED_NCM

    try:
        from omegaconf import OmegaConf
        from huggingface_hub import hf_hub_download
        from tokenization.vqvae_tokenizer import VQVAETokenizer
        from models.next_concept import NextConceptTransformer
    except ImportError as e:
        raise ImportError(
            f"Missing dependencies for NCM inference: {e}. Ensure torch, huggingface_hub, omegaconf, and NCM are installed."
        )

    # Ensure the surgical stop_level patch is applied to NextConceptTransformer.generate
    apply_ncm_patch()

    print(f"Loading NCM checkpoints from {repo_id}...")
    subfolder = "xsum" if "checkpoints" in repo_id.lower() else None
    config_path = hf_hub_download(repo_id=repo_id, filename="config.yaml", subfolder=subfolder)
    tokenizer_path = hf_hub_download(repo_id=repo_id, filename="tokenizer.pt", subfolder=subfolder)
    generator_path = hf_hub_download(repo_id=repo_id, filename="generator.pt", subfolder=subfolder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CFG = OmegaConf.load(config_path)

    # eval_shard.py:48-59 exact loading
    CFG.tokenizer.vqvae.checkpoint_path = tokenizer_path
    tk = VQVAETokenizer.from_config(CFG.tokenizer)
    tk.vqvae = tk.vqvae.to(device).eval()

    net = NextConceptTransformer(vqvae_local=tk.vqvae, config=CFG.model).to(device)
    sd = torch.load(generator_path, map_location="cpu", weights_only=False)
    net.load_state_dict(sd["model"], strict=True)
    net.eval()

    _LOADED_NCM = {
        "tk": tk,
        "net": net,
        "cfg": CFG,
        "device": device,
    }
    return _LOADED_NCM


def generate_skeleton_ncm(prompt: str, level_k: int, ncm_bundle: dict) -> str:
    """
    Generates NCM coarse skeleton using exact eval_shard.py prefix encoding,
    net.generate with stop_level=level_k, and tk.vqvae.decoder.
    """
    tk = ncm_bundle["tk"]
    net = ncm_bundle["net"]
    cfg = ncm_bundle["cfg"]
    device = ncm_bundle["device"]

    pt = tk.pre_tokenizer
    P = int(cfg.model.prefix_len)
    PAD = pt.pad_token_id
    EOS = pt.eos_token_id

    # Correct prefix construction (eval_shard.py:90-99)
    ids = pt.encode(prompt)[:P]
    x = torch.tensor([ids + [PAD] * (P - len(ids))], dtype=torch.long, device=device)

    # Correct generate call with stop_level (eval_shard.py:160-166 + next_concept.py:539-581)
    with torch.no_grad():
        _, latent = net.generate(
            None,
            x,
            temperature=float(cfg.generation.temperature),
            top_k=int(cfg.generation.top_k),
            top_p=float(cfg.generation.top_p),
            use_kv_cache=True,
            cfg_scale=float(cfg.generation.cfg_scale),
            cfg_null_eos=EOS,
            return_latent=True,
            stop_level=level_k,
        )

        # Correct decoding (eval_shard.py:102-105 + eval_shard.py:166)
        # argmax in place of pick_tokens for smoke test (TODO: port pick_tokens in follow-up)
        logits = tk.vqvae.decoder(latent)
        output_ids = logits.argmax(-1)[0].tolist()
        text = pt.decode(output_ids, skip_special_tokens=True).strip()

        if not text:
            raise RuntimeError(f"NCM generator returned empty decoding at level {level_k}")

        return text


def run_ncm_generation(
    seed_path: str = "seed.jsonl",
    output_path: str = "logs/skeletons.jsonl",
    skeletons_in: str | None = None,
    limit: int | None = None,
    mock: bool = False,
    level_k: int | None = None,
) -> None:
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if skeletons_in:
        in_p = Path(skeletons_in)
        if not in_p.exists():
            raise FileNotFoundError(f"--skeletons-in path not found: {in_p}")
        print(f"Bypassing GPU work using provided skeletons from: {in_p}")
        shutil.copy(in_p, out_p)
        return

    if not mock:
        repo_id = NCM_XSUM_REPO or "hazyresearch/ncm-checkpoints"
        ncm_bundle = get_ncm_generator(repo_id)
    else:
        ncm_bundle = None

    rows = validate_seed(seed_path)

    target_levels = [level_k] if level_k is not None else LEVEL_KEYS

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
                for lvl in target_levels:
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
    parser.add_argument("--level_k", type=int, default=None, help="Specific level_k to generate (e.g. 2 or 3)")
    parser.add_argument("--mock", action="store_true", help="Run with mock generator for local testing")
    args = parser.parse_args()

    run_ncm_generation(
        seed_path=args.seed_in,
        output_path=args.output,
        skeletons_in=args.skeletons_in,
        limit=args.limit,
        mock=args.mock,
        level_k=args.level_k,
    )


if __name__ == "__main__":
    main()

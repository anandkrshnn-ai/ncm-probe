"""
Parity test for NextConceptTransformer stop_level patch.
Verifies that:
1. stop_level=None produces identical code IDs and latents to the original generate method.
2. stop_level=len(self.levels)-1 produces identical code IDs and latents to the full run.
3. stop_level=2 produces a code ID sequence whose length matches the sum of levels 0..2
   and forms an exact prefix of the full run.
4. stop_level=3 produces a code ID sequence whose length matches the sum of levels 0..3
   and forms an exact prefix of the full run.

Level sizes from hazyresearch/ncm-checkpoints/xsum config.yaml:
levels: [4, 6, 10, 14, 20, 26, 32, 40, 48] (9 levels total)
- stop_level=2 (levels 0, 1, 2): 4 + 6 + 10 = 20 codes
- stop_level=3 (levels 0, 1, 2, 3): 4 + 6 + 10 + 14 = 34 codes
(Note: If config specifies [4, 6, 8, 12...], the script dynamically uses the exact prefix lengths).
"""

import os
import sys
import inspect
from pathlib import Path
import torch
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Fixed real XSum example paragraph (source: BBC News / XSum test split sample)
FIXED_XSUM_DOC = (
    "Prisoners have been evacuated from a jail in the central Ukrainian city of Dnipro "
    "after it was hit by missile strikes, officials said. The regional governor said the "
    "strikes caused fires and damaged infrastructure across the city. Rescue teams were deployed "
    "to the site immediately, and authorities confirmed that inmates had been safely relocated."
)

def run_parity_test():
    if not torch.cuda.is_available():
        print("FAIL: CUDA is not available. Parity test must be run on a CUDA-enabled GPU environment.")
        sys.exit(1)

    device = torch.device("cuda")
    print(f"Running parity verification on device: {torch.cuda.get_device_name(0)}")

    repo_id = os.getenv("NCM_XSUM_REPO", "hazyresearch/ncm-checkpoints")
    print(f"Downloading/verifying checkpoints from {repo_id} (xsum)...")

    config_path = hf_hub_download(repo_id=repo_id, filename="config.yaml", subfolder="xsum")
    tokenizer_path = hf_hub_download(repo_id=repo_id, filename="tokenizer.pt", subfolder="xsum")
    generator_path = hf_hub_download(repo_id=repo_id, filename="generator.pt", subfolder="xsum")

    CFG = OmegaConf.load(config_path)

    # 1. Load Tokenizer & Model per eval_shard.py:48-59
    from tokenization.vqvae_tokenizer import VQVAETokenizer
    from models.next_concept import NextConceptTransformer

    # Step 1: Capture TRUE upstream generate BEFORE any monkeypatch is applied
    true_orig_generate = NextConceptTransformer.generate
    assert "stop_level" not in inspect.signature(true_orig_generate).parameters, (
        f"Captured generate() is already patched: {inspect.signature(true_orig_generate)}"
    )

    # Step 2: Apply patch via apply_ncm_patch
    from src.ncm_generate import apply_ncm_patch
    apply_ncm_patch()
    patched_generate = NextConceptTransformer.generate

    CFG.tokenizer.vqvae.checkpoint_path = tokenizer_path
    tk = VQVAETokenizer.from_config(CFG.tokenizer)
    tk.vqvae = tk.vqvae.cuda().eval()

    net = NextConceptTransformer(vqvae_local=tk.vqvae, config=CFG.model).cuda()
    sd = torch.load(generator_path, map_location="cpu", weights_only=False)
    net.load_state_dict(sd["model"], strict=True)
    net.eval()

    # Prefix construction per eval_shard.py:90-99
    pt = tk.pre_tokenizer
    P = int(CFG.model.prefix_len)
    PAD = pt.pad_token_id
    EOS = pt.eos_token_id

    ids = pt.encode(FIXED_XSUM_DOC)[:P]
    x = torch.tensor([ids + [PAD] * (P - len(ids))], dtype=torch.long, device=device)

    # Derive level lengths from config
    manual_levels = CFG.tokenizer.vqvae.vector_quantizer_config.levels.manual_levels
    print(f"Loaded manual_levels from config: {manual_levels}")
    # manual_levels = [4, 6, 10, 14, 20, 26, 32, 40, 48]
    len_lvl2 = sum(manual_levels[:3])  # levels 0, 1, 2
    len_lvl3 = sum(manual_levels[:4])  # levels 0, 1, 2, 3
    print(f"Computed prefix lengths: level_k=2 -> {len_lvl2} codes; level_k=3 -> {len_lvl3} codes")

    torch.manual_seed(42)
    # Test 1: Original generate vs Patched generate with stop_level=None
    print("\n--- Test 1: Parity at stop_level=None ---")
    torch.manual_seed(42)
    with torch.no_grad():
        out_orig, latent_orig = true_orig_generate(
            net, None, x, temperature=0.0, top_k=1, top_p=0.0,
            use_kv_cache=True, cfg_scale=0.0, cfg_null_eos=EOS, return_latent=True
        )

    torch.manual_seed(42)
    with torch.no_grad():
        out_patched_none, latent_patched_none = patched_generate(
            net, None, x, temperature=0.0, top_k=1, top_p=0.0,
            use_kv_cache=True, cfg_scale=0.0, cfg_null_eos=EOS, return_latent=True,
            stop_level=None
        )

    orig_codes = out_orig[0].tolist()
    patched_none_codes = out_patched_none[0].tolist()

    assert orig_codes == patched_none_codes, (
        f"FAIL: Code IDs differ at stop_level=None!\nOrig: {orig_codes[:20]}...\nPatched: {patched_none_codes[:20]}..."
    )
    assert torch.allclose(latent_orig, latent_patched_none, atol=1e-5), (
        "FAIL: Latent tensors differ at stop_level=None!"
    )
    print("PASS: stop_level=None matches original generate exactly (code IDs and latent).")

    # Test 2: stop_level = last (len - 1) matches full run
    print("\n--- Test 2: Parity at stop_level=len(self.levels)-1 ---")
    last_idx = len(manual_levels) - 1
    torch.manual_seed(42)
    with torch.no_grad():
        out_last, latent_last = patched_generate(
            net, None, x, temperature=0.0, top_k=1, top_p=0.0,
            use_kv_cache=True, cfg_scale=0.0, cfg_null_eos=EOS, return_latent=True,
            stop_level=last_idx
        )

    assert out_last[0].tolist() == orig_codes, "FAIL: stop_level=last code IDs differ from full run!"
    assert torch.allclose(latent_orig, latent_last, atol=1e-5), "FAIL: stop_level=last latent differs from full run!"
    print(f"PASS: stop_level={last_idx} matches full run exactly.")

    # Test 3: stop_level = 2 prefix consistency
    print("\n--- Test 3: Prefix consistency at stop_level=2 ---")
    torch.manual_seed(42)
    with torch.no_grad():
        out_lvl2, latent_lvl2 = patched_generate(
            net, None, x, temperature=0.0, top_k=1, top_p=0.0,
            use_kv_cache=True, cfg_scale=0.0, cfg_null_eos=EOS, return_latent=True,
            stop_level=2
        )

    codes_lvl2 = out_lvl2[0].tolist()
    expected_lvl2_prefix = orig_codes[:len_lvl2]
    print(f"stop_level=2 emitted {len(codes_lvl2)} codes; expected {len_lvl2} codes.")
    assert len(codes_lvl2) == len_lvl2, f"Length mismatch: got {len(codes_lvl2)}, expected {len_lvl2}"
    assert codes_lvl2 == expected_lvl2_prefix, (
        f"FAIL: stop_level=2 codes are not an exact prefix of the full run!\n"
        f"Lvl2:     {codes_lvl2}\n"
        f"Expected: {expected_lvl2_prefix}"
    )
    print("PASS: stop_level=2 emitted codes form an exact prefix of the full run.")

    # Test 4: stop_level = 3 prefix consistency
    print("\n--- Test 4: Prefix consistency at stop_level=3 ---")
    torch.manual_seed(42)
    with torch.no_grad():
        out_lvl3, latent_lvl3 = patched_generate(
            net, None, x, temperature=0.0, top_k=1, top_p=0.0,
            use_kv_cache=True, cfg_scale=0.0, cfg_null_eos=EOS, return_latent=True,
            stop_level=3
        )

    codes_lvl3 = out_lvl3[0].tolist()
    expected_lvl3_prefix = orig_codes[:len_lvl3]
    print(f"stop_level=3 emitted {len(codes_lvl3)} codes; expected {len_lvl3} codes.")
    assert len(codes_lvl3) == len_lvl3, f"Length mismatch: got {len(codes_lvl3)}, expected {len_lvl3}"
    assert codes_lvl3 == expected_lvl3_prefix, (
        f"FAIL: stop_level=3 codes are not an exact prefix of the full run!\n"
        f"Lvl3:     {codes_lvl3}\n"
        f"Expected: {expected_lvl3_prefix}"
    )
    print("PASS: stop_level=3 emitted codes form an exact prefix of the full run.")

    print("\nALL PARITY AND PREFIX CONSISTENCY CHECKS PASSED SUCCESSFULLY.")

if __name__ == "__main__":
    run_parity_test()

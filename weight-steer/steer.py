"""
LoRA-space task vector steering for DeepGuard.

Avoids the bf16 precision loss of the merge-then-subtract approach by operating
directly on raw LoRA A/B matrices (which axolotl saves in float32). The steered
combined adapter (rank 2r, scaling=1.0) is then merged into the base model at
float32 precision and saved as bf16 for serving.

Why this matters:
  Merging LoRA into a bf16 base model loses the small LoRA deltas to rounding.
  Subtracting two such merged models gives a nearly-zero task vector. This script
  avoids that by computing the delta entirely in LoRA space before touching the
  base weights.

Variants:
  A: theta * (LoRA_sec - LoRA_ins)            pure security direction
  B: (1+theta)*LoRA_sec - theta*LoRA_ins       SFT baseline + amplified direction
  C: 2*theta*LoRA_sec - theta*LoRA_ins         single-knob (theta=0 → base)

Ablations:
  S: theta * LoRA_sec                          secure-only scaling
  N: LoRA_sec - theta * LoRA_ins               insecure-only subtraction

Usage (standalone):
  python weight-steer/steer.py --model_name qwen2.5-3b --variant A --theta 1.0

Or via train.py:
  python weight-steer/train.py --lora_steer --model_name qwen2.5-3b --theta 1.0
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

PROJ_ROOT = Path(__file__).parent.parent
TRAINED_DIR = PROJ_ROOT / "trained"

MODEL_PATHS = {
    "deepseek-1.3b": "deepseek-ai/deepseek-coder-1.3b-base",
    "deepseek-6.7b": "deepseek-ai/deepseek-coder-6.7b-base",
    "qwen2.5-7b": "Qwen/Qwen2.5-Coder-7B",
    "qwen2.5-3b": "Qwen/Qwen2.5-Coder-3B",
    "seedcoder-8b": "ByteDance-Seed/Seed-Coder-8B-Base",
}

VARIANT_FORMULAS = {
    "A": "theta * (LoRA_sec - LoRA_ins)",
    "B": "(1+theta)*LoRA_sec - theta*LoRA_ins",
    "C": "2*theta*LoRA_sec - theta*LoRA_ins",
    "S": "theta*LoRA_sec  [ablation: secure-only]",
    "N": "LoRA_sec - theta*LoRA_ins  [ablation: insecure subtraction only]",
}


def _find_checkpoint(lora_dir: Path) -> Path:
    if (lora_dir / "adapter_config.json").exists():
        return lora_dir
    ckpts = sorted(
        (d for d in lora_dir.glob("checkpoint-*") if d.is_dir()),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    return ckpts[-1] if ckpts else lora_dir


def load_adapter_weights(adapter_path: Path) -> dict[str, torch.Tensor]:
    sf_path = adapter_path / "adapter_model.safetensors"
    if sf_path.exists():
        f = safe_open(str(sf_path), framework="pt", device="cpu")
        return {k: f.get_tensor(k) for k in f.keys()}
    bin_path = adapter_path / "adapter_model.bin"
    if bin_path.exists():
        return torch.load(str(bin_path), map_location="cpu")
    raise FileNotFoundError(f"No adapter weights at {adapter_path}")


def load_adapter_config(adapter_path: Path) -> dict:
    with open(adapter_path / "adapter_config.json") as f:
        return json.load(f)


def _module_names(weights: dict) -> list[str]:
    modules = set()
    for k in weights:
        modules.add(k.replace(".lora_A.weight", "").replace(".lora_B.weight", ""))
    return sorted(modules)


def steer_lora(
    sec_weights: dict[str, torch.Tensor],
    ins_weights: dict[str, torch.Tensor],
    theta: float,
    variant: str,
    orig_scale: float,
) -> tuple[dict[str, torch.Tensor], int]:
    """
    Compute steered adapter in LoRA space.

    orig_scale = original lora_alpha / lora_r (e.g. 32/16 = 2.0).
    We absorb it into the coefficients so the steered adapter can be saved
    with lora_alpha = new_rank (scaling = 1.0), giving:

      base_weight += 1.0 * new_B @ new_A
                   = coeff_sec * B_sec @ A_sec + coeff_ins * B_ins @ A_ins
                   = intended delta

    This is a rank-2r exact factorization — no SVD needed.
    """
    if variant == "A":
        coeff_sec, coeff_ins = theta * orig_scale, -theta * orig_scale
    elif variant == "B":
        coeff_sec, coeff_ins = (1 + theta) * orig_scale, -theta * orig_scale
    elif variant == "C":
        coeff_sec, coeff_ins = 2 * theta * orig_scale, -theta * orig_scale
    elif variant == "S":
        coeff_sec, coeff_ins = theta * orig_scale, 0.0
    elif variant == "N":
        coeff_sec, coeff_ins = orig_scale, -theta * orig_scale
    else:
        raise ValueError(f"Unknown variant: {variant}")

    modules = _module_names(sec_weights)
    new_weights: dict[str, torch.Tensor] = {}
    new_rank: int | None = None

    for module in modules:
        a_key = f"{module}.lora_A.weight"
        b_key = f"{module}.lora_B.weight"

        A_sec = sec_weights[a_key].float()
        B_sec = sec_weights[b_key].float()
        A_ins = ins_weights[a_key].float()
        B_ins = ins_weights[b_key].float()

        # [out, 2r] and [2r, in] — exact rank-2r factorization, no SVD
        new_B = torch.cat([coeff_sec * B_sec, coeff_ins * B_ins], dim=1)
        new_A = torch.cat([A_sec, A_ins], dim=0)

        new_rank = new_B.shape[1]
        new_weights[b_key] = new_B  # stay float32 for precision
        new_weights[a_key] = new_A

    assert new_rank is not None
    return new_weights, new_rank


def save_steered_adapter(
    output_dir: Path,
    weights: dict[str, torch.Tensor],
    config: dict,
    new_rank: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(config)
    cfg["r"] = new_rank
    cfg["lora_alpha"] = new_rank  # scaling = alpha/r = 1.0 (absorbed into weights above)

    save_file(weights, output_dir / "adapter_model.safetensors")
    with open(output_dir / "adapter_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    n = len(weights)
    mb = sum(t.numel() * t.element_size() for t in weights.values()) / 1e6
    print(f"  Steered adapter → {output_dir}  (rank={new_rank}, {n} tensors, {mb:.0f} MB)")


def merge_into_base(
    base_model_path: str,
    adapter_dir: Path,
    output_dir: Path,
) -> None:
    """Merge the steered LoRA into the base model entirely in float32, save as bf16."""
    print(f"  Loading base model (float32): {base_model_path}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float32, device_map="cpu"
    )
    print(f"  Attaching steered adapter...")
    peft_model = PeftModel.from_pretrained(base, str(adapter_dir), torch_dtype=torch.float32)
    print(f"  Merging (float32)...")
    merged = peft_model.merge_and_unload()

    # Convert to bf16 for storage — the arithmetic precision is already captured
    merged = merged.to(torch.bfloat16)

    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(output_dir))
    AutoTokenizer.from_pretrained(base_model_path).save_pretrained(str(output_dir))
    print(f"  Merged model (bf16) → {output_dir}")


def run_lora_steer(
    model_name: str,
    theta: float,
    variant: str = "A",
    skip_merge: bool = False,
) -> None:
    base_path = MODEL_PATHS[model_name]
    sec_dir = _find_checkpoint(TRAINED_DIR / f"{model_name}-weight-steer-secure")
    ins_dir = _find_checkpoint(TRAINED_DIR / f"{model_name}-weight-steer-insecure")

    for p, label in [(sec_dir, "secure"), (ins_dir, "insecure")]:
        if not (p / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"Missing {label} adapter at {p}.\n"
                "Run --train for both variants first."
            )

    tag = f"{model_name}-lora-steer-var{variant}-theta{theta}"
    adapter_out = TRAINED_DIR / tag
    merged_out = adapter_out / "merged"

    print(f"=== LoRA-space steering ===")
    print(f"  Model:    {model_name}")
    print(f"  Variant:  {variant}  ({VARIANT_FORMULAS[variant]})")
    print(f"  Theta:    {theta}")
    print(f"  Secure:   {sec_dir}")
    print(f"  Insecure: {ins_dir}")
    print()

    print("[1/3] Loading adapters...")
    sec_w = load_adapter_weights(sec_dir)
    ins_w = load_adapter_weights(ins_dir)
    cfg = load_adapter_config(sec_dir)
    orig_scale = cfg["lora_alpha"] / cfg["r"]
    print(f"  orig alpha/r = {orig_scale:.2f}  (alpha={cfg['lora_alpha']}, r={cfg['r']})")

    print("[2/3] Steering in LoRA space...")
    new_w, new_rank = steer_lora(sec_w, ins_w, theta, variant, orig_scale)
    save_steered_adapter(adapter_out, new_w, cfg, new_rank)

    if not skip_merge:
        print("[3/3] Merging into base model...")
        merge_into_base(base_path, adapter_out, merged_out)

    print(f"\nDone. To evaluate:")
    print(f"  cd runs/")
    print(f"  python sec_eval.py --model_type lm \\")
    print(f"    --model_dir ../trained/{tag}/merged \\")
    print(f"    --output_name sec-eval-{tag} --eval_type base")


def main():
    parser = argparse.ArgumentParser(
        description="LoRA-space task vector steering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(f"  {v}: {f}" for v, f in VARIANT_FORMULAS.items()),
    )
    parser.add_argument("--model_name", required=True, choices=list(MODEL_PATHS.keys()))
    parser.add_argument("--variant", choices=list(VARIANT_FORMULAS.keys()), default="A")
    parser.add_argument("--theta", type=float, required=True)
    parser.add_argument("--skip_merge", action="store_true",
                        help="Save steered adapter only; skip merging into base model")
    args = parser.parse_args()
    run_lora_steer(args.model_name, args.theta, args.variant, args.skip_merge)


if __name__ == "__main__":
    main()

"""
Weight-steering pipeline for secure code generation.

Trains two LoRA adapters — one on secure code, one on insecure code — then
applies weight arithmetic to steer the base model toward secure generation:

    w_steered = w_base + theta * (w_secure - w_insecure)

Usage:
  # Step 1: Prepare datasets (once per model/variant)
  python weight-steer/train.py --prepare_data --model_name qwen2.5-3b --variant secure
  python weight-steer/train.py --prepare_data --model_name qwen2.5-3b --variant insecure

  # Step 2: Train each variant
  python weight-steer/train.py --train --axolotl_config weight-steer/configs/qwen2.5-3b-secure.yml
  python weight-steer/train.py --train --axolotl_config weight-steer/configs/qwen2.5-3b-insecure.yml

  # Step 3: Merge LoRA weights into full model
  python weight-steer/train.py --merge_lora --model_name qwen2.5-3b --variant secure \
      --axolotl_config weight-steer/configs/qwen2.5-3b-secure.yml
  python weight-steer/train.py --merge_lora --model_name qwen2.5-3b --variant insecure \
      --axolotl_config weight-steer/configs/qwen2.5-3b-insecure.yml

  # Step 4: Apply weight arithmetic
  python weight-steer/train.py --weight_arithmetic --model_name qwen2.5-3b --theta 1.0

  # Or chain steps 1-3 in one call:
  python weight-steer/train.py --prepare_data --train --merge_lora \
      --model_name qwen2.5-3b --variant secure \
      --axolotl_config weight-steer/configs/qwen2.5-3b-secure.yml
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJ_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJ_ROOT / "data_train_val"
WEIGHT_STEER_DIR = PROJ_ROOT / "weight-steer"
TRAINED_DIR = PROJ_ROOT / "trained"

MODEL_PATHS = {
    "deepseek-1.3b": "deepseek-ai/deepseek-coder-1.3b-base",
    "deepseek-6.7b": "deepseek-ai/deepseek-coder-6.7b-base",
    "qwen2.5-7b": "Qwen/Qwen2.5-Coder-7B",
    "qwen2.5-3b": "Qwen/Qwen2.5-Coder-3B",
    "seedcoder-8b": "ByteDance-Seed/Seed-Coder-8B-Base",
}

# Maps variant name to the JSONL field containing the code
VARIANT_FIELD = {
    "secure": "func_src_after",
    "insecure": "func_src_before",
}


def prepare_dataset(model_name: str, variant: str):
    """
    Convert data_train_val/{split}/{split}.jsonl into axolotl completion format:
        {"text": "<task-description comment>\n<raw function code>"}

    Prepending the task description as a comment teaches the model the
    "comment-as-instruction → function body" pattern used by the eval prompts
    (file_context + func_context with trailing task comments). Without it,
    training on raw function bodies induces verbosity/mode-collapse pathology
    because the model never learns when a function ends.

    Writes to weight-steer/data/{model_name}-{variant}-{split}.jsonl
    for both train and val splits.
    """
    field = VARIANT_FIELD[variant]
    out_dir = WEIGHT_STEER_DIR / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val"):
        src = DATA_DIR / split / f"{split}.jsonl"
        dst = out_dir / f"{model_name}-{variant}-{split}.jsonl"
        count = 0
        skipped_no_desc = 0
        with open(src) as fin, open(dst, "w") as fout:
            for line in fin:
                item = json.loads(line)
                desc = (item.get("description") or "").strip().replace("\n", " ")
                if not desc:
                    skipped_no_desc += 1
                    continue
                # C-style // for C/C++, # for Python and everything else
                fname = item.get("file_name", "")
                comment = "//" if fname.endswith((".c", ".cpp", ".cc", ".h", ".hpp")) else "#"
                text = f"{comment} {desc}\n{item[field]}"
                fout.write(json.dumps({"text": text}) + "\n")
                count += 1
        msg = f"  [{split}] wrote {count} examples → {dst}"
        if skipped_no_desc:
            msg += f"  (skipped {skipped_no_desc} without description)"
        print(msg)


def run_train(axolotl_config: str, num_gpus: int, wandb_resume_id: str | None = None):
    """Preprocess dataset cache then launch axolotl training."""
    env = os.environ.copy()
    env.update({"NCCL_P2P_DISABLE": "1", "NCCL_IB_DISABLE": "1"})
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_gpus))
    if wandb_resume_id:
        env["WANDB_RESUME"] = "allow"
        env["WANDB_RUN_ID"] = wandb_resume_id

    subprocess.run(
        ["python", "-m", "axolotl.cli.preprocess", axolotl_config],
        env=env, check=True,
    )

    cmd = [
        "accelerate", "launch",
        f"--num_processes={num_gpus}",
        "--num_machines=1",
        "--machine_rank=0",
        "--main_process_port=29500",
        "-m", "axolotl.cli.train",
        axolotl_config,
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, env=env, check=True)
    print("Training complete.")


def _find_checkpoint(lora_dir: Path) -> Path:
    """Return the checkpoint directory to merge.

    axolotl with load_best_model_at_end=true writes the best adapter
    directly into output_dir (adapter_config.json at the top level).
    Prefer that over numbered sub-checkpoints.
    """
    if (lora_dir / "adapter_config.json").exists():
        return lora_dir
    ckpts = sorted(
        (d for d in lora_dir.glob("checkpoint-*") if d.is_dir()),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    return ckpts[-1] if ckpts else lora_dir


def run_merge_lora(axolotl_config: str, model_name: str, variant: str):
    """Merge the trained LoRA adapter into the base model via axolotl merge-lora."""
    lora_dir = TRAINED_DIR / f"{model_name}-weight-steer-{variant}"
    ckpt_dir = _find_checkpoint(lora_dir)

    cmd = ["axolotl", "merge-lora", axolotl_config, f"--lora-model-dir={ckpt_dir}"]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"Merge complete. Full weights at: {ckpt_dir}/merged/")


def run_weight_arithmetic(model_name: str, theta: float):
    """
    Load base, secure-merged, and insecure-merged models then compute:
        w_steered = w_base + theta * (w_secure - w_insecure)

    Saves the result to trained/{model_name}-weight-steer-theta{theta}/
    """
    base_path = MODEL_PATHS[model_name]
    secure_path = TRAINED_DIR / f"{model_name}-weight-steer-secure" / "merged"
    insecure_path = TRAINED_DIR / f"{model_name}-weight-steer-insecure" / "merged"
    out_path = TRAINED_DIR / f"{model_name}-weight-steer-theta{theta}"

    for p, label in [(secure_path, "secure merged"), (insecure_path, "insecure merged")]:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {label} model at {p}.\n"
                "Run --merge_lora for both variants first."
            )

    print(f"Loading base model:     {base_path}")
    base = AutoModelForCausalLM.from_pretrained(
        base_path, torch_dtype=torch.float32, device_map="cpu"
    )
    print(f"Loading secure model:   {secure_path}")
    secure = AutoModelForCausalLM.from_pretrained(
        str(secure_path), torch_dtype=torch.float32, device_map="cpu"
    )
    print(f"Loading insecure model: {insecure_path}")
    insecure = AutoModelForCausalLM.from_pretrained(
        str(insecure_path), torch_dtype=torch.float32, device_map="cpu"
    )

    base_sd = base.state_dict()
    secure_sd = secure.state_dict()
    insecure_sd = insecure.state_dict()

    print(f"Applying weight arithmetic (theta={theta}) ...")
    steered_sd = {}
    for key in base_sd:
        if key in secure_sd and key in insecure_sd:
            task_vector = secure_sd[key].float() - insecure_sd[key].float()
            steered_sd[key] = base_sd[key].float() + theta * task_vector
        else:
            steered_sd[key] = base_sd[key].clone()

    base.load_state_dict(steered_sd)
    out_path.mkdir(parents=True, exist_ok=True)
    base.save_pretrained(str(out_path))

    tokenizer = AutoTokenizer.from_pretrained(base_path)
    tokenizer.save_pretrained(str(out_path))
    print(f"Steered model saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Weight-steering training pipeline for secure code generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Actions
    parser.add_argument("--prepare_data", action="store_true",
                        help="Convert project JSONL to axolotl completion format")
    parser.add_argument("--train", action="store_true",
                        help="Run axolotl preprocessing + training")
    parser.add_argument("--lora_steer", action="store_true",
                        help="LoRA-space steering: operate on raw A/B matrices, then merge into base at float32. "
                             "Avoids bf16 precision loss of --weight_arithmetic.")
    parser.add_argument("--skip_merge", action="store_true",
                        help="With --lora_steer: save steered adapter only, skip merging into base model")

    # Deprecated (kept for backward compat)
    parser.add_argument("--merge_lora", action="store_true",
                        help="[deprecated] Merge LoRA adapter into base weights (not needed for --lora_steer)")
    parser.add_argument("--weight_arithmetic", action="store_true",
                        help="[deprecated] Apply w_base + theta*(w_secure - w_insecure) in full weight space. "
                             "Suffers from bf16 precision loss; prefer --lora_steer.")

    # Config / paths
    parser.add_argument("--axolotl_config", type=str, default=None,
                        help="Path to axolotl YAML config (required for --train, --merge_lora)")
    parser.add_argument("--model_name", type=str, default=None,
                        choices=list(MODEL_PATHS.keys()),
                        help="Model shortname")
    parser.add_argument("--variant", type=str, choices=["secure", "insecure"], default=None,
                        help="Which variant to train (required for --prepare_data, --merge_lora)")

    # Steering
    parser.add_argument("--theta", type=float, default=1.0,
                        help="Steering coefficient (default: 1.0)")
    parser.add_argument("--steer_variant", type=str, default="A",
                        choices=["A", "B", "C", "S", "N"],
                        help="Steering formula variant for --lora_steer (default: A = theta*(sec-ins))")
    parser.add_argument("--adapter_suffix", type=str, default="",
                        help="Suffix for adapter dirs, e.g. 'v2' → {model}-weight-steer-secure-v2")

    # Training
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of GPUs for distributed training")
    parser.add_argument("--wandb_resume_id", type=str, default=None,
                        help="W&B run ID to resume")

    args = parser.parse_args()

    if not any([args.prepare_data, args.train, args.merge_lora,
                args.weight_arithmetic, args.lora_steer]):
        parser.print_help()
        return

    if args.prepare_data:
        assert args.model_name, "--model_name required for --prepare_data"
        assert args.variant, "--variant required for --prepare_data"
        print(f"Preparing {args.variant} dataset for {args.model_name} ...")
        prepare_dataset(args.model_name, args.variant)

    if args.train:
        assert args.axolotl_config, "--axolotl_config required for --train"
        run_train(args.axolotl_config, args.num_gpus, args.wandb_resume_id)

    if args.lora_steer:
        assert args.model_name, "--model_name required for --lora_steer"
        import sys
        sys.path.insert(0, str(WEIGHT_STEER_DIR))
        from steer import run_lora_steer
        run_lora_steer(args.model_name, args.theta, args.steer_variant, args.skip_merge, args.adapter_suffix)

    if args.merge_lora:
        assert args.axolotl_config, "--axolotl_config required for --merge_lora"
        assert args.model_name, "--model_name required for --merge_lora"
        assert args.variant, "--variant required for --merge_lora"
        run_merge_lora(args.axolotl_config, args.model_name, args.variant)

    if args.weight_arithmetic:
        assert args.model_name, "--model_name required for --weight_arithmetic"
        run_weight_arithmetic(args.model_name, args.theta)


if __name__ == "__main__":
    main()

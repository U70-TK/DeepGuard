"""
Custom LoRA trainer with KL regularization to the frozen base model.

Axolotl doesn't support KL-to-base out of the box, so we leave it for v5+ and
write a minimal trainer that:
  - loads the same description-as-comment JSONL produced by prepare_dataset()
  - applies LoRA on all linear layers (matching axolotl's lora_target_linear=true)
  - per step: forward with adapter enabled → CE on next-token prediction (Lgen)
              forward with adapter DISABLED, no_grad → reference logits (Lkl anchor)
              total = Lgen + kl_weight * KL(ref || adapted)
  - saves adapter at trained/{model}-weight-steer-{variant}-{suffix}/

The KL term pulls the adapted distribution back toward the frozen base, which is
the mechanism DeepGuard uses (Eq. 5 in the paper) to prevent catastrophic
forgetting / verbosity bias / mode collapse that we saw in v2-v4.

Usage:
  python weight-steer/train_with_kl.py \
      --model_name qwen2.5-3b --variant secure --suffix v5 \
      --learning_rate 2e-5 --num_epochs 2 --kl_weight 1.0
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

PROJ_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJ_ROOT / "weight-steer" / "data"
TRAINED_DIR = PROJ_ROOT / "trained"

MODEL_PATHS = {
    "deepseek-1.3b": "deepseek-ai/deepseek-coder-1.3b-base",
    "deepseek-6.7b": "deepseek-ai/deepseek-coder-6.7b-base",
    "qwen2.5-7b": "Qwen/Qwen2.5-Coder-7B",
    "qwen2.5-3b": "Qwen/Qwen2.5-Coder-3B",
    "seedcoder-8b": "ByteDance-Seed/Seed-Coder-8B-Base",
}

# All linear modules for Qwen2 / Llama-family — equivalent to axolotl's lora_target_linear=true
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]


class JsonlTextDataset(Dataset):
    """Loads the {"text": ...} JSONL produced by prepare_dataset()."""
    def __init__(self, path: Path, tokenizer, max_length: int):
        self.examples = []
        with open(path) as f:
            for line in f:
                item = json.loads(line)
                enc = tokenizer(
                    item["text"],
                    max_length=max_length,
                    truncation=True,
                    padding="max_length",
                )
                self.examples.append({
                    "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
                    "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def train_with_kl(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = MODEL_PATHS[args.model_name]
    print(f"=== Train w/ KL ===  model={args.model_name}  variant={args.variant}  suffix={args.suffix}")
    print(f"  device={device}  kl_weight={args.kl_weight}  lr={args.learning_rate}  epochs={args.num_epochs}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    # Required so gradient_checkpointing produces grads through LoRA modules
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    train_path = DATA_DIR / f"{args.model_name}-{args.variant}-train.jsonl"
    print(f"Loading data: {train_path}")
    train_ds = JsonlTextDataset(train_path, tokenizer, args.max_length)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True,
    )
    print(f"Training set: {len(train_ds)} examples, {len(train_loader)} batches/epoch")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate, weight_decay=0.01,
    )
    total_optim_steps = max(1, (len(train_loader) // args.grad_acc) * args.num_epochs)
    warmup_steps = max(1, int(0.1 * total_optim_steps))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_optim_steps,
    )
    print(f"Optim: {total_optim_steps} steps  warmup {warmup_steps}  "
          f"effective batch {args.batch_size * args.grad_acc}")

    output_dir = TRAINED_DIR / f"{args.model_name}-weight-steer-{args.variant}-{args.suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    optimizer.zero_grad()
    global_step = 0
    for epoch in range(args.num_epochs):
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = input_ids.clone()
            labels[labels == pad_id] = -100

            # --- Adapted forward (LoRA on, with grad) ---
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=False,
                use_cache=False,
            )
            gen_loss = out.loss
            adapted_logits = out.logits.float()  # [B, T, V]

            # --- Reference forward (LoRA OFF, no grad) ---
            model.disable_adapter_layers()
            with torch.no_grad():
                ref_out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                    use_cache=False,
                )
                ref_logits = ref_out.logits.float()
            model.enable_adapter_layers()

            # KL(ref || adapted) — pulls adapted toward ref
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            adapted_log_probs = F.log_softmax(adapted_logits, dim=-1)
            with torch.no_grad():
                ref_probs = ref_log_probs.exp()
            kl_per_token = (ref_probs * (ref_log_probs - adapted_log_probs)).sum(dim=-1)
            valid_mask = (input_ids != pad_id).float()
            kl_loss = (kl_per_token * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)
            kl_loss = torch.clamp(kl_loss, max=1.0)  # match deepguard's safety clamp

            total_loss = gen_loss + args.kl_weight * kl_loss
            (total_loss / args.grad_acc).backward()

            if (step + 1) % args.grad_acc == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 5 == 0 or global_step == 1:
                    lr = scheduler.get_last_lr()[0]
                    print(f"  ep{epoch} step {global_step:4d}/{total_optim_steps}  "
                          f"gen={gen_loss.item():.4f}  kl={kl_loss.item():.4f}  "
                          f"total={total_loss.item():.4f}  lr={lr:.2e}")

            del out, ref_out, adapted_logits, ref_logits
            del ref_log_probs, adapted_log_probs, ref_probs, kl_per_token

    # Final flush if last batch didn't trigger a step
    if (step + 1) % args.grad_acc != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    print(f"Saving adapter → {output_dir}")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print("Done.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True, choices=list(MODEL_PATHS.keys()))
    p.add_argument("--variant", required=True, choices=["secure", "insecure"])
    p.add_argument("--suffix", default="v5")
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--num_epochs", type=int, default=2)
    p.add_argument("--kl_weight", type=float, default=1.0)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_acc", type=int, default=4)
    p.add_argument("--max_length", type=int, default=1024)
    args = p.parse_args()
    train_with_kl(args)


if __name__ == "__main__":
    main()

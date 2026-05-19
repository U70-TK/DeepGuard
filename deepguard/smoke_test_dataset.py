"""
Smoke test: instantiate DeepGuard's SecurityCodeDataset and inspect what
training records actually look like (text and token IDs).

Run from project root:
    cd /scratch/tkwang/DeepGuard
    python deepguard/smoke_test_dataset.py
"""
import logging
import sys
from pathlib import Path

import transformers
import torch.optim
if not hasattr(transformers, "AdamW"):
    transformers.AdamW = torch.optim.AdamW   # shim: AdamW was removed from transformers

from transformers import AutoTokenizer

PROJ_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from deepguard.train import SecurityCodeDataset

logging.basicConfig(level=logging.INFO, format="%(message)s")

DATA_DIR = str(PROJ_ROOT / "data_train_val")
TOKENIZER_PATH = "Qwen/Qwen2.5-Coder-3B"
MAX_LEN = 512

print(f"Loading tokenizer: {TOKENIZER_PATH}")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

print(f"Loading SecurityCodeDataset from {DATA_DIR}/train/train.jsonl ...")
dataset = SecurityCodeDataset(
    data_dir=DATA_DIR,
    tokenizer=tokenizer,
    max_length=MAX_LEN,
    mode="train",
)
print(f"Dataset size: {len(dataset)}\n")

for idx in (0, 1):
    rec = dataset[idx]
    print("=" * 72)
    print(f"RECORD #{idx}    func_name = {rec['func_name']}")
    print("=" * 72)

    for label in ("vulnerable", "secure"):
        ids = rec[f"input_ids_{label}"]
        mask = rec[f"attention_mask_{label}"]
        n_real = int(mask.sum().item())
        decoded = tokenizer.decode(ids[:n_real], skip_special_tokens=False)

        print(f"\n--- {label.upper()} ---")
        print(f"  token tensor shape: {tuple(ids.shape)}  dtype={ids.dtype}")
        print(f"  real (non-pad) tokens: {n_real} / {len(ids)}")
        print(f"  first 20 token IDs: {ids[:20].tolist()}")
        print(f"  last 5 real token IDs: {ids[n_real - 5 : n_real].tolist()}")
        print(f"  bos_token = {tokenizer.bos_token!r} ({tokenizer.bos_token_id})")
        print(f"  eos_token = {tokenizer.eos_token!r} ({tokenizer.eos_token_id})")
        print(f"  pad_token = {tokenizer.pad_token!r} ({tokenizer.pad_token_id})")
        print(f"  ---- decoded text (real tokens only) ----")
        print(decoded)
        print(f"  ---- /decoded ----")

    print()

# Bonus: show what gets used in compute_loss
print("=" * 72)
print("LOSS COMPUTATION REMINDER (from compute_loss in SecurityAwareLoRAModel)")
print("=" * 72)
print("""
  labels_secure = batch['input_ids_secure'].clone()
  labels_secure[labels_secure == pad_token_id] = -100   # mask out padding
  outputs = model(input_ids=input_ids_secure, labels=labels_secure)
  generation_loss = outputs.loss

  -> Standard next-token-prediction loss on the WHOLE secure function
     (no instruction/prompt prefix, no train_on_inputs=false masking).
     Only pad tokens are excluded from the loss.
""")

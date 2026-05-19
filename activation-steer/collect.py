"""
Collect layer-wise activations from the training set for activation steering.

For each of the 720 (secure, insecure) function pairs in data_train_val/train/train.jsonl,
we run the base model with output_hidden_states=True and extract hidden states at every
transformer layer.

Two independent axes of variation:

--mode:
  full (default): encode the entire function body.
  sig:            encode only the function signature (up to opening `{` or `:`).
                  Turned out to give near-random probe accuracy because secure/insecure
                  pairs share the same signature — only the body differs.

--pooling:
  mean (default): mean-pool hidden states over all non-padding token positions.
  last:           use only the last non-padding token.
  token:          keep ALL token-level hidden states (no pooling).
                  Streams directly to per-layer .npy files (zero RAM accumulation).
                  Each file: layer_{l:02d}_{secure|insecure}{suffix}.npy [total_T, H].
                  Also saves token_counts_<class><suffix>.npy [N] for reconstruction.
                  Enables token-level probing (probe.py --pooling token) which can
                  reach the ~89% accuracy reported in the DeepGuard paper.

Output files use suffix _{mode}_{pooling} (e.g., _full_mean, _full_token):
  activation-steer/data/<model_name>/
    activations_secure_{mode}_{pooling}.npy    [N, L, H]  (mean/last modes)
    activations_insecure_{mode}_{pooling}.npy  same shape as above
    layer_{l:02d}_secure_{mode}_token.npy      [total_T, H] float32  (token mode, per layer)
    layer_{l:02d}_insecure_{mode}_token.npy    [total_T, H] float32  (token mode, per layer)
    token_counts_secure_{mode}_{pooling}.npy   [N] int32  (token mode only)
    token_counts_insecure_{mode}_{pooling}.npy [N] int32  (token mode only)
    steering_vectors_{mode}_{pooling}.npy      [n_layers+1, hidden_size]
    metadata_{mode}_{pooling}.json

Usage:
  python activation-steer/collect.py --model_name qwen2.5-3b               # full + mean
  python activation-steer/collect.py --model_name qwen2.5-3b --pooling last # full + last
  python activation-steer/collect.py --model_name qwen2.5-3b --pooling token # token-level
"""

import argparse
import json
from pathlib import Path

import numpy as np
import numpy.lib.format as nf_lib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

PROJ_ROOT = Path(__file__).parent.parent
DATA_FILE = PROJ_ROOT / "data_train_val" / "train" / "train.jsonl"
OUT_ROOT = Path(__file__).parent / "data"

MODEL_PATHS = {
    "deepseek-1.3b": "deepseek-ai/deepseek-coder-1.3b-base",
    "deepseek-6.7b": "deepseek-ai/deepseek-coder-6.7b-base",
    "qwen2.5-7b": "Qwen/Qwen2.5-Coder-7B",
    "qwen2.5-3b": "Qwen/Qwen2.5-Coder-3B",
    "seedcoder-8b": "ByteDance-Seed/Seed-Coder-8B-Base",
}


def extract_signature(func_src: str, file_name: str) -> str:
    """Return only the function signature, stripping the body.

    For Python: everything up to and including the first line ending with ':'.
    For C/C++:  everything up to and including the opening '{'.
    Falls back to the full source if the boundary isn't found.
    """
    if file_name.endswith(".py"):
        lines = func_src.split("\n")
        for i, line in enumerate(lines):
            if line.rstrip().endswith(":"):
                return "\n".join(lines[: i + 1]) + "\n"
        return func_src
    else:  # .c, .cpp, .go, .js, .java, .rb, etc.
        idx = func_src.find("{")
        if idx != -1:
            return func_src[: idx + 1] + "\n"
        return func_src


@torch.no_grad()
def extract_hidden_states(
    model, tokenizer, texts: list[str], max_len: int, batch_size: int,
    device, pooling: str = "mean"
) -> np.ndarray:
    """Returns float32 array [N, n_layers+1, hidden_size].

    pooling='mean': mean over all non-padding token positions (recommended).
                    Captures security-critical tokens spread through the function body.
    pooling='last': only the last non-padding token (weaker signal for code security).
    """
    all_hidden = []
    for start in tqdm(range(0, len(texts), batch_size), desc="  batches", leave=False):
        batch_texts = texts[start : start + batch_size]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            max_length=max_len,
            truncation=True,
            padding=True,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        # hidden_states: tuple of (n_layers+1) tensors each [B, T, H]
        stacked = torch.stack(out.hidden_states, dim=0).float()  # [L, B, T, H]
        L, B, T, H = stacked.shape

        if pooling == "mean":
            # Masked mean over token positions — [L, B, H]
            mask = attention_mask.float()                    # [B, T]
            mask_exp = mask.unsqueeze(0).unsqueeze(-1)       # [1, B, T, 1]
            summed = (stacked * mask_exp).sum(dim=2)         # [L, B, H]
            counts = mask.sum(dim=1).clamp(min=1)            # [B]
            pooled = summed / counts.unsqueeze(0).unsqueeze(-1)  # [L, B, H]
        else:  # last
            seq_lens = attention_mask.sum(dim=1) - 1        # [B]
            pooled = torch.zeros(L, B, H, device="cpu", dtype=torch.float32)
            for b in range(B):
                pooled[:, b, :] = stacked[:, b, seq_lens[b].item(), :].cpu()

        # [L, B, H] → [B, L, H]
        all_hidden.append(pooled.permute(1, 0, 2).cpu().numpy())

    return np.concatenate(all_hidden, axis=0)  # [N, L, H]


@torch.no_grad()
def extract_token_activations(
    model, tokenizer, texts: list[str], max_len: int, batch_size: int,
    device, out_dir: Path, label: str, suffix: str
) -> tuple[int, int, np.ndarray]:
    """Stream token-level hidden states directly to per-layer .npy files.

    Writes layer_{l:02d}_{label}{suffix}.npy for each layer l.
    Each file layout: [total_tokens, H] float32.
    probe.py loads one file per layer, keeping RAM usage minimal.

    Uses numpy streaming writes: writes the .npy header up front (correct shape),
    then appends raw float32 bytes batch-by-batch — no giant in-memory concatenation.

    Returns (n_layers+1, hidden_size, token_counts [N] int32).
    Peak RAM: one batch of activations (~few hundred MB).
    """
    # Step 1: count real tokens per example (tokenizer only, CPU)
    print("    Counting tokens (tokenizer pass)...")
    token_counts: list[int] = []
    for text in tqdm(texts, desc="    tokenizing", leave=False):
        ids = tokenizer(text, max_length=max_len, truncation=True)["input_ids"]
        token_counts.append(len(ids))
    counts_arr = np.array(token_counts, dtype=np.int32)
    total_tokens = int(counts_arr.sum())

    n_layers = model.config.num_hidden_layers + 1  # +1 for embedding layer
    H = model.config.hidden_size
    size_per_layer_gb = total_tokens * H * 4 / 1e9
    print(f"    total_tokens={total_tokens}  layers={n_layers}  H={H}")
    print(f"    → {size_per_layer_gb:.2f} GB/layer  {n_layers * size_per_layer_gb:.1f} GB total")

    # Step 2: open L file handles and write numpy headers up front.
    # Writing the correct shape in the header means each file is immediately a valid .npy
    # that np.load / mmap_mode='r' can open, even before all data is written.
    layer_fps = []
    for l in range(n_layers):
        fpath = out_dir / f"layer_{l:02d}_{label}{suffix}.npy"
        fp = open(str(fpath), "wb")
        nf_lib.write_array_header_1_0(fp, {
            "descr": "<f4",           # little-endian float32
            "fortran_order": False,
            "shape": (total_tokens, H),
        })
        layer_fps.append(fp)

    # Step 3: model inference — write each batch directly to layer files (no accumulation)
    for start in tqdm(range(0, len(texts), batch_size), desc="    batches", leave=False):
        batch_texts = texts[start : start + batch_size]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            max_length=max_len,
            truncation=True,
            padding=True,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        stacked = torch.stack(out.hidden_states, dim=0)  # [L, B, T, H]
        L, B, T, _ = stacked.shape

        for b in range(B):
            real_idxs = attention_mask[b].bool().nonzero(as_tuple=True)[0]
            # [L, n_real, H] float32 — must be float32: bfloat16→float16 overflows at deep layers
            chunk = stacked[:, b, real_idxs, :].cpu().float().numpy()
            for l in range(L):
                layer_fps[l].write(chunk[l].tobytes())  # append [n_real, H] raw bytes

        del stacked, out

    for fp in layer_fps:
        fp.close()

    return n_layers, H, counts_arr


def main():
    parser = argparse.ArgumentParser(description="Collect activation steering vectors")
    parser.add_argument("--model_name", required=True, choices=list(MODEL_PATHS.keys()))
    parser.add_argument("--mode", choices=["full", "sig"], default="full",
                        help="'full': encode entire function body (recommended); "
                             "'sig': encode function signature only (gives random probe accuracy)")
    parser.add_argument("--pooling", choices=["mean", "last", "token"], default="mean",
                        help="'mean': mean-pool over all token positions (recommended); "
                             "'last': use only the last token; "
                             "'token': keep all token hidden states (enables token-level probing)")
    parser.add_argument("--max_len", type=int, default=512,
                        help="Max token length per example")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Inference batch size (reduce if OOM)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override (e.g. 'cuda:1'); defaults to first available GPU")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Local path to model weights (overrides MODEL_PATHS lookup)")
    args = parser.parse_args()

    suffix = f"_{args.mode}_{args.pooling}"
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_path = args.model_dir or MODEL_PATHS[args.model_name]
    out_dir = OUT_ROOT / args.model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model:   {model_path}")
    print(f"Mode:    {args.mode}  Pooling: {args.pooling}  (suffix='{suffix}')")
    print(f"Device:  {device}")
    print(f"Out dir: {out_dir}")

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device
    )
    model.eval()

    print(f"Loading training pairs from {DATA_FILE} ...")
    secure_texts, insecure_texts = [], []
    with open(DATA_FILE) as f:
        for line in f:
            item = json.loads(line)
            if args.mode == "sig":
                secure_texts.append(extract_signature(item["func_src_after"], item["file_name"]))
                insecure_texts.append(extract_signature(item["func_src_before"], item["file_name"]))
            else:
                secure_texts.append(item["func_src_after"])
                insecure_texts.append(item["func_src_before"])
    N = len(secure_texts)
    print(f"  {N} pairs loaded")
    if args.mode == "sig":
        print(f"  Example signature:\n{secure_texts[0]}")

    if args.pooling == "token":
        print("Extracting secure token-level activations...")
        n_layers_plus, hidden_size, counts_sec = extract_token_activations(
            model, tokenizer, secure_texts, args.max_len, args.batch_size,
            device, out_dir, "secure", suffix
        )
        total_sec = int(counts_sec.sum())
        np.save(out_dir / f"token_counts_secure{suffix}.npy", counts_sec)

        print("Extracting insecure token-level activations...")
        _, _, counts_ins = extract_token_activations(
            model, tokenizer, insecure_texts, args.max_len, args.batch_size,
            device, out_dir, "insecure", suffix
        )
        total_ins = int(counts_ins.sum())
        np.save(out_dir / f"token_counts_insecure{suffix}.npy", counts_ins)

        # Compute steering vectors layer-by-layer (one layer file at a time, ~1.8 GB RAM)
        print("Computing steering vectors (layer-by-layer)...")
        steering = np.zeros((n_layers_plus, hidden_size), dtype=np.float32)
        for l in tqdm(range(n_layers_plus), desc="  steering layers", leave=False):
            arr_sec = np.load(out_dir / f"layer_{l:02d}_secure{suffix}.npy", mmap_mode="r")
            arr_ins = np.load(out_dir / f"layer_{l:02d}_insecure{suffix}.npy", mmap_mode="r")
            steering[l] = (
                np.asarray(arr_sec, dtype=np.float32).mean(axis=0)
                - np.asarray(arr_ins, dtype=np.float32).mean(axis=0)
            )
        np.save(out_dir / f"steering_vectors{suffix}.npy", steering)

        meta = {
            "model_name": args.model_name,
            "model_path": model_path,
            "mode": args.mode,
            "pooling": args.pooling,
            "n_samples": N,
            "total_tokens_secure": total_sec,
            "total_tokens_insecure": total_ins,
            "n_layers_plus1": n_layers_plus,
            "hidden_size": hidden_size,
            "max_len": args.max_len,
            "dtype": "float32",
            "layout": "per-layer files: layer_{l:02d}_{label}{suffix}.npy  [total_tokens, H]",
        }
        with open(out_dir / f"metadata{suffix}.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"\nSaved to {out_dir}/")
        print(f"  layer_00_secure{suffix}.npy ... layer_{n_layers_plus-1:02d}_secure{suffix}.npy")
        print(f"  layer_00_insecure{suffix}.npy ... layer_{n_layers_plus-1:02d}_insecure{suffix}.npy")
        print(f"  ({n_layers_plus * 2} files, {total_sec * hidden_size * 4 / 1e6:.0f} MB each for secure)")
        print(f"  steering_vectors{suffix}.npy  {steering.nbytes / 1e6:.1f} MB")
    else:
        print("Extracting secure activations...")
        acts_sec = extract_hidden_states(
            model, tokenizer, secure_texts, args.max_len, args.batch_size, device, args.pooling
        )
        print("Extracting insecure activations...")
        acts_ins = extract_hidden_states(
            model, tokenizer, insecure_texts, args.max_len, args.batch_size, device, args.pooling
        )

        n_layers_plus = acts_sec.shape[1]
        hidden_size = acts_sec.shape[2]
        print(f"  Shape: {acts_sec.shape}  (N={N}, layers+1={n_layers_plus}, H={hidden_size})")

        steering = acts_sec.mean(axis=0) - acts_ins.mean(axis=0)  # [n_layers+1, H]

        np.save(out_dir / f"activations_secure{suffix}.npy", acts_sec)
        np.save(out_dir / f"activations_insecure{suffix}.npy", acts_ins)
        np.save(out_dir / f"steering_vectors{suffix}.npy", steering)

        meta = {
            "model_name": args.model_name,
            "model_path": model_path,
            "mode": args.mode,
            "pooling": args.pooling,
            "n_samples": N,
            "n_layers_plus1": n_layers_plus,
            "hidden_size": hidden_size,
            "max_len": args.max_len,
        }
        with open(out_dir / f"metadata{suffix}.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"\nSaved to {out_dir}/")
        print(f"  activations_secure{suffix}.npy   {acts_sec.nbytes / 1e6:.0f} MB")
        print(f"  activations_insecure{suffix}.npy {acts_ins.nbytes / 1e6:.0f} MB")
        print(f"  steering_vectors{suffix}.npy     {steering.nbytes / 1e6:.1f} MB")
        print(f"  metadata{suffix}.json")

    print(f"\nNext step: python activation-steer/probe.py --model_name {args.model_name} --mode {args.mode} --pooling {args.pooling}")


if __name__ == "__main__":
    main()

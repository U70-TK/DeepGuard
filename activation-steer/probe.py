"""
Layer-wise linear probing to identify the best steering layer.

Loads activations collected by collect.py and trains a linear probe at each layer
(secure=1, insecure=0) with cross-validation.

--pooling mean/last  (sequence-level, 1440 examples):
  Standard k-fold CV with sklearn LogisticRegression.

--pooling token  (token-level):
  PyTorch linear probe (nn.Linear + BCEWithLogitsLoss) trained on GPU.
  Subsamples up to --max_tokens_per_func tokens per function (default 50).
  Uses GroupKFold so all tokens from one function stay in the same fold.
  Reports accuracy and Ppeak (mean probability assigned to correct class),
  matching the DeepGuard paper's Table 5 confidence metric.
  Saves the probe weight vector for the best layer as probe_weights_{suffix}.npy.

Output:
  activation-steer/data/<model_name>/probe_results_{suffix}.json
  activation-steer/data/<model_name>/probe_accuracy_{suffix}.png  (--plot)
  activation-steer/data/<model_name>/probe_weights_{suffix}.npy   (token mode)

Usage:
  python activation-steer/probe.py --model_name qwen2.5-3b
  python activation-steer/probe.py --model_name qwen2.5-3b --pooling token --plot
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

DATA_ROOT = Path(__file__).parent / "data"


# ---------------------------------------------------------------------------
# Sequence-level probing (mean / last pooling) — sklearn, CPU
# ---------------------------------------------------------------------------

def probe_sequence_level(acts_sec, acts_ins, n_folds, max_iter):
    N, L, H = acts_sec.shape
    y = np.array([1] * N + [0] * N)

    results = []
    best_layer, best_acc = 0, 0.0

    for layer in range(L):
        X = np.concatenate([acts_sec[:, layer, :], acts_ins[:, layer, :]], axis=0)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=max_iter, C=1.0, solver="lbfgs")),
        ])
        scores = cross_val_score(pipe, X, y, cv=n_folds, scoring="accuracy", n_jobs=-1)
        acc = float(scores.mean())
        std = float(scores.std())
        results.append({"layer": layer, "accuracy": round(acc, 4), "std": round(std, 4)})

        marker = "  ← best" if acc > best_acc else ""
        print(f"  layer {layer:3d}:  acc={acc:.4f} ± {std:.4f}{marker}")
        if acc > best_acc:
            best_acc = acc
            best_layer = layer

    return results, best_layer, best_acc


# ---------------------------------------------------------------------------
# Token-level probing — PyTorch linear probe, GPU
# ---------------------------------------------------------------------------

def _subsample_indices(counts, max_per_func, rng):
    parts = []
    offset = 0
    for cnt in counts:
        cnt = int(cnt)
        n = min(cnt, max_per_func)
        chosen = rng.choice(cnt, size=n, replace=False) if n < cnt else np.arange(cnt)
        parts.append(offset + chosen)
        offset += cnt
    return np.concatenate(parts)


def _train_linear_probe(X_tr, y_tr, X_te, y_te, device, n_steps=50):
    """Train logistic probe with L-BFGS + strong-Wolfe line search on GPU.

    Same algorithm as sklearn's lbfgs solver — converges monotonically, no LR to tune.
    L2 regularisation equivalent to sklearn's C=1.0: penalty = 0.5 * ||w||^2 / (C * n).
    """
    mu = X_tr.mean(0, keepdim=True)
    sigma = X_tr.std(0, keepdim=True).clamp(min=1e-8)
    X_tr = (X_tr - mu) / sigma
    X_te = (X_te - mu) / sigma

    n_tr = X_tr.shape[0]
    H    = X_tr.shape[1]
    probe = nn.Linear(H, 1, bias=True).to(device)
    nn.init.zeros_(probe.weight)
    nn.init.zeros_(probe.bias)

    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    opt = torch.optim.LBFGS(
        probe.parameters(),
        lr=1.0,
        max_iter=20,
        history_size=10,
        line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad()
        loss = criterion(probe(X_tr).squeeze(1), y_tr)
        loss = loss + 0.5 * (probe.weight ** 2).sum() / n_tr   # L2 reg (C=1)
        loss.backward()
        return loss

    probe.train()
    for _ in range(n_steps):
        opt.step(closure)

    probe.eval()
    with torch.no_grad():
        logits = probe(X_te).squeeze(1)
        proba  = torch.sigmoid(logits).cpu().numpy()
        preds  = (proba > 0.5).astype(int)
        y_np   = y_te.cpu().numpy().astype(int)
        acc    = float((preds == y_np).mean())
        correct_proba = np.where(y_np == 1, proba, 1.0 - proba)
        ppeak  = float(correct_proba.mean())

    w = (probe.weight.squeeze() / sigma.squeeze()).detach().cpu().numpy().astype(np.float32)
    return acc, ppeak, w, mu.squeeze().cpu().numpy(), sigma.squeeze().cpu().numpy()


def probe_token_level(data_dir, suffix, n_folds, max_iter, max_tokens_per_func,
                      device, n_steps):
    counts_sec = np.load(data_dir / f"token_counts_secure{suffix}.npy")
    counts_ins = np.load(data_dir / f"token_counts_insecure{suffix}.npy")

    # Read metadata for L and H (avoids loading any activation file just for shape)
    with open(data_dir / f"metadata{suffix}.json") as f:
        meta = json.load(f)
    L = meta["n_layers_plus1"]
    H = meta["hidden_size"]

    N = len(counts_sec)
    T_sec = int(counts_sec.sum())
    T_ins = int(counts_ins.sum())

    rng = np.random.RandomState(42)
    if max_tokens_per_func > 0:
        sec_idx = _subsample_indices(counts_sec, max_tokens_per_func, rng)
        ins_idx = _subsample_indices(counts_ins, max_tokens_per_func, rng)
    else:
        sec_idx = np.arange(T_sec)
        ins_idx = np.arange(T_ins)

    n_sec, n_ins = len(sec_idx), len(ins_idx)
    print(f"Token-level mode: {N} func pairs,  {T_sec} sec / {T_ins} ins raw tokens")
    print(f"After subsampling (max {max_tokens_per_func}/func): {n_sec} + {n_ins} = {n_sec+n_ins} total")
    print(f"Layers: {L}  GroupKFold: {n_folds}-fold  device: {device}  L-BFGS steps: {n_steps}\n")

    y = np.concatenate([np.ones(n_sec, dtype=np.float32), np.zeros(n_ins, dtype=np.float32)])

    groups_sec_full = np.repeat(np.arange(N, dtype=np.int32), counts_sec)
    groups_ins_full = np.repeat(np.arange(N, 2 * N, dtype=np.int32), counts_ins)
    groups = np.concatenate([groups_sec_full[sec_idx], groups_ins_full[ins_idx]])

    # Pre-compute fold splits (CPU, indices only)
    gkf = GroupKFold(n_splits=n_folds)
    fold_splits = list(gkf.split(y, y, groups=groups))

    results = []
    best_layer, best_acc, best_ppeak = 0, 0.0, 0.0
    best_layer_weights = None

    for layer in range(L):
        # Load per-layer files (each ~1.8 GB) — one at a time, no [L, T, H] monolith needed
        acts_sec = np.load(data_dir / f"layer_{layer:02d}_secure{suffix}.npy", mmap_mode="r")
        acts_ins = np.load(data_dir / f"layer_{layer:02d}_insecure{suffix}.npy", mmap_mode="r")
        X_sec_np = np.asarray(acts_sec, dtype=np.float32)[sec_idx]
        X_ins_np = np.asarray(acts_ins, dtype=np.float32)[ins_idx]
        X_np = np.concatenate([X_sec_np, X_ins_np], axis=0)

        # Push full layer data to GPU once; index into it per fold
        X_gpu = torch.tensor(X_np, device=device)
        y_gpu = torch.tensor(y, device=device)

        fold_accs, fold_ppeaks = [], []
        for train_idx, test_idx in fold_splits:
            train_idx_t = torch.tensor(train_idx, device=device)
            test_idx_t  = torch.tensor(test_idx,  device=device)
            acc, ppeak, _, _, _ = _train_linear_probe(
                X_gpu[train_idx_t], y_gpu[train_idx_t],
                X_gpu[test_idx_t],  y_gpu[test_idx_t],
                device, n_steps,
            )
            fold_accs.append(acc)
            fold_ppeaks.append(ppeak)

        acc = float(np.mean(fold_accs))
        std = float(np.std(fold_accs))
        ppeak = float(np.mean(fold_ppeaks))
        results.append({"layer": layer, "accuracy": round(acc, 4),
                        "std": round(std, 4), "ppeak": round(ppeak, 4)})

        marker = "  ← best" if acc > best_acc else ""
        print(f"  layer {layer:3d}:  acc={acc:.4f} ± {std:.4f}  ppeak={ppeak:.4f}{marker}")

        if acc > best_acc:
            best_acc = acc
            best_ppeak = ppeak
            best_layer = layer
            # Refit on all data to get probe weight vector
            _, _, w, _, _ = _train_linear_probe(
                X_gpu, y_gpu, X_gpu, y_gpu, device, n_steps,
            )
            best_layer_weights = w

        del X_gpu, y_gpu
        if device != "cpu":
            torch.cuda.empty_cache()

    return results, best_layer, best_acc, best_ppeak, best_layer_weights


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Layer-wise probe for steering layer selection")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--mode", choices=["full", "sig"], default="full")
    parser.add_argument("--pooling", choices=["mean", "last", "token"], default="mean",
                        help="'token' uses GPU linear probe + GroupKFold + Ppeak")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--max_iter", type=int, default=500,
                        help="Sequence-level only: max solver iterations")
    parser.add_argument("--max_tokens_per_func", type=int, default=50,
                        help="Token mode: subsample up to N tokens per function (0=all)")
    parser.add_argument("--n_steps", type=int, default=50,
                        help="Token mode: L-BFGS outer steps (each step runs up to 20 inner "
                             "Newton steps with strong-Wolfe line search)")
    parser.add_argument("--device", type=str, default=None,
                        help="Token mode: device override (default: cuda if available)")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    suffix = f"_{args.mode}_{args.pooling}"
    data_dir = DATA_ROOT / args.model_name

    print(f"Model:   {args.model_name}")
    print(f"Mode:    {args.mode}  Pooling: {args.pooling}  (suffix='{suffix}')")
    print(f"CV:      {args.n_folds}-fold\n")

    if args.pooling == "token":
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        results, best_layer, best_acc, best_ppeak, best_weights = probe_token_level(
            data_dir, suffix, args.n_folds, args.max_iter,
            args.max_tokens_per_func, device, args.n_steps,
        )
        print(f"\nBest layer: {best_layer}  (accuracy={best_acc:.4f}  ppeak={best_ppeak:.4f})")

        out = {
            "model_name": args.model_name,
            "mode": args.mode,
            "pooling": args.pooling,
            "best_layer": best_layer,
            "best_accuracy": round(best_acc, 4),
            "best_ppeak": round(best_ppeak, 4),
            "n_folds": args.n_folds,
            "max_tokens_per_func": args.max_tokens_per_func,
            "n_steps": args.n_steps,
            "layers": results,
        }
        if best_weights is not None:
            weights_path = data_dir / f"probe_weights{suffix}.npy"
            np.save(weights_path, best_weights)
            print(f"Probe weight vector → {weights_path}")

    else:
        acts_sec = np.load(data_dir / f"activations_secure{suffix}.npy")
        acts_ins = np.load(data_dir / f"activations_insecure{suffix}.npy")
        N, L, H = acts_sec.shape
        print(f"Samples: {N} pairs → {2*N} examples  Layers: {L}  H: {H}\n")

        results, best_layer, best_acc = probe_sequence_level(
            acts_sec, acts_ins, args.n_folds, args.max_iter
        )
        best_ppeak = None
        print(f"\nBest layer: {best_layer}  (accuracy={best_acc:.4f})")

        out = {
            "model_name": args.model_name,
            "mode": args.mode,
            "pooling": args.pooling,
            "best_layer": best_layer,
            "best_accuracy": round(best_acc, 4),
            "n_folds": args.n_folds,
            "layers": results,
        }

    out_path = data_dir / f"probe_results{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved → {out_path}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        accs  = [r["accuracy"] for r in results]
        stds  = [r["std"]      for r in results]
        xs    = list(range(len(results)))
        fig, ax1 = plt.subplots(figsize=(12, 4))
        if args.pooling == "token":
            ax2 = ax1.twinx()
            ppeaks = [r["ppeak"] for r in results]
            ax2.plot(xs, ppeaks, color="orange", linestyle="--", marker="s", ms=3, label="Ppeak")
            ax2.set_ylabel("Ppeak", color="orange")
            ax2.tick_params(axis="y", labelcolor="orange")
            ax2.legend(loc="upper right")
        ax1.plot(xs, accs, marker="o", ms=3, label="accuracy")
        ax1.fill_between(xs, [a-s for a,s in zip(accs,stds)],
                             [a+s for a,s in zip(accs,stds)], alpha=0.2)
        ax1.axvline(best_layer, color="red", linestyle="--", label=f"best={best_layer}")
        ax1.set_xlabel("Layer")
        ax1.set_ylabel("Accuracy (CV)")
        ax1.set_title(f"Probe accuracy — {args.model_name} ({args.mode}/{args.pooling})")
        ax1.legend(loc="upper left")
        plt.tight_layout()
        plot_path = data_dir / f"probe_accuracy{suffix}.png"
        plt.savefig(plot_path, dpi=150)
        print(f"Plot → {plot_path}")


if __name__ == "__main__":
    main()

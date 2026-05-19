"""
Evaluate a base model with activation steering applied at inference time.

Registers a forward hook on transformer layer L that adds:
    alpha * steering_vector[L]
to the residual stream hidden states at every generation step.

The steering vector is loaded from activation-steer/data/<model_name>/steering_vectors.npy
(computed by collect.py as mean_secure - mean_insecure).

The evaluation pipeline is identical to runs/sec_eval.py, so results land in
experiments/sec_eval/<output_name>/ and are directly comparable.

Usage:
  cd runs/
  python ../activation-steer/eval.py \
      --model_name qwen2.5-3b \
      --layer 18 \
      --alpha 20.0 \
      --output_name act-steer-qwen2.5-3b-L18-a20 \
      --eval_type base

  # Use probe_results.json to auto-select best layer:
  python ../activation-steer/eval.py \
      --model_name qwen2.5-3b \
      --alpha 20.0 \
      --output_name act-steer-qwen2.5-3b-auto-a20 \
      --eval_type base

  # Sweep multiple alpha values:
  for alpha in 5 10 20 40; do
    python ../activation-steer/eval.py \
        --model_name qwen2.5-3b --layer 18 --alpha $alpha \
        --output_name act-steer-qwen2.5-3b-L18-a${alpha} --eval_type base
  done
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Allow running from the runs/ directory (same as sec_eval.py)
_HERE = Path(__file__).parent
_PROJ = _HERE.parent
sys.path.insert(0, str(_PROJ))

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from sven.evaler import LMEvaler
from sven.utils import set_seed, set_logging, set_devices
from sven.constant import MODEL_DIRS, NOT_TRAINED, CWES_TRAINED

# Import the shared evaluation loop from sec_eval
_RUNS_DIR = _PROJ / "runs"
sys.path.insert(0, str(_RUNS_DIR))
from sec_eval import eval_all

MODEL_PATHS = {
    "deepseek-1.3b": "deepseek-ai/deepseek-coder-1.3b-base",
    "deepseek-6.7b": "deepseek-ai/deepseek-coder-6.7b-base",
    "qwen2.5-7b": "Qwen/Qwen2.5-Coder-7B",
    "qwen2.5-3b": "Qwen/Qwen2.5-Coder-3B",
    "seedcoder-8b": "ByteDance-Seed/Seed-Coder-8B-Base",
}

ACT_DATA_ROOT = _HERE / "data"


class ActSteerEvaler(LMEvaler):
    """LMEvaler with a residual-stream activation steering hook."""

    def __init__(self, args, steering_vector: torch.Tensor, layer: int, alpha: float):
        self._steering_vector = steering_vector  # [H]
        self._layer = layer
        self._alpha = alpha
        self._hook_handle = None
        super().__init__(args)  # calls load_model()

    def load_model(self):
        super().load_model()
        self._register_hook()

    def _register_hook(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()

        sv = self._steering_vector.to(self.input_device).to(self.model.dtype)
        alpha = self._alpha

        def hook(module, input, output):
            # output is (hidden_states, ...) for transformer decoder layers
            if isinstance(output, tuple):
                h = output[0] + alpha * sv
                return (h,) + output[1:]
            return output + alpha * sv

        # Qwen2/LLaMA: model.model.layers[L]
        target = self.model.model.layers[self._layer]
        self._hook_handle = target.register_forward_hook(hook)
        print(f"  Hook registered: model.model.layers[{self._layer}]  alpha={alpha}")

    def __del__(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()


def load_steering_vector(model_name: str, layer: int, mode: str, pooling: str) -> torch.Tensor:
    suffix = f"_{mode}_{pooling}"
    sv_path = ACT_DATA_ROOT / model_name / f"steering_vectors{suffix}.npy"
    if not sv_path.exists():
        raise FileNotFoundError(
            f"Steering vectors not found at {sv_path}.\n"
            f"Run: python activation-steer/collect.py --model_name {model_name} --mode {mode} --pooling {pooling}"
        )
    sv_all = np.load(sv_path)  # [n_layers+1, H]
    return torch.tensor(sv_all[layer], dtype=torch.float32)


def get_best_layer(model_name: str, mode: str, pooling: str) -> int:
    suffix = f"_{mode}_{pooling}"
    probe_path = ACT_DATA_ROOT / model_name / f"probe_results{suffix}.json"
    if not probe_path.exists():
        raise FileNotFoundError(
            f"Probe results not found at {probe_path}.\n"
            f"Run: python activation-steer/probe.py --model_name {model_name} --mode {mode} --pooling {pooling}"
        )
    with open(probe_path) as f:
        return json.load(f)["best_layer"]


def get_args():
    parser = argparse.ArgumentParser(
        description="Activation steering evaluation (compatible with sec_eval.py pipeline)"
    )
    parser.add_argument("--output_name", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True, choices=list(MODEL_PATHS.keys()))
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Local path to base model weights; overrides MODEL_PATHS lookup")

    parser.add_argument("--mode", choices=["full", "sig"], default="full",
                        help="Must match the mode used in collect.py and probe.py")
    parser.add_argument("--pooling", choices=["mean", "last"], default="mean",
                        help="Must match the pooling used in collect.py and probe.py")
    parser.add_argument("--layer", type=int, default=None,
                        help="Layer index to apply steering hook (0=embedding). "
                             "If omitted, uses best_layer from probe_results.json")
    parser.add_argument("--alpha", type=float, required=True,
                        help="Steering coefficient (positive = steer toward secure direction)")

    parser.add_argument("--eval_type", type=str, choices=["base", "untrain", "prompts"], default="base")
    parser.add_argument("--vul_type", type=str, default=None)

    parser.add_argument("--data_dir", type=str, default="../data_eval/sec_eval")
    parser.add_argument("--output_dir", type=str, default="../experiments/sec_eval")

    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--num_samples_per_gen", type=int, default=20)
    parser.add_argument("--temp", type=float, default=0.1)
    parser.add_argument("--max_gen_len", type=int, default=300)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=1)

    # Unused by LMEvaler but required by some sven utils
    parser.add_argument("--base_model", type=str, default="")
    parser.add_argument("--sec_model", type=str, default="")
    parser.add_argument("--exp_temp", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.3)

    args = parser.parse_args()

    # Resolve model dir
    if args.model_dir is None:
        args.model_dir = MODEL_PATHS[args.model_name]
    if args.model_dir in MODEL_DIRS:
        args.model_dir = MODEL_DIRS[args.model_dir]

    args.output_dir = os.path.join(args.output_dir, args.output_name, args.eval_type)
    args.data_dir = os.path.join(args.data_dir, args.eval_type)

    return args


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_logging(args, os.path.join(args.output_dir, "eval.log"))
    set_devices(args)
    set_seed(args)
    args.logger.info(f"args: {args}")

    # Resolve layer
    layer = args.layer
    if layer is None:
        layer = get_best_layer(args.model_name, args.mode, args.pooling)
        print(f"Auto-selected best layer: {layer}")

    # Load steering vector for chosen layer
    sv = load_steering_vector(args.model_name, layer, args.mode, args.pooling)

    print(f"Steering: layer={layer}, alpha={args.alpha}, |sv|={sv.norm():.4f}")
    evaler = ActSteerEvaler(args, sv, layer, args.alpha)

    if args.vul_type is not None:
        vul_types = [args.vul_type]
    elif args.eval_type == "untrain":
        vul_types = NOT_TRAINED
    else:
        vul_types = CWES_TRAINED

    eval_all(args, evaler, vul_types)


if __name__ == "__main__":
    main()

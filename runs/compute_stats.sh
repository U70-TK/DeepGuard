#!/bin/bash
#SBATCH --job-name=compute-stats
#SBATCH --account=def-m2nagapp
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%A.out
#SBATCH --error=logs/%x-%A.err

set -e
mkdir -p logs

module load gcc

cd /scratch/tkwang/DeepGuard
source .venv/bin/activate

cd runs

for name in \
    sec-eval-qwen2.5-3b-weight-steer-secure-v3 \
    sec-eval-qwen2.5-3b-weight-steer-insecure-v3 \
    sec-eval-qwen2.5-3b-weight-steer-secure-v5 \
    sec-eval-qwen2.5-3b-weight-steer-insecure-v5 \
    act-steer-qwen2.5-3b-full-mean-L6-a20; do
    echo "=== $name ==="
    python correctness_eval.py --paths "../experiments/sec_eval/${name}" --do_eval --num_seeds 1 --eval_type base 2>&1 | tail -2
    python new_stats.py --paths "../experiments/sec_eval/${name}" --eval_type base 2>&1
    echo ""
done

#!/bin/bash
#SBATCH --job-name=cosmos3-g1-weights
#SBATCH --partition=h100n2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/raid/user_marcospaulo/Psi0/logs/cosmos3-g1-weights-%j.out
#SBATCH --error=/raid/user_marcospaulo/Psi0/logs/cosmos3-g1-weights-%j.err

set -euo pipefail
REPO=/raid/user_marcospaulo/Psi0
CF=$REPO/third_party/cosmos-framework

set -a; source "$REPO/secrets/psi0.env"; set +a
export HF_HOME=$REPO/cache/huggingface

"$CF/.venv/bin/python" - <<EOF
from huggingface_hub import snapshot_download, hf_hub_download

p = snapshot_download("nvidia/Cosmos3-Nano")
print("Cosmos3-Nano ->", p)
v = hf_hub_download(
    "Wan-AI/Wan2.2-TI2V-5B", "Wan2.2_VAE.pth",
    local_dir="$CF/psi0_g1/checkpoints/wan22_vae",
)
print("VAE ->", v)
EOF
echo "WEIGHTS OK"

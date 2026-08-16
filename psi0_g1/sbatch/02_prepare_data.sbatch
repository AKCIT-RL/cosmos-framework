#!/bin/bash
#SBATCH --job-name=cosmos3-g1-data
#SBATCH --partition=h100n2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/raid/user_marcospaulo/Psi0/logs/cosmos3-g1-data-%j.out
#SBATCH --error=/raid/user_marcospaulo/Psi0/logs/cosmos3-g1-data-%j.err

set -euo pipefail
REPO=/raid/user_marcospaulo/Psi0
CF=$REPO/third_party/cosmos-framework
# torchcodec dlopens CUDA NPP libs shipped as pip wheels
export LD_LIBRARY_PATH=$CF/.venv/lib/python3.13/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}

"$CF/.venv/bin/python" "$CF/psi0_g1/prepare_g1_v3_subset.py" \
    --src "$REPO/data/G1ToteMix-psi0" \
    --dst "$REPO/data/G1ToteMix-cosmos3-v3-smoke" \
    --episodes 20 \
    --stats-out "$CF/cosmos_framework/data/generator/action/normalizer_stats/g1_wholebody_stats.json"

# smoke-load one sample through the dataset class (CPU; no GPU used)
cd "$CF"
G1_ROOT="$REPO/data/G1ToteMix-cosmos3-v3-smoke" .venv/bin/python - <<'EOF'
import os
from cosmos_framework.data.generator.action.datasets.g1_lerobot_dataset import G1WholebodyLeRobotDataset

ds = G1WholebodyLeRobotDataset(root=os.environ["G1_ROOT"], chunk_length=16)
print("len:", len(ds))
s = ds[0]
print("video:", tuple(s["video"].shape), s["video"].dtype)
print("action:", tuple(s["action"].shape), "domain_id:", int(s["domain_id"]))
print("caption:", s["ai_caption"][:80])
assert s["action"].shape == (16, 36)
print("DATA OK")
EOF

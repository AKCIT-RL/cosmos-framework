#!/bin/bash
# Submits the full Cosmos3-Nano G1 smoke pipeline with Slurm dependencies.
set -euo pipefail
SB=/raid/user_marcospaulo/Psi0/third_party/cosmos-framework/psi0_g1/sbatch

j1=$(sbatch --parsable "$SB/01_env_setup.sbatch")
j2=$(sbatch --parsable --dependency=afterok:$j1 "$SB/02_prepare_data.sbatch")
j3=$(sbatch --parsable --dependency=afterok:$j1 "$SB/03_download_weights.sbatch")
j4=$(sbatch --parsable --dependency=afterok:$j2,afterok:$j3 "$SB/04_smoke_sft.sbatch")
echo "env=$j1 data=$j2 weights=$j3 smoke=$j4"

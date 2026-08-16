#!/bin/bash
#SBATCH --job-name=cosmos3-g1-simple
#SBATCH --partition=h100n2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=/raid/user_marcospaulo/Psi0/logs/cosmos3-g1-simple-%j.out
#SBATCH --error=/raid/user_marcospaulo/Psi0/logs/cosmos3-g1-simple-%j.err

# Closed-loop smoke: sobe serve_cosmos_g1.py (venv do cosmos-framework, sem
# container) e roda o cliente SIMPLE eval-decoupled-wbc via apptainer com o SIF
# containers/simple-wmo-599d6c0.sif (docker foi removido do no dgx-H100-02).

set -euo pipefail
REPO=/raid/user_marcospaulo/Psi0
CF=$REPO/third_party/cosmos-framework
SIMPLE=$REPO/third_party/SIMPLE

# --- env (padrao 05_full_sft_1gpu.sbatch) ---
export LD_LIBRARY_PATH=$CF/.venv/lib/python3.13/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
export PATH=$HOME/.local/bin:$PATH
export UV_CACHE_DIR=$REPO/cache/uv
export UV_PYTHON_INSTALL_DIR=$REPO/cache/uv-python
set -a; source "$REPO/secrets/psi0.env"; set +a
export WANDB_PROJECT=cosmos3-g1  # projeto W&B dedicado ao Cosmos
export HF_HOME=$REPO/cache/huggingface
export WANDB_DIR=$REPO/cache/wandb
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

# --- parametros do eval (padrao run_wmo_totes_20260807.sh) ---
TASK=simple/G1WholebodyLocomotionPickTotesShelfToTableTeleop-v0
POLICY=psi0_decoupled_wbc            # cliente HTTP generico; server e Cosmos3
SIMPLE_COMMIT=${SIMPLE_COMMIT:-599d6c0e3364b31095f1a59fdbfa4ea507ae90c9}
SIF=${SIF:-$REPO/containers/simple-wmo-599d6c0.sif}
PORT=${PORT:-22085}
EPISODES=${EPISODES:-2}              # smoke
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-2400}
DATA_ROOT=$REPO/data
DATASET_RELATIVE_PATH=G1WholebodyLocomotionPickTotesShelfToTableTeleop-psi0
CONTAINER_DATASET=/workspace/SIMPLE/data/$DATASET_RELATIVE_PATH
export COSMOS_CHECKPOINT_PATH=${COSMOS_CHECKPOINT_PATH:-$CF/psi0_g1/outputs/train/cosmos3_action_g1/action_sft/action_policy_g1_nano_full_1gpu/checkpoints/iter_000020000}
COSMOS_CHUNK_LENGTH=${COSMOS_CHUNK_LENGTH:-24}
ACTION_EXEC_HORIZON=${ACTION_EXEC_HORIZON:-24}
# 2 GPUs do job: primeira para o servidor Cosmos, segunda para o Isaac/mujoco
ALL_GPUS=${CUDA_VISIBLE_DEVICES:-0,1}
SERVER_GPU=${ALL_GPUS%%,*}
CLIENT_GPU=${ALL_GPUS##*,}
LDP=/isaac-sim/extscache/omni.sensors.nv.camera-0.20.1-coreapi+lx64.r/bin
ISAAC_BACKGROUND_USD=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Environments/Simple_Warehouse/warehouse.usd

RUN_ID=${RUN_ID:-$(date -u '+%Y%m%dT%H%M%SZ')}
RESULT_DIR=$REPO/eval/results/cosmos3-g1-$(date -u '+%Y%m%d')/$RUN_ID
LOGS=$RESULT_DIR/logs
mkdir -p "$LOGS"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/orchestrator.log"; }
fail() { log "ERRO: $*"; exit 1; }

SERVER_PID=
stop_server() {
    if [[ -n ${SERVER_PID:-} ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=
}
cleanup() { stop_server; }
trap cleanup EXIT INT TERM

# --- pre-checks ---
[[ $(git -C "$SIMPLE" rev-parse HEAD) == "$SIMPLE_COMMIT" ]] || fail "commit SIMPLE diferente do esperado"
[[ -d $COSMOS_CHECKPOINT_PATH ]] || fail "checkpoint Cosmos ausente: $COSMOS_CHECKPOINT_PATH"
[[ -f ${COSMOS_CHECKPOINT_PATH%/checkpoints/*}/config.yaml ]] || fail "config.yaml do treino ausente"
[[ -d $DATA_ROOT/$DATASET_RELATIVE_PATH ]] || fail "dataset SIMPLE ausente"
[[ -f $SIF ]] || fail "SIF ausente: $SIF"
module load apptainer 2>/dev/null || true
command -v apptainer >/dev/null || fail "apptainer indisponivel"
mkdir -p "$REPO/cache/simple/home" "$REPO/cache/simple/uv-cache" "$REPO/cache/simple/tmp"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

# --- 1) policy server Cosmos3 (background, venv do framework, sem container) ---
log "Subindo serve_cosmos_g1.py na porta $PORT (gpu=$SERVER_GPU chunk=$COSMOS_CHUNK_LENGTH exec=$ACTION_EXEC_HORIZON)"
(
    cd "$CF"
    export CUDA_VISIBLE_DEVICES=$SERVER_GPU
    exec .venv/bin/python psi0_g1/serve_cosmos_g1.py \
        --host=0.0.0.0 --port="$PORT" \
        --checkpoint-path="$COSMOS_CHECKPOINT_PATH" \
        --chunk-length="$COSMOS_CHUNK_LENGTH" \
        --action-exec-horizon="$ACTION_EXEC_HORIZON"
) >> "$LOGS/policy-server.log" 2>&1 &
SERVER_PID=$!

log "Aguardando /health (ate 60min: load do modelo ~16B + warmup)"
server_ok=0
for _ in $(seq 1 360); do
    if curl -fsS "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"status":"ok"'; then
        server_ok=1; break
    fi
    kill -0 "$SERVER_PID" 2>/dev/null || fail "policy server morreu; veja $LOGS/policy-server.log"
    sleep 10
done
[[ $server_ok == 1 ]] || fail "policy server nao respondeu /health"
log "Server saudavel"

# --- 2) cliente SIMPLE (apptainer, mesmos flags do run_wmo) ---
log "Rodando eval-decoupled-wbc via apptainer: $EPISODES episodios (gpu=$CLIENT_GPU)"
(
    apptainer exec --nv --cleanenv --writable-tmpfs \
        --home "$REPO/cache/simple/home:$HOME" \
        --bind "$DATA_ROOT:/workspace/SIMPLE/data" \
        --bind "$REPO/eval:/workspace/eval" \
        --bind "$REPO/cache/simple/uv-cache:/workspace/.uv-cache" \
        --bind "$REPO/cache/simple/tmp:/tmp" \
        --pwd /workspace/SIMPLE \
        --env CUDA_VISIBLE_DEVICES="$CLIENT_GPU" \
        --env OMNI_KIT_ACCEPT_EULA=Y \
        --env MUJOCO_GL=egl \
        --env UV_CACHE_DIR=/workspace/.uv-cache \
        --env LD_LIBRARY_PATH="/usr/local/cuda-12.8/targets/x86_64-linux/lib:/isaac-sim/extscache/omni.kit.streamsdk.plugins-6.1.7+106.2.0.lx64.r/bin:$LDP" \
        --env SIMPLE_ISAAC_BACKGROUND_USD="$ISAAC_BACKGROUND_USD" \
        "$SIF" \
        uv run --no-sync eval-decoupled-wbc \
        "$TASK" "$POLICY" level-0 \
        --host=localhost --port="$PORT" \
        --sim-mode=mujoco_isaac --headless \
        --max-episode-steps="$MAX_EPISODE_STEPS" \
        --data-format=lerobot \
        --data-dir="$CONTAINER_DATASET" \
        --eval-dir="/workspace/eval/results/cosmos3-g1-$(date -u '+%Y%m%d')/$RUN_ID/smoke" \
        --num-episodes="$EPISODES" --episode-start=0 --num-workers=1
) >> "$LOGS/eval-smoke.log" 2>&1 || fail "eval falhou; veja $LOGS/eval-smoke.log"

# --- 3) metadata ---
sif_sha=$(sha256sum "$SIF" | awk '{print $1}')
main_commit=$(git -C "$REPO" rev-parse HEAD)
cf_commit=$(git -C "$CF" rev-parse HEAD)
cat > "$RESULT_DIR/metadata.json" <<META
{
  "evaluation_scope": "closed-loop smoke; Cosmos3-Nano G1 policy server",
  "task": "$TASK",
  "policy_client": "$POLICY",
  "policy_server": "psi0_g1/serve_cosmos_g1.py",
  "checkpoint": "$COSMOS_CHECKPOINT_PATH",
  "chunk_length": $COSMOS_CHUNK_LENGTH,
  "action_exec_horizon": $ACTION_EXEC_HORIZON,
  "rtc": false,
  "episodes": $EPISODES,
  "max_episode_steps": $MAX_EPISODE_STEPS,
  "main_repository_commit": "$main_commit",
  "cosmos_framework_commit": "$cf_commit",
  "simple_commit": "$SIMPLE_COMMIT",
  "apptainer_sif": "$SIF",
  "apptainer_sif_sha256": "$sif_sha",
  "isaac_background_usd": "$ISAAC_BACKGROUND_USD",
  "policy_port": $PORT,
  "slurm_job_id": "${SLURM_JOB_ID:-unknown}",
  "result_directory": "$RESULT_DIR"
}
META

stats=$RESULT_DIR/smoke/eval_stats.txt
episodes=$(awk '/^run:/{count=0} /^episode_/{count++} END{print count+0}' "$stats" 2>/dev/null || echo 0)
successes=$(awk '/^run:/{count=0} /^episode_/{if ($2=="True") count++} END{print count+0}' "$stats" 2>/dev/null || echo 0)
log "Smoke concluido: $successes/$episodes sucessos. Resultados em $RESULT_DIR"
echo "SERVE+EVAL OK"

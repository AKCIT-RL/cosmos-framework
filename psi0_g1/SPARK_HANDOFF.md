# Handoff — Cosmos3-Nano action policy G1 na DGX Spark

Contexto para o agente que vai trabalhar na DGX Spark (GB10, 128GB unificada,
aarch64, CUDA 13). Estado em 2026-08-16. Fonte de verdade do historico:
`memory/cosmos3_g1_smoke_run.md` no repo Psi0 (branch `dev/marcos`).

## O que ja foi validado no cluster H100 (NAO refazer)

- **Arquitetura validada**: smoke SFT completo (job 30169, 16/08): 40/40
  iters a ~0.38s/iter em 1x H100 80GB, loss 13.4 -> 12.4, checkpoint
  iter_000000040 salvo (29G, DCP). W&B:
  https://wandb.ai/ih-akcit/psi-h100/runs/0ou0lmtf
- Dados: `data/G1ToteMix-psi0` (LeRobot v2.1) convertido p/ layout v3 em
  `data/G1ToteMix-cosmos3-v3-smoke` (subset 20 eps; 19 usaveis, 24.227
  janelas). Script: `psi0_g1/prepare_g1_v3_subset.py`.
- Acao G1 = **36D** (feature `action` do meta/info.json), embodiment
  `g1_wholebody` registrado como **domain_id 24**, raw_action_dim 36,
  padding automatico ate max_action_dim=64.
- Stats de normalizacao: `cosmos_framework/data/generator/action/`
  `normalizer_stats/g1_wholebody_stats.json` (mean/std/min/max/q01/q99).
- Vista unica ego: `camera_mode="image"` (third_person_view). fps 50.
- Decode de video: backend **pyav** (torchcodec falha nos nos sem FFmpeg de
  sistema; em aarch64 idem — manter pyav).
- chunk_length=16 (deve ser divisivel por 4).

## Codigo (branch `dev/marcos` do fork AKCIT-RL do cosmos-framework)

Mudancas minimas no framework:
- `cosmos_framework/data/generator/action/utils/domain_utils.py` — registro
  g1_wholebody (domain 24, 36D).
- `cosmos_framework/data/generator/action/datasets/g1_lerobot_dataset.py` —
  dataset class (novo arquivo, espelha o LIBERO; pyav backend).
- `cosmos_framework/data/generator/action/datasets/action_sft_dataset.py` —
  factory `get_action_g1_sft_dataset`.
- `cosmos_framework/configs/base/experiment/action/posttrain_config/`
  `action_policy_g1_nano.py` — experimento `action_policy_g1_nano`
  (fsdp_master_dtype=bfloat16 aplicado aqui; reavaliar p/ SFT longo).
- `cosmos_framework/configs/base/config.py` — registro do experimento.
- `psi0_g1/` — scripts de preparo, TOMLs e sbatch (sbatch e especifico do
  cluster Slurm; na Spark rode torchrun direto).

## Licoes de VRAM/ambiente (evite repetir nossos erros)

1. Modelo ~16B (Qwen3-VL-8B + diffusion expert 8B): pesos bf16 ~32GB.
2. **EMA** guarda copia fp32 do gen-pathway (~32GB) -> desligada no smoke
   (`[model.ema] enabled=false`). Na Spark (128GB) pode religar.
3. **FusedAdam** cria master fp32 + m + v (12 bytes/param). moe_gen treinavel
   => ~96GB so de otimizador. No smoke congelamos moe_gen via
   `[optimizer].keys_to_select` (so adapters de acao treinam). Para SFT
   completo com moe_gen: precisa de 4-8 GPUs H100 ou aceitar so-adapters.
4. `uv` precisa estar no PATH (checkpoint_db chama `uv run hf download`).
5. LD_LIBRARY_PATH deve incluir `.venv/.../nvidia/cu13/lib` se usar torchcodec
   (na pratica: use pyav e esqueca).
6. Na Spark (aarch64): recriar venv com `uv sync` (extra cu130); wheels ARM
   sbsa existem p/ torch 2.10. Risco principal: flash-attn/fmha em sm_121 —
   se falhar, teste `[model.activation_checkpointing] save_ops_regex=[]` e/ou
   atencao eager como fallback.

## Pesos e dados necessarios na Spark (tudo no HF, use HF_TOKEN do projeto)

- Dataset v3 pronto (77M): `agentereal/G1ToteMix-cosmos3-v3-smoke`
  (dataset privado; baixe para `data/G1ToteMix-cosmos3-v3-smoke`).
- Checkpoint do smoke treinado (29G, DCP): `agentereal/cosmos3-nano-g1-smoke`
  (path `iter_000000040`) — use direto na inferencia offline, sem retreinar.
- Pesos base `nvidia/Cosmos3-Nano` (HF, sem gate, ~33GB) + `Wan2.2_VAE.pth` —
  necessarios p/ tokenizer/VAE e p/ novos treinos. Converta p/ DCP:
  `python -m cosmos_framework.scripts.convert_model_to_dcp -o <out> --checkpoint-path Cosmos3-Nano`
- Para SFT maior: dataset completo v2.1 `agentereal/G1ToteMix-psi0` no HF
  (308 eps) + `psi0_g1/prepare_g1_v3_subset.py` (ajuste `--num-episodes`).

## O que falta fazer (ordem)

1. **Inferencia offline**: acoes preditas vs ground truth no subset (L1 por
   grupo de juntas). Template: `cosmos_framework/inference/action.py` e
   `scripts/action_policy_server_libero.py` (adaptar p/ g1_wholebody).
2. **SFT maior**: mais episodios (converter mais do v2.1), mais iters,
   avaliar religar moe_gen/EMA conforme memoria disponivel.
3. **Closed-loop**: policy server G1 + SIMPLE feat/weg_wmo, task
   `simple/G1WholebodyLocomotionPickTotesShelfToTableTeleop-v0` (isso exige o
   cluster — SIMPLE/containers estao la).
4. Comparar com baseline Psi0 (AKCITWMOPOC/psi0-g1totemix) na mesma task.

## Convencoes

- W&B: entity/projeto ih-akcit/psi-h100, um run por experimento, config+commit
  registrados. Segredos via env file local (nunca em codigo/commit).
- Decisoes autonomas: registre cada uma com justificativa no md de memoria.
- Diffs no framework: minimos e isolados; novos arquivos > edicao de existentes.

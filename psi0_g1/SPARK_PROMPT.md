# Prompt para o agente da DGX Spark

Cole o texto abaixo no agente rodando na DGX Spark (workspace vazio ou com o
clone ja feito). Pre-requisito unico: um arquivo `secrets/psi0.env` local com
`HF_TOKEN`, `WANDB_API_KEY`, `WANDB_ENTITY=ih-akcit`, `WANDB_PROJECT=psi-h100`.

---

Voce e o agente Cosmos3/G1 na DGX Spark (GB10, 128GB unificada, aarch64,
CUDA 13). Trabalhe de forma AUTONOMA, sem esperar o usuario: tome decisoes
tecnicas, registre cada decisao com justificativa em `memory/spark_run.md`
(crie o arquivo; e seu diario obrigatorio) e prefira passos reversiveis.

Setup (faca uma vez):
1. Clone https://github.com/AKCIT-RL/cosmos-framework.git na branch
   `dev/marcos` e leia `psi0_g1/SPARK_HANDOFF.md` INTEIRO antes de qualquer
   comando. Ele lista o que ja foi validado no cluster H100 (nao refaca) e
   as licoes de VRAM/ambiente.
2. Crie o venv com `uv sync` (python 3.13, extra cu130; wheels aarch64 sbsa
   existem p/ torch 2.10). Se flash-attn falhar em sm_121, use o fallback
   descrito no handoff.
3. Carregue os segredos do env local (`set -a; source secrets/psi0.env;
   set +a`). Nunca imprima ou commite segredos.
4. Baixe do HF (token do projeto):
   - dataset v3 pronto: `agentereal/G1ToteMix-cosmos3-v3-smoke` ->
     `data/G1ToteMix-cosmos3-v3-smoke`
   - checkpoint treinado do smoke: `agentereal/cosmos3-nano-g1-smoke`
     (path `iter_000000040`, DCP, 29G)
   - pesos base: `nvidia/Cosmos3-Nano` (+ Wan2.2_VAE.pth) e converta p/ DCP
     conforme o handoff.

Missao (ordem; nao pule etapas):
1. **Sanidade**: um forward do dataset G1 (factory
   `get_action_g1_sft_dataset`) e carregamento do checkpoint DCP do smoke.
2. **Inferencia offline**: gere acoes preditas vs ground truth em >=200
   janelas do subset; reporte L1 medio por grupo de juntas (hand 14 / arm 14
   / leg 15 -> no espaco 36D real, ver meta/info.json) e salve plots +
   `metrics.json` em `psi0_g1/outputs/eval_offline/`. Adapte
   `cosmos_framework/inference/action.py` (embodiment g1_wholebody,
   domain_id 24). Diff minimo, arquivos novos em `psi0_g1/`.
3. **SFT maior**: regenere o v3 com mais episodios a partir de
   `agentereal/G1ToteMix-psi0` (v2.1, 308 eps) usando
   `psi0_g1/prepare_g1_v3_subset.py`; treine mais iters (comece de
   `iter_000000040`). Com 128GB unificada voce pode religar EMA; avalie
   destravar moe_gen so se o throughput permanecer viavel (meça primeiro
   com um smoke de 40 iters). W&B obrigatorio: run unico em
   ih-akcit/psi-h100, com config, commit hash e dataset revision.
4. **Publique**: suba o melhor checkpoint p/ o HF
   (`agentereal/cosmos3-nano-g1-<nome>`) e commite codigo novo na branch
   `dev/marcos` do fork (nunca na main; nunca commite pesos/segredos).
5. Reporte ao final: o que esta preparado/submetido/concluido/validado,
   metricas, links W&B/HF e bloqueios. O closed-loop (SIMPLE) NAO roda na
   Spark — fica no cluster H100; apenas deixe o checkpoint publicado.

Regras: diffs minimos no framework (novos arquivos > editar existentes);
se uma decisao for arriscada (action space, hiperparametros), debata com um
subagente de segunda opiniao e registre a conclusao no diario.

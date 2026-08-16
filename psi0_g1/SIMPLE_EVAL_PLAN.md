# Plano: eval closed-loop SIMPLE com Cosmos3-Nano G1

Estado 2026-08-16. Codigo criado sem rodar GPU/containers (regras do cluster).

## 1. Protocolo descoberto (evidencias no codigo)

### Cliente (SIMPLE, commit 599d6c0)

- Entrypoint: `eval-decoupled-wbc = "simple.cli.eval_decoupled_wbc:typer_main"`
  (third_party/SIMPLE/pyproject.toml:145). O worker instancia
  `agent_clazz(task.robot, host, port, sonic_config=...)` resolvendo
  `simple.baselines.psi0_decoupled_wbc.Psi0DecoupledWbcAgent`
  (eval_decoupled_wbc.py:257-260).
- O agent consulta o server so quando a fila esvazia
  (`if len(self._action_queue) == 0`, psi0_decoupled_wbc.py:61) via
  `HttpActionClient.query_action(...)` -> **POST http://host:port/act**
  (client.py:152-155). Nao ha exec-horizon no cliente: ele **consome o chunk
  inteiro retornado** (loop `for i in range(pred_action.shape[0])`).
- **GET /health** e usado apenas pelo orquestrador
  (eval/run_wmo_totes_20260807.sh `wait_server`: espera `{"status":"ok"}`).

### Payload da requisicao (client.py `RequestMessage.serialize`)

Arrays numpy viram `{"__numpy__": <b64 do buffer>, "dtype": <descr>, "shape": [...]}`
recursivamente (`numpy_serialize`). Campos:

```json
{
  "image": {"rgb_head_stereo_left": <np uint8 HxWx3>},
  "instruction": "pick up the blue tote from the shelf and bring it to the table.",
  "history": {"reset": true}          // so na 1a query do episodio, senao {}
  "state": {"states": <np float32 (1,32)>},
  "condition": {},
  "gt_action": [],
  "dataset_name": "simple",
  "timestamp": "..."
}
```

- Imagem: `observation["head_stereo_left"]` (psi0_decoupled_wbc.py:63-65),
  mesma camera ego do dataset (360x640).
- Instrucao: `task.instruction` do env (eval_decoupled_wbc.py:311) — para a
  task WMO e fixa: "pick up the blue tote from the shelf and bring it to the
  table." (g1_wholebody_locomotion_pick_totes_shelf_to_table_teleop.py:144-148)
  — **identica a caption do dataset v3** (meta/tasks parquet).
- Estado 32D (STATE_SLICES + `_last_cmd_torso_rpyh`, psi0_decoupled_wbc.py:14-21,
  67-74): hand(14, ordem psi0: thumb/middle/index esq + mao dir) + arm(14) +
  rpy+height(4). O server Cosmos **ignora** (modelo condiciona so em video+texto,
  como no treino `mode="wam"`).

### Resposta esperada (client.py `ResponseMessage`)

```json
{"action": <np (Ta,36)>, "err": 0.0, "traj_image": <np (1,1,3) uint8>}
```

`traj_image` precisa ser ndarray ndim==3 (client.py:166) — serve_psi0 devolve
zeros(1,1,3); replicamos.

### Como o cliente aplica a acao (psi0_decoupled_wbc.py:90-110)

Consome cada linha de 36D **na convencao do dataset psi0** (sem transformacao
de espaco, so reordenacao p/ nomes de juntas):
- `[:28]` -> `from_psi0_upper_joints`: thumb[0:3], middle[3:5], index[5:7],
  right_hand[7:14], arms[14:28] -> alvos de junta upper body;
- `[28:31]` -> waist roll/pitch/yaw; `[31:32]` -> base_height;
  `[32:36]` -> navigate_cmd (vx, vy, vyaw, target_yaw).
E o mesmo layout 36D do action do G1ToteMix (modality.json) usado no SFT
Cosmos3 => **nenhum remapeamento necessario no server**.

## 2. Mapeamento observacao SIMPLE -> entrada Cosmos3

| SIMPLE | Cosmos3 (igual eval_offline_g1.py) |
|---|---|
| `image["rgb_head_stereo_left"]` HxWx3 uint8 | float/255 -> resize bilinear 256x256 (G1 dataset `_resize`) -> uint8 `[3,1,256,256]` -> repetido T+1 frames -> reflection pad -> prompt JSON |
| `instruction` | `ai_caption` no `ActionPromptJsonFormatter` (viewpoint `ego_view`, `conditioning_fps=50`, `mode="wam"`, `idle_frames=0`) |
| `state["states"]` (1,32) | **ignorado** (nao usado no treino) |
| `history["reset"]` | logado; sem estado entre queries (sem RTC) |

## 3. Mapeamento chunk Cosmos3 -> resposta

`generate_samples_from_batch` -> `samples["action"][0]` `[Tp,64]` em [-1,1] ->
slice `[:, :36]` -> `denormalize_action(..., "quantile", g1_wholebody_stats.json)`
-> slice `[:Ta]` -> resposta `{"action", "err":0.0, "traj_image":zeros(1,1,3)}`
com a mesma serializacao numpy-b64 do cliente.

## 4. Chunk / horizonte / fps

- Treino: chunk_length=16 @ 50 fps (0.32 s). Psi0 baseline: exec_horizon 24
  @ 50 Hz com RTC.
- Requisito >=24 passos: default do server `--chunk-length=24`
  (divisivel por 4, sequence_plan com action_length=24) — **extrapolacao vs
  treino (16)**; validar open-loop (eval_offline com chunk 24) antes do lote.
  Fallback seguro: `COSMOS_CHUNK_LENGTH=16` (cliente consome os 16 e re-consulta).
- Sem RTC: Cosmos3 nao tem o fluxo prev_actions do Psi0; a fila do cliente
  bloqueia durante a inferencia (o sim espera; risco so de tempo de parede).

## 5. Riscos

1. **Chunk 24 extrapola o treino (16)** — validar offline; fallback 16.
2. **Latencia**: Cosmos3 ~16B gera acoes num pipeline de video; s/chunk medido
   no eval offline (metrics.json `mean_inference_s_per_window`). Se >> que o
   tempo real do chunk (0.32-0.48 s), o episodio demora mas nao quebra
   (cliente e sincrono). Medir antes do lote; ajustar `--num-steps` se preciso.
3. **Ordem de juntas**: confiamos que dataset G1ToteMix == espaco consumido
   pelo `from_psi0_upper_joints`; conferido estaticamente (secao 1), mas o
   smoke com video e o teste final.
4. **fps de conditioning**: server fixa 50 (nativo do dataset). O passo real
   do sim (control_dt) deve casar com 50 Hz como no Psi0.
5. **Dominio visual**: render Isaac vs videos de treino; unico frame repetido
   T+1 vezes (mesma aproximacao do eval offline).
6. **Sem estado proprioceptivo**: policy cega a proprio corpo; drift possivel
   em loco-manipulacao longa.
7. **Warmup**: primeira inferencia compila/aloca; server so abre HTTP depois
   do warmup, e o orquestrador espera /health ate 60 min.

## 6. So validavel com GPU (nao feito aqui)

- Carregamento do checkpoint iter_000000020000 (job 30171 ainda gerando).
- Shapes reais de `generate_samples_from_batch` com chunk 24.
- Latencia por chunk e por episodio.
- Handshake HTTP completo cliente<->server e video nao vazio do smoke.
- Sucesso/estatisticas do eval (2 episodios smoke -> depois lote com seed
  e comparacao com baseline Psi0 em eval/results/wmo-totes-20260807/).

## 7. Execucao (quando autorizado)

```bash
sbatch third_party/cosmos-framework/psi0_g1/sbatch/07_serve_and_eval_simple.sbatch
# overrides: COSMOS_CHECKPOINT_PATH, COSMOS_CHUNK_LENGTH=16, EPISODES, PORT
```

Resultados em `eval/results/cosmos3-g1-<data>/<run_id>/` (metadata.json,
logs/policy-server.log, logs/eval-smoke.log, smoke/eval_stats.txt, videos).

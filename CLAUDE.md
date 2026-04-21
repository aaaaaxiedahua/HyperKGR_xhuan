# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

HyperKGR is a PyTorch research codebase for knowledge graph reasoning in hyperbolic (Poincaré ball) space with a GNN that encodes symbolic path information (EMNLP 2025, Liu 2025). CUDA is assumed — code calls `.cuda()` unconditionally and sets the device via `--gpu`.

## Repository layout

Four near-duplicated code trees, one per (variant × setting) combination. Each tree is standalone and must be run from its own directory:

```
sample/      # node-sampling variant (AdaProp-style, primary)
  transductive/   # family, umls, WN18RR, fb15k-237, nell, YAGO
  inductive/      # WN18RR_v{1..4}, fb237_v{1..4}, nell_v{1..4}
not_sample/  # earlier variant without sampling; mostly superseded by sample/
  transductive/
  inductive/
paper/       # paper PDF only
```

When asked to change model behavior, confirm which tree the user means — changes do **not** propagate between `sample/` and `not_sample/`, nor between `transductive/` and `inductive/`. `sample/transductive/` is the most actively edited tree and is the only one with `hpo_optuna.py`.

Inside each tree the files play fixed roles:
- `train.py` — CLI entrypoint. Contains hardcoded per-dataset hyperparameters in an `if dataset == '<name>'` chain. Adding a new dataset = adding a branch there.
- `base_model.py` — `BaseModel` wraps `GNNModel`, owns Adam + `ExponentialLR`, the train/eval loops, checkpoint save/load, and the softmax-loss computation. Checkpoints go to `{data_path}/saveModel/`; per-run logs to `results/{dataset}/{modelName}_perf.txt`.
- `models.py` — hyperbolic primitives (`expmap0`, `logmap0`, `mobius_add`, `project`, `artanh`, `hyp_distance*`) plus `GNNLayer` and `GNNModel`. This is where hyperbolic math lives — the Poincaré-ball curvature `c` is a learnable `nn.Parameter` per layer, clamped by `safe_curvature(c) = c.clamp_min(1e-6)`.
- `load_data.py` — `DataLoader` reads `entities.txt`, `relations.txt`, `facts.txt`, `train.txt`, `valid.txt`, `test.txt`. It doubles every triple with an inverse relation (id `r + n_rel`) and reserves `2*n_rel` as a self-loop/identity relation added in `load_graph`. `get_neighbors` does neighbor expansion via a sparse `csr_matrix` multiplication — this is the hot path on every GNN layer forward.
- `utils.py` — ranking metrics (`cal_ranks`, `cal_performance`: MRR / Hits@1 / Hits@10) and `checkPath`.

## How a forward pass flows

Read `models.py:GNNModel.forward` alongside `load_data.py:DataLoader.get_neighbors` — the two are tightly coupled and hard to understand in isolation:

1. Start from query subjects `q_sub`; `nodes` is `[batch_idx, node_idx]`.
2. For each GNN layer, `loader.get_neighbors(...)` expands the current frontier by one hop, returning the new `nodes`, `edges` (with relative head/tail indices), and `old_nodes_new_idx` so the previous layer's hidden states can be re-indexed.
3. `GNNLayer.forward` computes an attention `alpha` over edges, projects the source hidden via a **Householder reflection sequence** (`_apply_relation_operator`, `M` reflections per relation — this is the "symbolic path" encoding), then does a **hyperbolic TransE-style message**: `message = logmap0(project(mobius_add(expmap0(hs_rel), expmap0(hr))))`. The message is aggregated via `torch_scatter.scatter` and mapped back through the Poincaré ball.
4. If `n_node_topk > 0`, the layer also samples top-k frontier nodes using `gumbel_softmax` (train) or `softmax` (eval), then calls `_frontier_mix` which does prototype-based attention among the sampled new nodes to refine their hidden states before a sigmoid-gated residual.
5. `GNNModel` combines the layer output with `h0` through a `nn.GRU` cell, using `old_nodes_new_idx` to scatter stale states into the expanded node set.
6. Final scoring: `W_final` on every visited node's hidden; unvisited entities get score 0.

The important invariant: everything the GNN touches is in the tangent space (output of `logmap0`); hyperbolic operations happen inside the message computation and are always sandwiched between `expmap0`/`logmap0`.

## Data convention (transductive)

The original `train.txt` from the upstream dataset is split into `facts.txt` and `train.txt` (default ratio ~3:1, controlled by `--fact_ratio`, higher = more facts = typically better). `shuffle_train()` re-splits every epoch from the combined pool. `--remove_1hop_edges` drops facts that are also direct train edges (used for fb15k-237 to prevent leakage of the exact query edge into the message graph).

Datasets are fetched externally (see README.md) from the RED-GNN repo; this repo does not ship the raw datasets.

## Common commands

All commands assume `cd` into the relevant subdirectory first.

### `sample/transductive/`

Train (per-dataset defaults are baked into `train.py`):
```
python3 train.py --data_path ./data/family/   --train --topk 100  --layers 8 --fact_ratio 0.90 --gpu 0
python3 train.py --data_path ./data/WN18RR/   --train --topk 1000 --layers 8 --fact_ratio 0.96 --gpu 0
python3 train.py --data_path ./data/fb15k-237/ --train --topk 2000 --layers 7 --fact_ratio 0.99 --remove_1hop_edges --gpu 0
python3 train.py --data_path ./data/nell/     --train --topk 2000 --layers 6 --fact_ratio 0.95 --gpu 0
python3 train.py --data_path ./data/YAGO/     --train --topk 1000 --layers 8 --fact_ratio 0.995 --gpu 0
python3 train.py --data_path ./data/umls/     --train --topk 100  --layers 5 --fact_ratio 0.90 --gpu 0
```

Evaluate from a checkpoint (no training):
```
python3 train.py --data_path ./data/WN18RR/ --eval --topk 1000 --layers 8 --gpu 0 --weight ./data/WN18RR/8-layers-best.pt
```

Hyperparameter search (Optuna TPE, 20 trials by default, early-stopping on stale val MRR):
```
python3 hpo_optuna.py --data_path ./data/WN18RR/ --gpu 0 --n_trials 20
```

Extra relevant flags: `--M` (Householder reflection count), `--d_p` (prototype dim; defaults to `hidden_dim`), `--p_mix` (frontier-mix dropout), `--lambda` / `lambda_mix` (frontier-mix bias weight), `--tau` (Gumbel-softmax temperature), `--eval_interval`, `--epoch`.

### `sample/inductive/`

Train a single split:
```
python3 train.py --data_path ./data/WN18RR_v1
```
Per-split hyperparameters are hardcoded in `train.py` (keyed by substring match on `data_path`). To sweep all 12 splits use `bash reproduce.sh`. Note: `train.py` pins `gpu = 0` and only trains for 30 epochs — if this needs to change, edit the script.

### `not_sample/`

Same CLI shape as above, simpler argument set:
```
cd not_sample/transductive && python -W ignore train.py --data_path=data/WN18RR
cd not_sample/inductive   && python -W ignore train.py --data_path=data/WN18RR_v1
```

## Dependencies

PyTorch + `torch_scatter` are required (the `scatter` call in `GNNLayer.forward` is load-bearing). README.md lists pytorch 1.9.1+cu102 / torch_scatter 2.0.9; `sample/README.md` lists torch 1.12.1 / torch_scatter 2.0.9 / numpy 1.21.6 / scipy 1.10.1. `hpo_optuna.py` additionally needs `optuna`. There is no `requirements.txt` or lockfile — install manually to match your CUDA.

## Gotchas

- No tests, no linter, no CI. The only notion of "correctness" is reproducing paper numbers on the listed datasets.
- `base_model.train_batch` contains an explicit NaN-guard that overwrites any NaN parameters with random values after each optimizer step. If numerics blow up this hides the symptom rather than fixing the cause — the usual culprits are `safe_curvature` thresholds or clamps inside `mobius_add` / `artanh`.
- `torch.cuda.set_device(opts.gpu)` is called even when `--gpu -1`; running without a GPU is not supported.
- Per-dataset hyperparameters live in three places that must stay in sync when adding/editing a dataset: the `if dataset == ...` chain in `sample/transductive/train.py`, the `DATASET_DEFAULTS` dict in `sample/transductive/hpo_optuna.py`, and any documented commands in `sample/README.md`.
- Checkpoint filenames embed the full config (`{n_layer}-layers-{topk}-...-dp{d_p}-M{M}-pmix{..}-lam{..}.pt`), so changing those knobs at eval time without matching `--weight` will silently load the wrong file name or fail.

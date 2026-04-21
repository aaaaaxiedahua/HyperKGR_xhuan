import argparse
import copy
import json
import os
import random

import numpy as np
import optuna
import torch

from base_model import BaseModel
from load_data import DataLoader
from utils import checkPath


DATASET_DEFAULTS = {
    'family': {
        'lr': 0.0036,
        'decay_rate': 0.999,
        'lamb': 0.000017,
        'hidden_dim': 48,
        'attn_dim': 5,
        'dropout': 0.29,
        'act': 'relu',
        'topk': 100,
        'layers': 8,
        'fact_ratio': 0.90,
        'n_batch': 20,
        'n_tbatch': 20,
    },
    'umls': {
        'lr': 0.0012,
        'decay_rate': 0.998,
        'lamb': 0.00014,
        'hidden_dim': 64,
        'attn_dim': 5,
        'dropout': 0.01,
        'act': 'tanh',
        'topk': 100,
        'layers': 5,
        'fact_ratio': 0.90,
        'n_batch': 10,
        'n_tbatch': 10,
    },
    'WN18RR': {
        'lr': 0.0030,
        'decay_rate': 0.994,
        'lamb': 0.00014,
        'hidden_dim': 64,
        'attn_dim': 5,
        'dropout': 0.02,
        'act': 'idd',
        'topk': 1000,
        'layers': 8,
        'fact_ratio': 0.96,
        'n_batch': 50,
        'n_tbatch': 50,
    },
    'fb15k-237': {
        'lr': 0.0009,
        'decay_rate': 0.9938,
        'lamb': 0.000080,
        'hidden_dim': 48,
        'attn_dim': 5,
        'dropout': 0.0391,
        'act': 'idd',
        'topk': 2000,
        'layers': 7,
        'fact_ratio': 0.99,
        'n_batch': 10,
        'n_tbatch': 10,
    },
    'nell': {
        'lr': 0.0011,
        'decay_rate': 0.9938,
        'lamb': 0.000089,
        'hidden_dim': 128,
        'attn_dim': 64,
        'dropout': 0.2593,
        'act': 'idd',
        'topk': 2000,
        'layers': 6,
        'fact_ratio': 0.95,
        'n_batch': 10,
        'n_tbatch': 10,
    },
    'YAGO': {
        'lr': 0.001,
        'decay_rate': 0.9429713470775948,
        'lamb': 0.000946516892415447,
        'hidden_dim': 64,
        'attn_dim': 2,
        'dropout': 0.19456805575101324,
        'act': 'relu',
        'topk': 1000,
        'layers': 8,
        'fact_ratio': 0.995,
        'n_batch': 5,
        'n_tbatch': 5,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description='Optuna HPO for HyperKGR transductive setting')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--topk', type=int, default=-1)
    parser.add_argument('--layers', type=int, default=-1)
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--scheduler', type=str, default='exp')
    parser.add_argument('--remove_1hop_edges', action='store_true')
    parser.add_argument('--fact_ratio', type=float, default=-1.0)
    parser.add_argument('--epoch', type=int, default=120)
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--n_trials', type=int, default=20)
    parser.add_argument('--early_stop_rounds', type=int, default=3)
    parser.add_argument('--study_early_stop_rounds', type=int, default=3)
    parser.add_argument('--study_name', type=str, default=None)
    parser.add_argument('--storage', type=str, default=None)
    parser.add_argument('--weight', type=str, default=None)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_dataset_name(data_path):
    parts = data_path.split('/')
    if len(parts[-1]) > 0:
        return parts[-1]
    return parts[-2]


def unique_preserve_order(items):
    seen = set()
    ordered = []
    for item in items:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def apply_dataset_defaults(opts, dataset):
    if dataset not in DATASET_DEFAULTS:
        raise ValueError(f'Unsupported dataset for HPO: {dataset}')
    base = dict(DATASET_DEFAULTS[dataset])
    if opts.topk > 0:
        base['topk'] = opts.topk
    if opts.layers > 0:
        base['layers'] = opts.layers
    if opts.fact_ratio > 0:
        base['fact_ratio'] = opts.fact_ratio
    opts.lr = base['lr']
    opts.decay_rate = base['decay_rate']
    opts.lamb = base['lamb']
    opts.hidden_dim = base['hidden_dim']
    opts.attn_dim = base['attn_dim']
    opts.dropout = base['dropout']
    opts.act = base['act']
    opts.topk = base['topk']
    opts.layers = base['layers']
    opts.fact_ratio = base['fact_ratio']
    opts.n_batch = base['n_batch']
    opts.n_tbatch = base['n_tbatch']
    return base


def build_layer_choices(base_layers):
    return unique_preserve_order(
        max(2, base_layers + offset)
        for offset in [-2, -1, 0, 1, 2]
    )


def build_topk_search_space(base_topk):
    step = max(10, int(round(base_topk * 0.1)))
    min_topk = max(step, int(round(base_topk * 0.5 / step)) * step)
    max_topk = min(1500, int(round(base_topk * 1.5 / step)) * step)
    if max_topk <= min_topk:
        max_topk = min_topk + step
    return min_topk, max_topk, step


def build_fact_ratio_search_space(base_ratio):
    min_ratio = max(0.75, round(base_ratio - 0.10, 3))
    max_ratio = min(0.995, round(base_ratio + 0.03, 3))
    if max_ratio <= min_ratio:
        max_ratio = min(0.995, round(min_ratio + 0.01, 3))
    return min_ratio, max_ratio


def build_search_space(dataset, base_cfg):
    min_topk, max_topk, topk_step = build_topk_search_space(base_cfg['topk'])
    min_ratio, max_ratio = build_fact_ratio_search_space(base_cfg['fact_ratio'])
    return {
        'layers': build_layer_choices(base_cfg['layers']),
        'topk': (min_topk, max_topk, topk_step),
        'fact_ratio': (min_ratio, max_ratio, 0.005),
        'lr': (1e-4, 5e-2, True),
        'decay_rate': (0.90, 0.9999, False),
        'lamb': (1e-7, 1e-2, True),
        'hidden_dim': [32, 48, 64, 96, 128, 192, 256],
        'attn_dim': unique_preserve_order([base_cfg['attn_dim'], 2, 4, 5, 8, 16, 32, 64]),
        'dropout': (0.0, 0.5),
        'act': unique_preserve_order([base_cfg['act'], 'idd', 'relu', 'tanh']),
        'shortcut_hops': [2, 3, 4],
        'shortcut_topk': [4, 8, 12, 16],
        'shortcut_decay': (0.3, 0.9, 0.1),
        'shortcut_lambda': (0.05, 0.4),
        'shortcut_candidate_cap': [16, 32, 64, 96, 128],
        'd_hop': [16, 32, 48, 64, 96, 128],
        'shortcut_prune_lambda': (0.0, 0.3),
    }


def suggest_hyperparams(trial, opts, dataset, base_cfg):
    search_space = build_search_space(dataset, base_cfg)

    opts.layers = trial.suggest_categorical('layers', search_space['layers'])
    min_topk, max_topk, topk_step = search_space['topk']
    opts.topk = trial.suggest_int('topk', min_topk, max_topk, step=topk_step)
    min_ratio, max_ratio, ratio_step = search_space['fact_ratio']
    opts.fact_ratio = trial.suggest_float('fact_ratio', min_ratio, max_ratio, step=ratio_step)
    lr_min, lr_max, lr_log = search_space['lr']
    opts.lr = trial.suggest_float('lr', lr_min, lr_max, log=lr_log)
    decay_min, decay_max, _ = search_space['decay_rate']
    opts.decay_rate = trial.suggest_float('decay_rate', decay_min, decay_max)
    lamb_min, lamb_max, lamb_log = search_space['lamb']
    opts.lamb = trial.suggest_float('lamb', lamb_min, lamb_max, log=lamb_log)
    opts.hidden_dim = trial.suggest_categorical('hidden_dim', search_space['hidden_dim'])
    opts.attn_dim = trial.suggest_categorical('attn_dim', search_space['attn_dim'])
    dropout_min, dropout_max = search_space['dropout']
    opts.dropout = trial.suggest_float('dropout', dropout_min, dropout_max)
    opts.act = trial.suggest_categorical('act', search_space['act'])
    opts.shortcut_hops = trial.suggest_categorical('shortcut_hops', search_space['shortcut_hops'])
    opts.shortcut_topk = trial.suggest_categorical('shortcut_topk', search_space['shortcut_topk'])
    sc_decay_min, sc_decay_max, sc_decay_step = search_space['shortcut_decay']
    opts.shortcut_decay = trial.suggest_float('shortcut_decay', sc_decay_min, sc_decay_max, step=sc_decay_step)
    sc_lam_min, sc_lam_max = search_space['shortcut_lambda']
    opts.shortcut_lambda = trial.suggest_float('shortcut_lambda', sc_lam_min, sc_lam_max)
    opts.shortcut_candidate_cap = trial.suggest_categorical('shortcut_candidate_cap', search_space['shortcut_candidate_cap'])
    opts.d_hop = trial.suggest_categorical('d_hop', search_space['d_hop'])
    sc_prune_min, sc_prune_max = search_space['shortcut_prune_lambda']
    opts.shortcut_prune_lambda = trial.suggest_float('shortcut_prune_lambda', sc_prune_min, sc_prune_max)
    opts.n_edge_topk = -1
    opts.n_layer = opts.layers
    opts.n_node_topk = [opts.topk] * opts.layers


def append_jsonl(path, payload):
    with open(path, 'a+', encoding='utf-8') as f:
        f.write(json.dumps(payload, ensure_ascii=False) + '\n')


def build_trial_opts(args, dataset, trial):
    opts = copy.deepcopy(args)
    base_cfg = apply_dataset_defaults(opts, dataset)
    suggest_hyperparams(trial, opts, dataset, base_cfg)
    return opts


def summarize_trial_opts(opts):
    return {
        'layers': opts.layers,
        'topk': opts.topk,
        'fact_ratio': opts.fact_ratio,
        'lr': opts.lr,
        'decay_rate': opts.decay_rate,
        'lamb': opts.lamb,
        'hidden_dim': opts.hidden_dim,
        'attn_dim': opts.attn_dim,
        'dropout': opts.dropout,
        'act': opts.act,
        'shortcut_hops': opts.shortcut_hops,
        'shortcut_topk': opts.shortcut_topk,
        'shortcut_decay': opts.shortcut_decay,
        'shortcut_lambda': opts.shortcut_lambda,
        'shortcut_candidate_cap': opts.shortcut_candidate_cap,
        'd_hop': opts.d_hop,
        'shortcut_prune_lambda': opts.shortcut_prune_lambda,
        'epoch': opts.epoch,
    }


def objective_factory(args, dataset, trial_log_path):
    def objective(trial):
        set_seed(args.seed)
        opts = build_trial_opts(args, dataset, trial)
        print(f'==> trial {trial.number} params: {json.dumps(summarize_trial_opts(opts), ensure_ascii=False)}')
        loader = DataLoader(opts)
        opts.n_ent = loader.n_ent
        opts.n_rel = loader.n_rel

        model = BaseModel(opts, loader)
        if opts.weight is not None:
            model.loadModel(opts.weight)
            model._update()
            model.model.updateTopkNums(opts.n_node_topk)

        best_v_mrr = float('-inf')
        best_epoch = 0
        stale_rounds = 0
        trial_status = 'ok'
        try:
            for epoch in range(opts.epoch):
                model.train_batch()
                if (epoch + 1) % opts.eval_interval != 0:
                    continue

                result_dict, _ = model.evaluate(eval_val=True, eval_test=False, verbose=False)
                current_v_mrr = float(result_dict['v_mrr'])

                if current_v_mrr > best_v_mrr:
                    best_v_mrr = current_v_mrr
                    best_epoch = epoch + 1
                    stale_rounds = 0
                else:
                    stale_rounds += 1

                trial.report(best_v_mrr, epoch + 1)
                if stale_rounds >= opts.early_stop_rounds:
                    break
        except RuntimeError as exc:
            if 'out of memory' not in str(exc).lower():
                raise
            trial_status = 'oom'
            best_v_mrr = 0.0
            best_epoch = 0
            stale_rounds = opts.early_stop_rounds
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if best_v_mrr == float('-inf'):
            best_v_mrr = 0.0

        trial.set_user_attr('best_epoch', best_epoch)
        trial.set_user_attr('stale_rounds', stale_rounds)
        trial.set_user_attr('status', trial_status)

        append_jsonl(
            trial_log_path,
            {
                'trial': trial.number,
                'value': best_v_mrr,
                'best_epoch': best_epoch,
                'status': trial_status,
                'params': trial.params,
            },
        )

        del model
        del loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return best_v_mrr

    return objective


def save_best_result(study, path):
    best_summary = {
        'best_value': study.best_value,
        'best_trial': study.best_trial.number,
        'params': study.best_params,
        'user_attrs': study.best_trial.user_attrs,
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(best_summary, f, indent=2, ensure_ascii=False)


class StudyEarlyStopCallback:
    def __init__(self, patience):
        self.patience = patience
        self.best_value = None
        self.bad_rounds = 0

    def __call__(self, study, trial):
        current_best = study.best_value
        if self.best_value is None or current_best > self.best_value:
            self.best_value = current_best
            self.bad_rounds = 0
            return

        self.bad_rounds += 1
        if self.bad_rounds >= self.patience:
            study.stop()


def main():
    args = parse_args()
    dataset = infer_dataset_name(args.data_path)
    study_name = args.study_name or f'{dataset}_optuna_hpo'

    checkPath('./results/')
    checkPath(f'./results/{dataset}/')

    trial_log_path = f'./results/{dataset}/{study_name}_trials.jsonl'
    best_result_path = f'./results/{dataset}/{study_name}_best.json'

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    print('==> gpu:', args.gpu)
    print('==> dataset:', dataset)
    print('==> study_name:', study_name)

    sampler = optuna.samplers.TPESampler(
        seed=args.seed,
        n_startup_trials=3,
    )
    pruner = optuna.pruners.NopPruner()
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        storage=args.storage,
        load_if_exists=True,
    )

    objective = objective_factory(args, dataset, trial_log_path)
    callbacks = [StudyEarlyStopCallback(args.study_early_stop_rounds)]
    study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True, callbacks=callbacks)

    save_best_result(study, best_result_path)

    print('==> best trial:', study.best_trial.number)
    print('==> best v_mrr:', study.best_value)
    print('==> best params:')
    for key, value in study.best_params.items():
        print(f'   {key}: {value}')
    print('==> saved best result to:', best_result_path)
    print('==> saved trial log to:', trial_log_path)


if __name__ == '__main__':
    main()

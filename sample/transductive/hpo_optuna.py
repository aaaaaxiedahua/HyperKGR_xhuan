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
    parser.add_argument('--d_path', type=int, default=32)
    parser.add_argument('--d_score', type=int, default=-1)
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--scheduler', type=str, default='exp')
    parser.add_argument('--remove_1hop_edges', action='store_true')
    parser.add_argument('--fact_ratio', type=float, default=-1.0)
    parser.add_argument('--epoch', type=int, default=120)
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--n_trials', type=int, default=20)
    parser.add_argument('--startup_random_trials', type=int, default=3)
    parser.add_argument('--early_stop_rounds', type=int, default=3)
    parser.add_argument('--study_name', type=str, default=None)
    parser.add_argument('--storage', type=str, default=None)
    parser.add_argument('--weight', type=str, default=None)
    parser.add_argument('--trial_json', '--trail_json', dest='trial_json', type=str, default=None)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
    except RuntimeError as exc:
        if 'out of memory' not in str(exc).lower():
            raise
        safe_cuda_empty_cache()
    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed_all(seed)
        except RuntimeError as exc:
            if 'out of memory' not in str(exc).lower():
                raise
            safe_cuda_empty_cache()


def safe_cuda_empty_cache():
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
    except RuntimeError:
        pass


def infer_dataset_name(data_path):
    parts = data_path.split('/')
    if len(parts[-1]) > 0:
        return parts[-1]
    return parts[-2]


def study_name_safe(study_name, dataset):
    value = study_name or f'{dataset}_optuna_hpo'
    return value.replace('/', '_').replace('\\', '_').replace(':', '_').replace(' ', '_')


def load_trial_json(trial_json_arg):
    if trial_json_arg is None:
        return None
    if os.path.isfile(trial_json_arg):
        with open(trial_json_arg, 'r', encoding='utf-8') as f:
            payload = json.load(f)
    else:
        payload = json.loads(trial_json_arg)
    if not isinstance(payload, dict):
        raise ValueError('--trial_json/--trail_json must be a JSON object or a path to a JSON object file')
    return payload


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
    opts.d_path = 32 if getattr(opts, 'd_path', -1) <= 0 else opts.d_path
    opts.d_score = opts.hidden_dim if getattr(opts, 'd_score', -1) <= 0 else opts.d_score
    opts.n_node_topk = [opts.topk] * opts.layers
    opts.n_edge_topk = -1
    opts.n_layer = opts.layers
    return base


def build_search_space(dataset, base_cfg):
    return {
        'layers': [4, 6, 8],
        'topk': [950, 1000, 1050],
        'fact_ratio': (0.95, 0.96, 0.01),
        'lr': (1e-4, 1e-2, True),
        'decay_rate': (0.90, 0.9999, False),
        'lamb': (1e-7, 1e-2, True),
        'hidden_dim': [32, 48, 64, 96],
        'd_path': [16, 32, 48, 64],
        'd_score': [32, 48, 64, 96],
        'attn_dim': unique_preserve_order([base_cfg['attn_dim'], 2, 4, 5, 6, 8, 16]),
        'dropout': (0.0, 0.5),
        'act': unique_preserve_order([base_cfg['act'], 'idd', 'relu', 'tanh']),
    }


def suggest_hyperparams(trial, opts, dataset, base_cfg):
    search_space = build_search_space(dataset, base_cfg)

    opts.layers = trial.suggest_categorical('layers', search_space['layers'])
    opts.topk = trial.suggest_categorical('topk', search_space['topk'])
    min_ratio, max_ratio, ratio_step = search_space['fact_ratio']
    opts.fact_ratio = trial.suggest_float('fact_ratio', min_ratio, max_ratio, step=ratio_step)
    lr_min, lr_max, lr_log = search_space['lr']
    opts.lr = trial.suggest_float('lr', lr_min, lr_max, log=lr_log)
    decay_min, decay_max, _ = search_space['decay_rate']
    opts.decay_rate = trial.suggest_float('decay_rate', decay_min, decay_max)
    lamb_min, lamb_max, lamb_log = search_space['lamb']
    opts.lamb = trial.suggest_float('lamb', lamb_min, lamb_max, log=lamb_log)
    opts.hidden_dim = trial.suggest_categorical('hidden_dim', search_space['hidden_dim'])
    opts.d_path = trial.suggest_categorical('d_path', search_space['d_path'])
    opts.d_score = trial.suggest_categorical('d_score', search_space['d_score'])
    opts.attn_dim = trial.suggest_categorical('attn_dim', search_space['attn_dim'])
    dropout_min, dropout_max = search_space['dropout']
    opts.dropout = trial.suggest_float('dropout', dropout_min, dropout_max)
    opts.act = trial.suggest_categorical('act', search_space['act'])
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
        'd_path': opts.d_path,
        'd_score': opts.d_score,
        'attn_dim': opts.attn_dim,
        'dropout': opts.dropout,
        'act': opts.act,
        'epoch': opts.epoch,
    }


def objective_factory(args, dataset, trial_log_path):
    def objective(trial):
        opts = None
        loader = None
        model = None
        current_stage = 'objective_start'
        best_v_mrr = float('-inf')
        best_t_mrr = float('-inf')
        best_epoch = 0
        stale_rounds = 0
        trial_status = 'ok'
        best_ckpt_path = None
        try:
            current_stage = 'set_seed'
            print(f'==> trial {trial.number} stage: {current_stage}')
            set_seed(args.seed)
            current_stage = 'build_trial_opts'
            print(f'==> trial {trial.number} stage: {current_stage}')
            opts = build_trial_opts(args, dataset, trial)
            print(f'==> trial {trial.number} params: {json.dumps(summarize_trial_opts(opts), ensure_ascii=False)}')
            current_stage = 'build_dataloader'
            print(f'==> trial {trial.number} stage: {current_stage}')
            loader = DataLoader(opts)
            checkPath(f'{loader.task_dir}/saveModel/')
            opts.n_ent = loader.n_ent
            opts.n_rel = loader.n_rel

            current_stage = 'build_model'
            print(f'==> trial {trial.number} stage: {current_stage}')
            model = BaseModel(opts, loader)
            model.modelName = f'{study_name_safe(opts.study_name if hasattr(opts, "study_name") else None, dataset)}-trial{trial.number}'
            opts.perf_file = f'results/{dataset}/{model.modelName}_perf.txt'
            config_str = '%.4f, %.4f, %.6f,  %d, %d, %d, %d, %.4f,%s, %d, %d\n' % (
                opts.lr,
                opts.decay_rate,
                opts.lamb,
                opts.hidden_dim,
                opts.attn_dim,
                opts.n_layer,
                opts.n_batch,
                opts.dropout,
                opts.act,
                opts.d_path,
                opts.d_score,
            )
            print(config_str)
            with open(opts.perf_file, 'a+', encoding='utf-8') as f:
                f.write(config_str)
            if opts.weight is not None:
                current_stage = 'load_weight'
                print(f'==> trial {trial.number} stage: {current_stage}')
                model.loadModel(opts.weight)
                model._update()
                model.model.updateTopkNums(opts.n_node_topk)

            for epoch in range(opts.epoch):
                current_stage = f'epoch_{epoch + 1}_train'
                model.train_batch()
                if (epoch + 1) % opts.eval_interval != 0:
                    continue

                current_stage = f'epoch_{epoch + 1}_eval'
                result_dict, out_str = model.evaluate(eval_val=True, eval_test=True, verbose=True)
                current_v_mrr = float(result_dict['v_mrr'])
                current_t_mrr = float(result_dict['t_mrr'])
                print(f'==> trial {trial.number} epoch {epoch + 1}: {out_str.strip()}')
                with open(opts.perf_file, 'a+', encoding='utf-8') as f:
                    f.write(f'==> trial {trial.number} epoch {epoch + 1}: {out_str}')

                if current_v_mrr > best_v_mrr:
                    best_v_mrr = current_v_mrr
                    best_t_mrr = current_t_mrr
                    best_epoch = epoch + 1
                    stale_rounds = 0
                    metric_str = f'epoch{epoch + 1}_ValMRR_{str(best_v_mrr)[:5]}_TestMRR_{str(best_t_mrr)[:5]}'
                    model.saveModelToFiles(metric_str, deleteLastFile=True)
                    best_ckpt_path = model.lastSaveGNNPath
                else:
                    stale_rounds += 1

                trial.report(best_v_mrr, epoch + 1)
                if stale_rounds >= opts.early_stop_rounds:
                    print(f'==> trial {trial.number} early stop at epoch {epoch + 1} after {stale_rounds} stale eval rounds.')
                    break
        except RuntimeError as exc:
            if 'out of memory' not in str(exc).lower():
                raise
            trial_status = 'oom'
            best_v_mrr = 0.0
            best_t_mrr = 0.0
            best_epoch = 0
            stale_rounds = getattr(opts, 'early_stop_rounds', args.early_stop_rounds)
            print(f'==> trial {trial.number} OOM at stage: {current_stage}')
            if best_ckpt_path is not None and os.path.exists(best_ckpt_path):
                os.remove(best_ckpt_path)
            best_ckpt_path = None
            safe_cuda_empty_cache()

        if best_v_mrr == float('-inf'):
            best_v_mrr = 0.0
        if best_t_mrr == float('-inf'):
            best_t_mrr = -1.0

        trial.set_user_attr('best_epoch', best_epoch)
        trial.set_user_attr('stale_rounds', stale_rounds)
        trial.set_user_attr('status', trial_status)
        trial.set_user_attr('best_t_mrr', best_t_mrr)
        trial.set_user_attr('checkpoint_path', best_ckpt_path)
        trial.set_user_attr('last_stage', current_stage)

        append_jsonl(
            trial_log_path,
            {
                'trial': trial.number,
                'value': best_v_mrr,
                'best_epoch': best_epoch,
                'best_t_mrr': best_t_mrr,
                'status': trial_status,
                'last_stage': current_stage,
                'checkpoint_path': best_ckpt_path,
                'params': trial.params,
            },
        )

        if model is not None:
            del model
        if loader is not None:
            del loader
        safe_cuda_empty_cache()

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


def main():
    args = parse_args()
    dataset = infer_dataset_name(args.data_path)
    study_name = args.study_name or f'{dataset}_optuna_hpo'
    safe_study_name = study_name_safe(study_name, dataset)
    fixed_trial_params = load_trial_json(args.trial_json)

    checkPath('./results/')
    checkPath(f'./results/{dataset}/')

    trial_log_path = f'./results/{dataset}/{study_name}_trials.jsonl'
    best_result_path = f'./results/{dataset}/{study_name}_best.json'
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    print('==> gpu:', args.gpu)
    print('==> dataset:', dataset)
    print('==> study_name:', study_name)
    startup_trial_count = args.startup_random_trials + (1 if fixed_trial_params is not None else 0)
    print('==> startup_random_trials:', args.startup_random_trials)
    if fixed_trial_params is not None:
        print('==> startup schedule: 1 fixed trial + %d random trial(s), then TPE' % args.startup_random_trials)
    else:
        print('==> startup schedule: %d random trial(s), then TPE' % args.startup_random_trials)

    sampler = optuna.samplers.TPESampler(
        seed=args.seed,
        n_startup_trials=startup_trial_count,
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
    if fixed_trial_params is not None:
        print(f'==> enqueue fixed trial params: {json.dumps(fixed_trial_params, ensure_ascii=False)}')
        study.enqueue_trial(fixed_trial_params)

    objective = objective_factory(args, dataset, trial_log_path)
    study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)

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

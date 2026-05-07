"""
Runner CLI pour BaseGrad.

Usage:
    python baselines/BaseGrad/run_basegrad.py --dataset etth1 --model gru --seeds 1 9 30
"""

import argparse
import os
import sys
import time
import numpy as np
import torch

# Ajouter le projet au path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from baselines.common.bounds import compute_bounds_np, load_bounds_params, get_rl_config_path
from baselines.common.result_io import check_cache, save_results, aggregate_seeds, build_result_dict
from baselines.common.evaluator_wrapper import run_evaluation
from baselines.common.visualization import plot_cf_example
from baselines.BaseGrad.basegrad import BaseGrad
from src.models.Forecaster.forecaster_wrapper_v2 import ForecasterWrapperV2
from src.training.RL_trainers.trainer_last import prepare_rl_data
from src.utils.config import load_config


def run_basegrad_single_seed(dataset, model, seed, device, n_batches=20):
    """
    Exécute BaseGrad pour un (dataset, model, seed) donné.
    
    Returns
    -------
    dict
        Métriques pour ce seed
    """
    # Fixer les seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Charger configs
    cfg_f_path = f"assets/configs/models/{dataset}_dataset/forecasters/{model}/{dataset}_96_48_S.json"
    cfg_ae_path = f"assets/configs/models/{dataset}_dataset/ae/tcn_ae.json"
    rl_config_path = get_rl_config_path(dataset, model)
    
    cfg_f = load_config(cfg_f_path)
    cfg_ae = load_config(cfg_ae_path)
    bounds_params = load_bounds_params(rl_config_path)
    
    print(f"\n[BaseGrad] Dataset={dataset}, Model={model}, Seed={seed}")
    print(f"  Bounds: rho={bounds_params['rho']}, fr={bounds_params['fr']}, direction={bounds_params['direction']}")
    
    # Charger forecaster
    forecaster = ForecasterWrapperV2(cfg_f, device)
    forecaster.model.eval()
    for p in forecaster.model.parameters():
        p.requires_grad_(False)
    
    # Charger données
    train_loader, test_loader, _ = prepare_rl_data(cfg_f, cfg_ae, device)
    
    # Collecter train set (pour plausibilité)
    print("  Collecting train set (for plausibility)...")
    X_train_list = []
    for i, batch in enumerate(train_loader):
        if i >= 20:  # Limiter à 20 batches pour la plausibilité
            break
        batch_x, _, _, _ = batch
        batch_x = batch_x.float().to(device)
        X_train_list.append(batch_x[:, :, -1:].cpu().numpy())
    
    X_train = np.concatenate(X_train_list, axis=0)
    print(f"  Train set: {len(X_train)} samples")
    
    # Collecter test set
    print(f"  Collecting test set ({n_batches} batches)...")
    X_test_list, Y_hat_test_list, X_full_list, X_mark_list = [], [], [], []
    for i, batch in enumerate(test_loader):
        if i >= n_batches:
            break
        batch_x, _, batch_x_mark, _ = batch
        batch_x = batch_x.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        
        with torch.no_grad():
            y_hat = forecaster.predict_ot(batch_x, batch_x_mark)
        
        X_test_list.append(batch_x[:, :, -1:].cpu().numpy())
        Y_hat_test_list.append(y_hat.cpu().numpy())
        X_full_list.append(batch_x)
        X_mark_list.append(batch_x_mark)
    
    X_test = np.concatenate(X_test_list, axis=0)
    Y_hat_test = np.concatenate(Y_hat_test_list, axis=0)
    X_full = torch.cat(X_full_list, dim=0)
    X_mark = torch.cat(X_mark_list, dim=0)
    print(f"  Test set: {len(X_test)} samples")
    
    # Calculer bornes RL-MCF
    alphas, betas = compute_bounds_np(
        x_ot=X_test,
        y_hat=Y_hat_test,
        rho=bounds_params['rho'],
        fr=bounds_params['fr'],
        direction=bounds_params['direction'],
        global_sigma=bounds_params['global_sigma']
    )
    
    # BaseGrad
    print("  Running BaseGrad...")
    t0 = time.time()
    
    basegrad = BaseGrad(
        forecaster=forecaster,
        device=device,
        lr=0.01,
        max_iter=300,
        w_validity=1.0,
        w_proximity=0.5
    )
    X_cf, Y_cf = basegrad.transform(X_test, alphas, betas, X_full, X_mark)
    
    runtime = time.time() - t0
    print(f"  Runtime: {runtime:.2f}s ({runtime/len(X_test)*1000:.1f}ms/sample)")
    
    # Évaluation
    print("  Evaluating...")
    eval_results = run_evaluation(
        x_orig=X_test,
        x_cf=X_cf,
        y_hat=Y_hat_test,
        y_cf=Y_cf,
        alphas=alphas,
        betas=betas,
        x_train=X_train,
        method_name="BaseGrad",
        seed=seed
    )
    
    # Générer figure de visualisation (premier sample)
    print("  Generating visualization...")
    fig_path = plot_cf_example(
        x_orig=X_test[0, :, 0],
        x_cf=X_cf[0, :, 0],
        y_orig=Y_hat_test[0, :, 0],
        y_cf=Y_cf[0, :, 0],
        alpha=alphas[0],
        beta=betas[0],
        method_name="BaseGrad",
        dataset=dataset,
        model=model,
        output_dir="baselines/BaseGrad/figures",
        sample_idx=0
    )
    print(f"  [Figure saved] {fig_path}")
    
    # Extraire les métriques
    metrics = {}
    for k, v in eval_results['extended_metrics'].items():
        if isinstance(v, dict) and 'mean' in v:
            metrics[k] = v['mean']
    
    result_dict = {
        'metrics': metrics,
        'runtime': runtime,
        'n_samples': len(X_test)
    }
    
    return result_dict


def main():
    parser = argparse.ArgumentParser(description="Run BaseGrad baseline")
    parser.add_argument("--dataset", type=str, required=True, choices=["etth1", "etth2", "weather"])
    parser.add_argument("--model", type=str, required=True, choices=["gru", "itransformer", "patchtst", "dlinear", "timesnet"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 9, 30])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_batches", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="baselines/BaseGrad/results")
    
    args = parser.parse_args()
    
    # Vérifier cache
    output_path = os.path.join(args.output_dir, f"basegrad_{args.dataset}_{args.model}.json")
    checkpoint_path = f"assets/checkpoints/{args.dataset}_chpts/forecaster/chpt_{args.dataset}_96_48_{args.model}_S.pth"
    
    if check_cache(output_path, checkpoint_path):
        print(f"[Cache hit] {output_path}")
        return
    
    device = torch.device(args.device)
    
    # Exécuter pour chaque seed
    seed_results = []
    all_results = []
    total_runtime = 0
    
    for seed in args.seeds:
        result = run_basegrad_single_seed(args.dataset, args.model, seed, device, args.n_batches)
        seed_results.append(result['metrics'])
        all_results.append(result)
        total_runtime += result['runtime']
    
    # Agréger les seeds
    avg_metrics = aggregate_seeds(seed_results)
    n_samples = all_results[0]['n_samples'] if all_results else 0
    
    # Charger bounds params pour le JSON
    rl_config_path = get_rl_config_path(args.dataset, args.model)
    bounds_params = load_bounds_params(rl_config_path)
    
    # Construire le résultat final
    final_result = build_result_dict(
        method="BaseGrad",
        dataset=args.dataset,
        model=args.model,
        n_samples=n_samples,
        runtime_seconds=total_runtime / len(args.seeds),
        avg_metrics=avg_metrics,
        forecastcf_metrics=None,
        seeds=args.seeds,
        bounds_params=bounds_params,
        checkpoint_path=checkpoint_path
    )
    
    # Sauvegarder
    save_results(final_result, output_path)
    
    print(f"\n{'='*60}")
    print(f"BaseGrad — {args.dataset}/{args.model}")
    print(f"{'='*60}")
    print(f"Validity Ratio:       {avg_metrics.get('validity_ratio', {}).get('mean', 0):.4f} ± {avg_metrics.get('validity_ratio', {}).get('std', 0):.4f}")
    print(f"Stepwise AUC:         {avg_metrics.get('stepwise_auc', {}).get('mean', 0):.4f} ± {avg_metrics.get('stepwise_auc', {}).get('std', 0):.4f}")
    print(f"Proximity L2:         {avg_metrics.get('proximity_l2', {}).get('mean', 0):.4f} ± {avg_metrics.get('proximity_l2', {}).get('std', 0):.4f}")
    print(f"Compactness:          {avg_metrics.get('compactness', {}).get('mean', 0):.4f} ± {avg_metrics.get('compactness', {}).get('std', 0):.4f}")
    print(f"Temporal Consistency: {avg_metrics.get('temporal_consistency', {}).get('mean', 0):.4f} ± {avg_metrics.get('temporal_consistency', {}).get('std', 0):.4f}")
    print(f"Plausibility:         {avg_metrics.get('plausibility_ensemble', {}).get('mean', 0):.4f} ± {avg_metrics.get('plausibility_ensemble', {}).get('std', 0):.4f}")
    print(f"Runtime:              {total_runtime/len(args.seeds):.2f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

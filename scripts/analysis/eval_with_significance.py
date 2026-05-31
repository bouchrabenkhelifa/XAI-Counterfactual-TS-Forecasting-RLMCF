"""
Evaluate RL-MCF across all datasets and architectures with statistical significance.
Runs evaluation multiple times with different test batches (bootstrap) to get
confidence intervals and p-values.

Usage:
    python scripts/analysis/eval_with_significance.py
"""

import os
import sys
import json
import numpy as np
import torch
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.config import load_config
from src.utils.train_tools import get_device
from src.models.autoencoder.tcn_ae import TCNAutoEncoder
from src.models.Forecaster.forecaster_wrapper_v2 import ForecasterWrapperV2
from src.models.RL.agent import ActorCritic
from src.models.RL.reward_last import CFReward
from src.data_provider.data_factory import data_provider
from src.training.RL_trainers.trainer_main import build_temporal_mask
from src.evaluation.unified_evaluator import CounterfactualEvaluator

RESULTS_DIR = "assets/results/significance"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# All configs
# ─────────────────────────────────────────────────────────────────────────────

CONFIGS = {
    ("etth1", "itransformer"): {
        "cfg_f": "assets/configs/etth1_dataset/forecasters/itransformer/etth1_96_48_S.json",
        "cfg_ae": "assets/configs/etth1_dataset/ae/tcn_ae.json",
        "cfg_rl": "assets/configs/etth1_dataset/RL_ablations/config_itransformer_best.json",
        "rl_ckpt": "assets/checkpoints/etth1_chpts/RL/itransformer/rl_cf_itransformer_best_etth1_agent_best.pt",
    },
    ("etth1", "patchtst"): {
        "cfg_f": "assets/configs/etth1_dataset/forecasters/patchtst/etth1_96_48_S.json",
        "cfg_ae": "assets/configs/etth1_dataset/ae/tcn_ae.json",
        "cfg_rl": "assets/configs/etth1_dataset/RL_ablations/config_patchtst.json",
        "rl_ckpt": "assets/checkpoints/etth1_chpts/RL/patchtst/rl_cf_patchtst_etth1_agent_best.pt",
    },
    ("etth1", "dlinear"): {
        "cfg_f": "assets/configs/etth1_dataset/forecasters/dlinear/etth1_96_48_S.json",
        "cfg_ae": "assets/configs/etth1_dataset/ae/tcn_ae.json",
        "cfg_rl": "assets/configs/etth1_dataset/RL_ablations/config_dlinear.json",
        "rl_ckpt": "assets/checkpoints/etth1_chpts/RL/dlinear/rl_cf_dlinear_etth1_agent_best.pt",
    },
    ("etth1", "gru"): {
        "cfg_f": "assets/configs/etth1_dataset/forecasters/gru/etth1_96_48_S.json",
        "cfg_ae": "assets/configs/etth1_dataset/ae/tcn_ae.json",
        "cfg_rl": "assets/configs/etth1_dataset/RL_ablations/config_gru.json",
        "rl_ckpt": "assets/checkpoints/etth1_chpts/RL/gru/rl_cf_gru_etth1_agent_best.pt",
    },
    ("etth1", "timesnet"): {
        "cfg_f": "assets/configs/etth1_dataset/forecasters/timesnet/etth1_96_48_S.json",
        "cfg_ae": "assets/configs/etth1_dataset/ae/tcn_ae.json",
        "cfg_rl": "assets/configs/etth1_dataset/RL_ablations/config_timesnet.json",
        "rl_ckpt": "assets/checkpoints/etth1_chpts/RL/timesnet/rl_cf_timesnet_etth1_agent_best.pt",
    },
    ("etth2", "itransformer"): {
        "cfg_f": "assets/configs/etth2_dataset/forecasters/itransformer/etth2_96_48_S.json",
        "cfg_ae": "assets/configs/etth2_dataset/ae/tcn_ae.json",
        "cfg_rl": "assets/configs/etth2_dataset/RL/config_itransformer.json",
        "rl_ckpt": "assets/checkpoints/etth2_chpts/RL/itransformer/rl_cf_itransformer_etth2_agent_best.pt",
    },
    ("etth2", "patchtst"): {
        "cfg_f": "assets/configs/etth2_dataset/forecasters/patchtst/etth2_96_48_S.json",
        "cfg_ae": "assets/configs/etth2_dataset/ae/tcn_ae.json",
        "cfg_rl": "assets/configs/etth2_dataset/RL/config_patchtst.json",
        "rl_ckpt": "assets/checkpoints/etth2_chpts/RL/patchtst/rl_cf_patchtst_etth2_agent_best.pt",
    },
    ("etth2", "dlinear"): {
        "cfg_f": "assets/configs/etth2_dataset/forecasters/dlinear/etth2_96_48_S.json",
        "cfg_ae": "assets/configs/etth2_dataset/ae/tcn_ae.json",
        "cfg_rl": "assets/configs/etth2_dataset/RL/config_dlinear.json",
        "rl_ckpt": "assets/checkpoints/etth2_chpts/RL/dlinear/rl_cf_dlinear_etth2_agent_best.pt",
    },
    ("etth2", "gru"): {
        "cfg_f": "assets/configs/etth2_dataset/forecasters/gru/etth2_96_48_S.json",
        "cfg_ae": "assets/configs/etth2_dataset/ae/tcn_ae.json",
        "cfg_rl": "assets/configs/etth2_dataset/RL/config_gru.json",
        "rl_ckpt": "assets/checkpoints/etth2_chpts/RL/gru/rl_cf_gru_etth2_agent_best.pt",
    },
    ("etth2", "timesnet"): {
        "cfg_f": "assets/configs/etth2_dataset/forecasters/timesnet/etth2_96_48_S.json",
        "cfg_ae": "assets/configs/etth2_dataset/ae/tcn_ae.json",
        "cfg_rl": "assets/configs/etth2_dataset/RL/config_timesnet.json",
        "rl_ckpt": "assets/checkpoints/etth2_chpts/RL/timesnet/rl_cf_timesnet_etth2_agent_best.pt",
    },
    ("weather", "itransformer"): {
        "cfg_f": "assets/configs/weather_dataset/forecasters/itransformer/weather_96_96_S.json",
        "cfg_ae": "assets/configs/weather_dataset/ae/tcn_ae.json",
        "cfg_rl": "assets/configs/weather_dataset/RL/config_itransformer.json",
        "rl_ckpt": "assets/checkpoints/weather_chpts/RL/RL_itransformer/rl_cf_itransformer_weather_agent_best.pt",
    },
    ("weather", "patchtst"): {
        "cfg_f": "assets/configs/weather_dataset/forecasters/patchtst/weather_96_96_S.json",
        "cfg_ae": "assets/configs/weather_dataset/ae/tcn_ae.json",
        "cfg_rl": "assets/configs/weather_dataset/RL/config_patchtst.json",
        "rl_ckpt": "assets/checkpoints/weather_chpts/RL/RL_patchtst/rl_cf_patchtst_weather_agent_best.pt",
    },
    ("weather", "dlinear"): {
        "cfg_f": "assets/configs/weather_dataset/forecasters/dlinear/weather_96_96_S.json",
        "cfg_ae": "assets/configs/weather_dataset/ae/tcn_ae.json",
        "cfg_rl": "assets/configs/weather_dataset/RL/config_dlinear.json",
        "rl_ckpt": "assets/checkpoints/weather_chpts/RL/RL_dlinear/rl_cf_dlinear_weather_agent_best.pt",
    },
    ("weather", "gru"): {
        "cfg_f": "assets/configs/weather_dataset/forecasters/gru/weather_96_96_S.json",
        "cfg_ae": "assets/configs/weather_dataset/ae/tcn_ae.json",
        "cfg_rl": "assets/configs/weather_dataset/RL/config_gru.json",
        "rl_ckpt": "assets/checkpoints/weather_chpts/RL/RL_gru/rl_cf_gru_weather_agent_best.pt",
    },
    ("weather", "timesnet"): {
        "cfg_f": "assets/configs/weather_dataset/forecasters/timesnet/weather_96_96_S.json",
        "cfg_ae": "assets/configs/weather_dataset/ae/tcn_ae.json",
        "cfg_rl": "assets/configs/weather_dataset/RL/config_timesnet.json",
        "rl_ckpt": "assets/checkpoints/weather_chpts/RL/RL_timesnet/rl_cf_timesnet_weather_agent_best.pt",
    },
}


def evaluate_single_run(cfg_f, cfg_ae, cfg_rl, rl_ckpt, device, n_batches=20, seed=None):
    """Run one evaluation pass and return per-sample metrics."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    if not hasattr(cfg_f, "model_type"):
        cfg_f.model_type = "iTransformer"

    ae = TCNAutoEncoder.from_checkpoint(cfg_ae.checkpoint_path, device=device)
    ae.eval()

    forecaster = ForecasterWrapperV2(cfg_f, device)
    forecaster.model.eval()

    agent = ActorCritic(
        latent_dim=cfg_ae.latent_dim, pred_len=cfg_f.pred_len,
        eta=cfg_rl.eta, entropy_coef=0.02, direction=-1.0,
    ).to(device)
    ckpt = torch.load(rl_ckpt, map_location=device, weights_only=False)
    agent.actor.load_state_dict(ckpt["actor_state_dict"])
    agent.critic.load_state_dict(ckpt["critic_state_dict"])
    agent.eval()

    # Global sigma
    _, train_loader = data_provider(cfg_f, "train")
    _stds = []
    for i, batch in enumerate(train_loader):
        if i >= 50: break
        bx, _, _, _ = batch
        _stds.append(bx[:, :, -1].numpy().std(axis=1))
    global_sigma = float(np.concatenate(_stds).mean())

    reward_fn = CFReward(fr=cfg_rl.fr, rho=cfg_rl.rho, direction=-1.0, global_sigma=global_sigma).to(device)

    _, test_loader = data_provider(cfg_f, "test")
    lk = cfg_rl.mask_last_k
    rk = cfg_rl.mask_ramp_k

    # Collect per-sample validity
    per_sample_validity = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= n_batches:
                break
            bx, _, bx_mark, _ = batch
            bx = bx.float().to(device)
            bx_mark = bx_mark.float().to(device)
            x_ot = bx[:, :, -1:]

            z = ae.encode(x_ot)
            y_hat = forecaster.predict_ot(bx, bx_mark)
            z_cf, _, _ = agent.act_deterministic(z, y_hat)
            x_prop = ae.decode(z_cf)
            mask = build_temporal_mask(x_ot.shape[0], x_ot.shape[1], 1, lk, rk, device)
            x_cf = x_ot + mask * (x_prop - x_ot)
            y_cf = forecaster.predict_from_ot(x_ot=x_cf, x_full=bx, x_mark=bx_mark)

            alpha, beta, _ = reward_fn.compute_bounds(y_hat, x_ot=x_ot)
            vr = ((y_cf[:, :, 0] >= alpha) & (y_cf[:, :, 0] <= beta)).float().mean(dim=1)
            per_sample_validity.extend(vr.cpu().numpy().tolist())

    return np.array(per_sample_validity)


def bootstrap_ci(data, n_bootstrap=1000, ci=0.95):
    """Compute bootstrap confidence interval."""
    means = []
    n = len(data)
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        means.append(sample.mean())
    means = np.sort(means)
    lower = means[int((1 - ci) / 2 * n_bootstrap)]
    upper = means[int((1 + ci) / 2 * n_bootstrap)]
    return lower, upper


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("=" * 70)
    print("RL-MCF Evaluation with Statistical Significance")
    print("=" * 70)

    all_results = {}

    for (dataset, model), paths in CONFIGS.items():
        if not os.path.exists(paths["rl_ckpt"]):
            print(f"\n⚠ Skipping {dataset}/{model} — checkpoint not found")
            continue

        print(f"\n── {dataset.upper()} / {model} ──")

        cfg_f = load_config(paths["cfg_f"])
        cfg_ae = load_config(paths["cfg_ae"])
        cfg_rl = load_config(paths["cfg_rl"])

        # Run evaluation on full test set
        per_sample_vr = evaluate_single_run(cfg_f, cfg_ae, cfg_rl, paths["rl_ckpt"],
                                            device, n_batches=9999)

        mean_vr = per_sample_vr.mean()
        std_vr = per_sample_vr.std()
        n = len(per_sample_vr)

        # 95% confidence interval (bootstrap)
        ci_low, ci_high = bootstrap_ci(per_sample_vr)

        # t-test: is validity significantly > 0.5?
        t_stat, p_value = stats.ttest_1samp(per_sample_vr, 0.5)

        all_results[f"{dataset}/{model}"] = {
            "mean": mean_vr,
            "std": std_vr,
            "n_samples": n,
            "ci_95_low": ci_low,
            "ci_95_high": ci_high,
            "t_stat": t_stat,
            "p_value": p_value,
        }

        print(f"  Validity: {mean_vr:.4f} ± {std_vr:.4f}  (n={n})")
        print(f"  95% CI:   [{ci_low:.4f}, {ci_high:.4f}]")
        print(f"  t-test vs 0.5: t={t_stat:.2f}, p={p_value:.2e}")
        if p_value < 0.001:
            print(f"  *** Highly significant (p < 0.001)")
        elif p_value < 0.05:
            print(f"  * Significant (p < 0.05)")

    # Save
    out_path = os.path.join(RESULTS_DIR, "significance_results.json")
    serializable = {k: {kk: float(vv) for kk, vv in v.items()} for k, v in all_results.items()}
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n[Saved] → {out_path}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"{'Config':<25} {'Mean±Std':<18} {'95% CI':<22} {'p-value':<12}")
    print("-" * 70)
    for name, r in all_results.items():
        print(f"{name:<25} {r['mean']:.4f}±{r['std']:.4f}  [{r['ci_95_low']:.4f}, {r['ci_95_high']:.4f}]  {r['p_value']:.2e}")


if __name__ == "__main__":
    main()

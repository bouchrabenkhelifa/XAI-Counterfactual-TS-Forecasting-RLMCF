"""
Validity Analysis: Mean, Std, and Gap Computation across architectures.
"""
import numpy as np

SEP = "=" * 70
DASH = "-" * 60
LINE = chr(9472) * 70  # ─

datasets = {
    "ETTh1": {
        "BaseNN":      [0.5781, 0.6500],
        "BaseGrad":    [0.4781, 0.9743],
        "ForecastCF":  [0.7830, 0.9392],
        "RL-MCF":      [0.9740, 0.9740, 0.5490, 0.9396, 0.9229],
    },
    "ETTh2": {
        "BaseNN":      [0.9083, 0.9010],
        "BaseGrad":    [0.2812, 0.9760],
        "ForecastCF":  [0.4990, 0.9917],
        "RL-MCF":      [0.8100, 0.6675, 0.8462, 0.6950, 0.8296],
    },
    "Weather": {
        "BaseNN":      [0.9094, 0.9604],
        "BaseGrad":    [0.6375, 0.9823],
        "ForecastCF":  [0.9099, 0.9187],
        "RL-MCF":      [0.8443, 0.8208, 0.9490, 0.5203, 0.6609],
    },
}

print(f"\n{SEP}")
print("  VALIDITY ANALYSIS: Mean, Std, and Gap Computation")
print(SEP)

for ds_name, methods in datasets.items():
    print(f"\n{LINE}")
    print(f"  Dataset: {ds_name}")
    print(LINE)
    header = f"  {'Method':<14} {'N':>3}  {'Mean':>8}  {'Std':>8}  Available archs"
    print(header)
    print(f"  {DASH}")

    best_baseline_mean = -1
    best_baseline_name = ""
    best_baseline_std = 0

    for method, scores in methods.items():
        arr = np.array(scores)
        mean = arr.mean()
        std = arr.std(ddof=1) if len(arr) > 1 else 0.0
        n = len(arr)

        if method == "RL-MCF":
            archs = "iTransformer, PatchTST, TimesNet, GRU, DLinear"
        else:
            archs = "iTransformer, DLinear only"

        print(f"  {method:<14} {n:>3}  {mean:>8.4f}  {std:>8.4f}  {archs}")

        if method != "RL-MCF" and mean > best_baseline_mean:
            best_baseline_mean = mean
            best_baseline_name = method
            best_baseline_std = std

    # RL-MCF stats
    rl_arr = np.array(methods["RL-MCF"])
    rl_mean = rl_arr.mean()
    rl_std = rl_arr.std(ddof=1)

    gap = rl_mean - best_baseline_mean
    exceeds = abs(gap) > best_baseline_std

    print(f"\n  >> Best baseline: {best_baseline_name} (mean={best_baseline_mean:.4f}, std={best_baseline_std:.4f})")
    print(f"  >> RL-MCF mean: {rl_mean:.4f} (std={rl_std:.4f})")
    print(f"  >> Gap (RL-MCF - best baseline): {gap:+.4f}")
    print(f"  >> |Gap| > baseline std? {'YES' if exceeds else 'NO'} (|{abs(gap):.4f}| vs {best_baseline_std:.4f})")

print(f"\n{SEP}")
print("  IMPORTANT CAVEAT")
print(SEP)
print("  Baselines only have 2/5 architectures (iTransformer + DLinear).")
print("  RL-MCF has all 5. This makes direct mean comparison asymmetric:")
print("  - Baseline means are computed over 2 points (high variance estimate)")
print("  - RL-MCF means are computed over 5 points")
print("  - Missing baseline values (PatchTST, TimesNet, GRU) could shift means.")
print("  Recommendation: compare only on shared architectures, or report per-arch.")
print(SEP)

"""
Run all baselines (BaseNN, BaseGrad, ForecastCF) for ETTh1 with corrected plausibility
=======================================================================================

Usage:
    python scripts/evals/run_all_baselines_etth1.py
"""

import subprocess
import sys

BASELINES = [
    ("BaseNN", "baselines/BaseNN/run_basenn.py"),
    ("BaseGrad", "baselines/BaseGrad/run_basegrad.py"),
    ("ForecastCF", "baselines/ForecastCF_PyTorch/run_forecastcf_pt.py"),
]

MODELS = ["itransformer", "dlinear"]

def run_baseline(baseline_name, script_path, dataset, model):
    """Run a single baseline."""
    print(f"\n{'='*80}")
    print(f"  Running {baseline_name} — {dataset}/{model}")
    print(f"{'='*80}\n")
    
    cmd = [sys.executable, script_path, "--dataset", dataset, "--model", model, "--seeds", "1"]
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"\n❌ {baseline_name} failed for {model}")
        return False
    else:
        print(f"\n✅ {baseline_name} completed for {model}")
        return True

def main():
    print("\n" + "="*80)
    print("  RUNNING ALL BASELINES FOR ETTh1 (with corrected plausibility)")
    print("="*80 + "\n")
    
    results = {}
    
    for baseline_name, script_path in BASELINES:
        results[baseline_name] = {}
        for model in MODELS:
            success = run_baseline(baseline_name, script_path, "etth1", model)
            results[baseline_name][model] = "✅" if success else "❌"
    
    # Print summary
    print("\n" + "="*80)
    print("  SUMMARY")
    print("="*80 + "\n")
    
    for baseline_name in results:
        print(f"  {baseline_name}:")
        for model in MODELS:
            status = results[baseline_name][model]
            print(f"    {model:15} {status}")
    
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()

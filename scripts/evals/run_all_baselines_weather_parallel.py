"""
Run all baselines (BaseNN, BaseGrad, ForecastCF) for Weather in parallel
=========================================================================

Usage:
    python scripts/evals/run_all_baselines_weather_parallel.py
"""

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

BASELINES = [
    ("BaseNN", "baselines/BaseNN/run_basenn.py"),
    ("BaseGrad", "baselines/BaseGrad/run_basegrad.py"),
    ("ForecastCF", "baselines/ForecastCF_PyTorch/run_forecastcf_pt.py"),
]

MODELS = ["itransformer", "dlinear"]

def run_baseline(baseline_name, script_path, dataset, model):
    """Run a single baseline."""
    print(f"\n🚀 Starting {baseline_name} — {dataset}/{model}")
    
    start_time = time.time()
    cmd = [sys.executable, script_path, "--dataset", dataset, "--model", model, "--seeds", "1"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"\n❌ {baseline_name}/{model} failed after {elapsed:.1f}s")
        print(f"Error: {result.stderr[:200]}")
        return (baseline_name, model, False, elapsed)
    else:
        print(f"\n✅ {baseline_name}/{model} completed in {elapsed:.1f}s")
        return (baseline_name, model, True, elapsed)

def main():
    print("\n" + "="*80)
    print("  RUNNING ALL BASELINES FOR WEATHER (PARALLEL)")
    print("="*80 + "\n")
    
    # Create all tasks
    tasks = []
    for baseline_name, script_path in BASELINES:
        for model in MODELS:
            tasks.append((baseline_name, script_path, "weather", model))
    
    print(f"📋 Total tasks: {len(tasks)}")
    print(f"🔄 Running in parallel with {len(tasks)} workers\n")
    
    results = {}
    start_time = time.time()
    
    # Run all tasks in parallel
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(run_baseline, baseline_name, script_path, dataset, model): (baseline_name, model)
            for baseline_name, script_path, dataset, model in tasks
        }
        
        for future in as_completed(futures):
            baseline_name, model, success, elapsed = future.result()
            
            if baseline_name not in results:
                results[baseline_name] = {}
            
            results[baseline_name][model] = {
                "success": success,
                "time": elapsed
            }
    
    total_time = time.time() - start_time
    
    # Print summary
    print("\n" + "="*80)
    print("  SUMMARY")
    print("="*80 + "\n")
    
    for baseline_name in ["BaseNN", "BaseGrad", "ForecastCF"]:
        if baseline_name not in results:
            continue
        print(f"  {baseline_name}:")
        for model in MODELS:
            if model not in results[baseline_name]:
                continue
            status = "✅" if results[baseline_name][model]["success"] else "❌"
            elapsed = results[baseline_name][model]["time"]
            print(f"    {model:15} {status}  ({elapsed:.1f}s)")
        print()
    
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()

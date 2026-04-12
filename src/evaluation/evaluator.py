import numpy as np

from .validity_metrics import (
    delta_mean,
    relative_reduction,
    target_gap,
    success_indicator,
)

from .proximity_metrics import (
    l1_distance,
    l2_distance,
    euclidean_distance,
    manhattan_distance,
    dtw_distance,
)

from .plausibility_metrics import (
    plausibility_score,
    plausibility_score_all,
)

from .realism_metrics import (
    roughness,
    roughness_ratio,
    derivative_distance,
    second_derivative_distance,
    temporal_consistency,
    autocorrelation_preservation,
    spectral_similarity,
)

from .sparsity_metrics import (
    sparsity_ratio,
    mean_change_magnitude,
    segment_sparsity,
)

from .actionability_metrics import (
    rate_of_change_feasibility,
    action_efficiency_ratio,
    reachability_score,
    causal_compactness,
    forecast_monotonicity,
)


class CounterfactualEvaluator:
    def __init__(self, plausibility_model=None, rho=0.10, x_train=None):
    
        self.plausibility_model = plausibility_model
        self.rho = rho
        self.x_train = x_train

    def evaluate_batch(
        self,
        x,
        x_cf,
        y_hat,
        y_cf,
        include_dtw=False,
        forecaster=None,
        include_reachability=False,
    ):
    
        results = {}

        # ------------------------------------------------------------------
        # Validity
        # ------------------------------------------------------------------
        results["delta_mean"]         = delta_mean(y_hat, y_cf)
        results["relative_reduction"] = relative_reduction(y_hat, y_cf)
        results["target_gap"]         = target_gap(y_hat, y_cf, rho=self.rho)
        results["success"]            = success_indicator(y_hat, y_cf, rho=self.rho)

        # ------------------------------------------------------------------
        # Proximity
        # ------------------------------------------------------------------
        results["l1"]        = l1_distance(x, x_cf)
        results["l2"]        = l2_distance(x, x_cf)
        results["euclidean"] = euclidean_distance(x, x_cf)
        results["manhattan"] = manhattan_distance(x, x_cf)

        if include_dtw:
            results["dtw"] = dtw_distance(x, x_cf)

        # ------------------------------------------------------------------
        # Plausibility
        # ------------------------------------------------------------------
        if self.plausibility_model is not None:
            results["plausibility_x"]  = plausibility_score(self.plausibility_model, x)
            results["plausibility_cf"] = plausibility_score(self.plausibility_model, x_cf)

            all_pl_x = plausibility_score_all(self.plausibility_model, x)
            for k, v in all_pl_x.items():
                results[f"plausibility_x_{k}"] = v

            all_pl_cf = plausibility_score_all(self.plausibility_model, x_cf)
            for k, v in all_pl_cf.items():
                results[f"plausibility_cf_{k}"] = v

        # ------------------------------------------------------------------
        # Realism
        # ------------------------------------------------------------------
        results["roughness_x"]               = roughness(x)
        results["roughness_cf"]              = roughness(x_cf)
        results["roughness_ratio"]           = roughness_ratio(x, x_cf)
        results["derivative_distance"]       = derivative_distance(x, x_cf)
        results["second_derivative_distance"] = second_derivative_distance(x, x_cf)
        results["temporal_consistency"]      = temporal_consistency(x_cf)
        results["autocorrelation_similarity"] = autocorrelation_preservation(x, x_cf)
        results["spectral_similarity"]       = spectral_similarity(x, x_cf)

        # ------------------------------------------------------------------
        # Sparsity
        # ------------------------------------------------------------------
        results["sparsity_ratio"]   = sparsity_ratio(x, x_cf)
        results["change_magnitude"] = mean_change_magnitude(x, x_cf)
        results["segment_sparsity"] = segment_sparsity(x, x_cf)

        # ------------------------------------------------------------------
        # Actionability
        # ------------------------------------------------------------------
        if self.x_train is not None:
            results["rate_of_change_feasibility"] = rate_of_change_feasibility(
                x, x_cf, self.x_train
            )

        results["action_efficiency_ratio"] = action_efficiency_ratio(
            x, x_cf, y_hat, y_cf
        )

        results["causal_compactness"] = causal_compactness(x, x_cf)

        if self.x_train is not None and include_reachability:
            results["reachability_score"] = reachability_score(
                x, x_cf, self.x_train
            )

        if forecaster is not None:
            results["forecast_monotonicity"] = forecast_monotonicity(
                x, x_cf, forecaster
            )

        return results

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    def summarize(self, results_dict):
        summary = {}
        for k, v in results_dict.items():
            v = np.asarray(v)
            summary[k] = float(np.mean(v))
        return summary

    def summarize_with_std(self, results_dict):
        summary = {}
        for k, v in results_dict.items():
            v = np.asarray(v)
            summary[k] = {
                "mean": float(np.mean(v)),
                "std":  float(np.std(v)),
                "n":    int(v.size),
            }
        return summary

    def summarize_selected(self, results_dict, keys):
        out = {}
        for k in keys:
            if k not in results_dict:
                continue
            v = np.asarray(results_dict[k])
            out[k] = {
                "mean": float(np.mean(v)),
                "std":  float(np.std(v)),
                "n":    int(v.size),
            }
        return out

    # ------------------------------------------------------------------
    # Pretty print
    # ------------------------------------------------------------------

    def pretty_print_summary(self, summary):
        print("\n===== Counterfactual Evaluation Summary =====")

        print("\n[Validity]")
        print(f"  Delta mean              : {summary.get('delta_mean', float('nan')):.4f}")
        print(f"  Relative reduction      : {summary.get('relative_reduction', float('nan')):.4f}")
        print(f"  Target gap              : {summary.get('target_gap', float('nan')):.4f}")
        print(f"  Success rate            : {summary.get('success', float('nan')):.4f}")

        print("\n[Proximity]")
        print(f"  L1(X, Xcf)              : {summary.get('l1', float('nan')):.4f}")
        print(f"  L2(X, Xcf)              : {summary.get('l2', float('nan')):.4f}")
        print(f"  Euclidean(X, Xcf)       : {summary.get('euclidean', float('nan')):.4f}")
        print(f"  Manhattan(X, Xcf)       : {summary.get('manhattan', float('nan')):.4f}")
        if "dtw" in summary:
            print(f"  DTW(X, Xcf)             : {summary.get('dtw', float('nan')):.4f}")

        print("\n[Plausibility]")
        print(f"  Plausibility(X)         : {summary.get('plausibility_x', float('nan')):.4f}")
        print(f"  Plausibility(Xcf)       : {summary.get('plausibility_cf', float('nan')):.4f}")

        for k in ["if", "lof", "ocsvm", "ensemble"]:
            key_x  = f"plausibility_x_{k}"
            key_cf = f"plausibility_cf_{k}"
            if key_x in summary:
                print(f"  Plausibility(X) [{k}]   : {summary[key_x]:.4f}")
            if key_cf in summary:
                print(f"  Plausibility(Xcf)[{k}]  : {summary[key_cf]:.4f}")

        print("\n[Realism / Temporal Quality]")
        print(f"  Roughness(X)            : {summary.get('roughness_x', float('nan')):.4f}")
        print(f"  Roughness(Xcf)          : {summary.get('roughness_cf', float('nan')):.4f}")
        print(f"  Roughness ratio         : {summary.get('roughness_ratio', float('nan')):.4f}")
        print(f"  Derivative distance     : {summary.get('derivative_distance', float('nan')):.4f}")
        print(f"  2nd-derivative distance : {summary.get('second_derivative_distance', float('nan')):.4f}")
        print(f"  Temporal consistency    : {summary.get('temporal_consistency', float('nan')):.4f}")
        print(f"  Autocorr similarity     : {summary.get('autocorrelation_similarity', float('nan')):.4f}")
        print(f"  Spectral similarity     : {summary.get('spectral_similarity', float('nan')):.4f}")

        print("\n[Sparsity]")
        print(f"  Sparsity ratio          : {summary.get('sparsity_ratio', float('nan')):.4f}")
        print(f"  Change magnitude        : {summary.get('change_magnitude', float('nan')):.4f}")
        print(f"  Segment sparsity        : {summary.get('segment_sparsity', float('nan')):.4f}")

        print("\n[Actionability]")
        print(f"  Rate of change feasib.  : {summary.get('rate_of_change_feasibility', float('nan')):.4f}")
        print(f"  Action efficiency ratio : {summary.get('action_efficiency_ratio', float('nan')):.4f}")
        print(f"  Reachability score      : {summary.get('reachability_score', float('nan')):.4f}")
        print(f"  Causal compactness      : {summary.get('causal_compactness', float('nan')):.4f}")
        print(f"  Forecast monotonicity   : {summary.get('forecast_monotonicity', float('nan')):.4f}")

        print("=============================================\n")


__all__ = [
    "CounterfactualEvaluator",
]
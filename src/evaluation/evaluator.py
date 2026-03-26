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


class CounterfactualEvaluator:
    def __init__(self, plausibility_model=None, rho=0.10):
        self.plausibility_model = plausibility_model
        self.rho = rho

    def evaluate_batch(self, x, x_cf, y_hat, y_cf, include_dtw=False):
        results = {}

        # Validity
        results["delta_mean"] = delta_mean(y_hat, y_cf)
        results["relative_reduction"] = relative_reduction(y_hat, y_cf)
        results["target_gap"] = target_gap(y_hat, y_cf, rho=self.rho)
        results["success"] = success_indicator(y_hat, y_cf, rho=self.rho)

        # Proximity
        results["l1"] = l1_distance(x, x_cf)
        results["l2"] = l2_distance(x, x_cf)
        results["euclidean"] = euclidean_distance(x, x_cf)
        results["manhattan"] = manhattan_distance(x, x_cf)

        if include_dtw:
            results["dtw"] = dtw_distance(x, x_cf)

        # Plausibility
        if self.plausibility_model is not None:
            results["plausibility"] = plausibility_score(
                self.plausibility_model, x_cf
            )

            all_pl = plausibility_score_all(self.plausibility_model, x_cf)
            for k, v in all_pl.items():
                results[f"plausibility_{k}"] = v

        # Realism
        results["roughness_x"] = roughness(x)
        results["roughness_cf"] = roughness(x_cf)
        results["roughness_ratio"] = roughness_ratio(x, x_cf)
        results["derivative_distance"] = derivative_distance(x, x_cf)
        results["second_derivative_distance"] = second_derivative_distance(x, x_cf)
        results["temporal_consistency"] = temporal_consistency(x_cf)
        results["autocorrelation_similarity"] = autocorrelation_preservation(x, x_cf)
        results["spectral_similarity"] = spectral_similarity(x, x_cf)

        # Sparsity
        results["sparsity_ratio"] = sparsity_ratio(x, x_cf)
        results["change_magnitude"] = mean_change_magnitude(x, x_cf)
        results["segment_sparsity"] = segment_sparsity(x, x_cf)

        return results

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
                "std": float(np.std(v)),
                "n": int(v.size),
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
                "std": float(np.std(v)),
                "n": int(v.size),
            }
        return out


__all__ = [
    "CounterfactualEvaluator",
]
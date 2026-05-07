"""
ForecastCF-PyTorch: Réimplémentation native PyTorch de ForecastCF (Wang et al., ICDM 2023).
Utilise les bornes RL-MCF pour une comparaison équitable.
"""

from .forecastcf_pt import ForecastCFPyTorch

__all__ = ["ForecastCFPyTorch"]

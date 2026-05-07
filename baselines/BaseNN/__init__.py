"""
BaseNN: Baseline 1-nearest-neighbour.
Pour chaque sample test, retourne le sample d'entraînement dont le forecast
est le plus proche du centre de la bande cible [α, β].
"""

from .basenn import BaseNN

__all__ = ["BaseNN"]

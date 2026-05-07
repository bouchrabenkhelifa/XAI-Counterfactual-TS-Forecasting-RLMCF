"""
BaseNN: Baseline 1-nearest-neighbour.

Pour chaque sample test, trouve le sample d'entraînement dont le forecast
est le plus proche du centre de la bande cible [α, β].

Algorithme:
    1. Calculer target = (alpha + beta) / 2 pour chaque sample test
    2. Pour chaque sample test, trouver le NN dans Y_hat_train le plus proche de target
    3. Retourner X_train[nn_idx] comme counterfactual
"""

import numpy as np


class BaseNN:
    """
    Baseline 1-nearest-neighbour sur les forecasts du train set.
    
    Utilise les bornes RL-MCF pour définir la cible.
    """
    
    def __init__(self):
        self.X_train = None
        self.Y_hat_train = None
    
    def fit(self, X_train, Y_hat_train):
        """
        Stocke le train set et ses forecasts.
        
        Parameters
        ----------
        X_train : np.ndarray
            Séries d'entraînement (OT channel), shape [M, BH, 1]
        Y_hat_train : np.ndarray
            Forecasts d'entraînement, shape [M, H, 1]
        """
        self.X_train = X_train          # [M, BH, 1]
        self.Y_hat_train = Y_hat_train  # [M, H, 1]
        print(f"[BaseNN] Fitted with {len(X_train)} train samples")
    
    def transform(self, X_test, alphas, betas):
        """
        Pour chaque sample test, trouve le NN dans Y_hat_train
        le plus proche de target = (alpha+beta)/2.
        
        Parameters
        ----------
        X_test : np.ndarray
            Séries test (OT channel), shape [N, BH, 1]
        alphas : np.ndarray
            Bornes inférieures (bornes RL-MCF), shape [N, H]
        betas : np.ndarray
            Bornes supérieures (bornes RL-MCF), shape [N, H]
        
        Returns
        -------
        X_cf : np.ndarray
            Séries contrefactuelles, shape [N, BH, 1]
        Y_cf : np.ndarray
            Forecasts contrefactuels, shape [N, H, 1]
        """
        if self.X_train is None or self.Y_hat_train is None:
            raise RuntimeError("BaseNN not fitted. Call fit() first.")
        
        N = X_test.shape[0]
        
        # Calculer les targets (centre de la bande)
        targets = (alphas + betas) / 2.0  # [N, H]
        
        # Flatten Y_hat_train pour la recherche NN
        ytr = self.Y_hat_train[:, :, 0]  # [M, H]
        
        X_cf_list = []
        Y_cf_list = []
        
        for i in range(N):
            target_i = targets[i]  # [H]
            
            # Distance L2 entre chaque forecast train et target_i
            dists = np.linalg.norm(ytr - target_i[np.newaxis, :], axis=1)  # [M]
            
            # Trouver le nearest neighbor
            nn_idx = int(np.argmin(dists))
            
            # Retourner le sample train correspondant
            X_cf_list.append(self.X_train[nn_idx])
            Y_cf_list.append(self.Y_hat_train[nn_idx])
        
        X_cf = np.stack(X_cf_list, axis=0)  # [N, BH, 1]
        Y_cf = np.stack(Y_cf_list, axis=0)  # [N, H, 1]
        
        return X_cf, Y_cf

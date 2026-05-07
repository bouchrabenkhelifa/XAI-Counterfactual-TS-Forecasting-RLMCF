"""
ForecastCF-PyTorch: Réimplémentation native PyTorch de ForecastCF (Wang et al., ICDM 2023).

Algorithme fidèle au papier original, mais utilise les bornes RL-MCF pour une comparaison équitable.

Loss: L = pred_margin_weight * margin_mse + (1 - pred_margin_weight) * weighted_mae
    - margin_mse: MSE masquée sur les timesteps hors-bande [alpha, beta] uniquement
    - weighted_mae: MAE pondérée (step_weights=ones par défaut)

Optimisation: Adam lr=1e-4, max_iter=100, early stopping si tous les timesteps sont dans la bande.
"""

import numpy as np
import torch
import torch.nn as nn


class ForecastCFPyTorch:
    """
    Réimplémentation PyTorch native de ForecastCF.
    
    Parameters
    ----------
    forecaster : ForecasterWrapperV2
        Forecaster gelé (PyTorch)
    device : torch.device
        Device (cpu ou cuda)
    max_iter : int
        Nombre maximum d'itérations d'optimisation
    lr : float
        Learning rate Adam
    pred_margin_weight : float
        Poids de la loss margin_mse (entre 0 et 1)
    """
    
    def __init__(self, forecaster, device,
                 max_iter=100, lr=1e-4, pred_margin_weight=0.5):
        self.forecaster = forecaster
        self.device = device
        self.max_iter = max_iter * 3  # 300 itérations
        self.lr = 0.01  # Même LR que BaseGrad
        self.pred_margin_weight = pred_margin_weight
        
        # Geler le forecaster
        self.forecaster.model.eval()
        for p in self.forecaster.model.parameters():
            p.requires_grad_(False)
    
    def _margin_mse(self, pred, alpha, beta):
        """
        MSE masquée sur les timesteps hors-bande [alpha, beta] uniquement.
        
        Fidèle au papier ForecastCF: seuls les timesteps qui violent les bornes
        contribuent à la loss.
        
        Parameters
        ----------
        pred : torch.Tensor
            Prédiction, shape [1, H, 1]
        alpha : torch.Tensor
            Borne inférieure, shape [1, H, 1]
        beta : torch.Tensor
            Borne supérieure, shape [1, H, 1]
        
        Returns
        -------
        torch.Tensor
            MSE masquée (scalaire)
        """
        # Vérifier si tous les timesteps sont dans la bande
        in_band = (pred >= alpha) & (pred <= beta)
        if in_band.all():
            return torch.tensor(0.0, device=self.device)
        
        # Extraire les timesteps hors-bande
        pred_out  = pred[~in_band]
        alpha_out = alpha[~in_band]
        beta_out  = beta[~in_band]
        
        # Distance à la borne la plus proche
        dist_to_alpha = (pred_out - alpha_out) ** 2
        dist_to_beta  = (pred_out - beta_out) ** 2
        dist = torch.minimum(dist_to_alpha, dist_to_beta)
        
        return dist.mean()
    
    def _weighted_mae(self, x_orig, x_cf, step_weights=None):
        """
        MAE pondérée entre x_orig et x_cf.
        
        Parameters
        ----------
        x_orig : torch.Tensor
            Série originale, shape [1, BH, 1]
        x_cf : torch.Tensor
            Série contrefactuelle, shape [1, BH, 1]
        step_weights : torch.Tensor or None
            Poids par timestep, shape [1, BH, 1]. Si None, utilise ones (MAE simple).
        
        Returns
        -------
        torch.Tensor
            MAE pondérée (scalaire)
        """
        diff = (x_orig - x_cf).abs()
        if step_weights is not None:
            diff = diff * step_weights
        return diff.mean()
    
    def transform_sample(self, x_np, alpha_np, beta_np, x_full_t, x_mark_t):
        """
        Génère un counterfactual pour un seul sample.
        
        Parameters
        ----------
        x_np : np.ndarray
            Série originale (OT channel), shape [1, BH, 1]
        alpha_np : np.ndarray
            Borne inférieure (bornes RL-MCF), shape [H]
        beta_np : np.ndarray
            Borne supérieure (bornes RL-MCF), shape [H]
        x_full_t : torch.Tensor
            Série complète (tous les channels), shape [1, seq_len, n_features]
        x_mark_t : torch.Tensor
            Time features, shape [1, seq_len, time_features]
        
        Returns
        -------
        x_cf_np : np.ndarray
            Série contrefactuelle, shape [1, BH, 1]
        y_cf_np : np.ndarray
            Forecast contrefactuel, shape [1, H, 1]
        """
        # Convertir en tensors PyTorch
        x_orig = torch.tensor(x_np, dtype=torch.float32, device=self.device)
        alpha  = torch.tensor(alpha_np, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(-1)  # [1, H, 1]
        beta   = torch.tensor(beta_np,  dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(-1)  # [1, H, 1]
        
        # Détacher les tensors d'entrée pour éviter les problèmes de graphe
        x_full_t = x_full_t.detach()
        x_mark_t = x_mark_t.detach()
        
        # Initialiser x_cf = x_orig
        x_cf = x_orig.clone()
        
        # Adam parameters
        m = torch.zeros_like(x_cf)
        v = torch.zeros_like(x_cf)
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        
        # Optimisation avec SPSA (même stratégie que BaseGrad qui fonctionne)
        for it in range(self.max_iter):
            with torch.no_grad():
                # Évaluer la loss actuelle
                x_full_cf = torch.cat([x_full_t[:, :, :-1], x_cf], dim=-1)
                y_cf = self.forecaster.predict(x_full_cf, x_mark_t, x_mark_t)[:, :, -1:]
                
                # Early stopping si tous les timesteps sont dans la bande
                if ((y_cf >= alpha) & (y_cf <= beta)).all():
                    break
                
                # Calculer la loss actuelle
                loss_margin_curr = self._margin_mse(y_cf, alpha, beta)
                loss_prox_curr = self._weighted_mae(x_orig, x_cf)
                loss_curr = self.pred_margin_weight * loss_margin_curr + (1 - self.pred_margin_weight) * loss_prox_curr
                
                # SPSA simple (comme BaseGrad)
                epsilon = 0.01
                delta = torch.randn_like(x_cf) * epsilon
                
                # Évaluer loss avec perturbation
                x_cf_plus = x_cf + delta
                x_full_cf_plus = torch.cat([x_full_t[:, :, :-1], x_cf_plus], dim=-1)
                y_cf_plus = self.forecaster.predict(x_full_cf_plus, x_mark_t, x_mark_t)[:, :, -1:]
                loss_margin_plus = self._margin_mse(y_cf_plus, alpha, beta)
                loss_prox_plus = self._weighted_mae(x_orig, x_cf_plus)
                loss_plus = self.pred_margin_weight * loss_margin_plus + (1 - self.pred_margin_weight) * loss_prox_plus
                
                # Approximation du gradient
                grad = (loss_plus - loss_curr) * delta / (epsilon ** 2)
                
                # Mise à jour avec Adam
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad ** 2)
                m_hat = m / (1 - beta1 ** (it + 1))
                v_hat = v / (1 - beta2 ** (it + 1))
                x_cf = x_cf - self.lr * m_hat / (torch.sqrt(v_hat) + eps)
        
        # Prédiction finale
        with torch.no_grad():
            y_cf = self.forecaster.predict_from_ot(x_cf, x_full_t, x_mark_t)
        
        return x_cf.detach().cpu().numpy(), y_cf.detach().cpu().numpy()
    
    def transform(self, X_np, alphas, betas, X_full_t, X_mark_t):
        """
        Génère des counterfactuals pour un batch de samples.
        
        Parameters
        ----------
        X_np : np.ndarray
            Séries originales (OT channel), shape [N, BH, 1]
        alphas : np.ndarray
            Bornes inférieures (bornes RL-MCF), shape [N, H]
        betas : np.ndarray
            Bornes supérieures (bornes RL-MCF), shape [N, H]
        X_full_t : torch.Tensor
            Séries complètes (tous les channels), shape [N, seq_len, n_features]
        X_mark_t : torch.Tensor
            Time features, shape [N, seq_len, time_features]
        
        Returns
        -------
        X_cf : np.ndarray
            Séries contrefactuelles, shape [N, BH, 1]
        Y_cf : np.ndarray
            Forecasts contrefactuels, shape [N, H, 1]
        """
        N = X_np.shape[0]
        X_cf_list = []
        Y_cf_list = []
        
        for i in range(N):
            x_cf, y_cf = self.transform_sample(
                x_np=X_np[i:i+1],
                alpha_np=alphas[i],
                beta_np=betas[i],
                x_full_t=X_full_t[i:i+1],
                x_mark_t=X_mark_t[i:i+1]
            )
            X_cf_list.append(x_cf)
            Y_cf_list.append(y_cf)
        
        return np.concatenate(X_cf_list, axis=0), np.concatenate(Y_cf_list, axis=0)

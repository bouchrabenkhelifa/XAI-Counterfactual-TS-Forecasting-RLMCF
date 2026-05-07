"""
BaseGrad: Baseline de descente de gradient directe.

Optimise x_cf par gradient descent avec hinge loss sur TOUS les timesteps
(contrairement à ForecastCF qui masque les timesteps déjà dans la bande).

Algorithme:
    1. Initialiser x_cf = x_orig
    2. Pour max_iter itérations:
        a. Calculer y_cf = forecaster(x_cf)
        b. Calculer L_validity = mean(relu(α - y_cf) + relu(y_cf - β))
        c. Calculer L_proximity = mean(|x_cf - x_orig|)
        d. Calculer L = w_v * L_validity + w_p * L_proximity
        e. Gradient descent: x_cf -= lr * ∇L
        f. Clipper x_cf dans [x_min - 3σ, x_max + 3σ]
    3. Retourner x_cf final
"""

import numpy as np
import torch
import torch.nn.functional as F


class BaseGrad:
    """
    Baseline de descente de gradient directe avec hinge loss.
    
    Utilise les bornes RL-MCF pour définir la cible.
    """
    
    def __init__(self, forecaster, device, lr=0.01, max_iter=300, 
                 w_validity=1.0, w_proximity=0.5):
        """
        Parameters
        ----------
        forecaster : ForecasterWrapperV2
            Forecaster gelé (PyTorch)
        device : torch.device
            Device (cpu ou cuda)
        lr : float
            Learning rate pour Adam
        max_iter : int
            Nombre maximum d'itérations
        w_validity : float
            Poids de la loss de validity
        w_proximity : float
            Poids de la loss de proximity
        """
        self.forecaster = forecaster
        self.device = device
        self.lr = lr
        self.max_iter = max_iter
        self.w_validity = w_validity
        self.w_proximity = w_proximity
        
        # Geler le forecaster mais le mettre en mode train pour éviter les in-place ops
        self.forecaster.model.train()
        for p in self.forecaster.model.parameters():
            p.requires_grad_(False)
    
    def _compute_loss(self, x_cf, x_orig, alpha_t, beta_t, x_full_orig, x_mark):
        """
        Calcule la loss totale.
        
        Parameters
        ----------
        x_cf : torch.Tensor
            Série contrefactuelle (OT channel), shape [1, BH, 1]
        x_orig : torch.Tensor
            Série originale (OT channel), shape [1, BH, 1]
        alpha_t : torch.Tensor
            Borne inférieure, shape [1, H, 1]
        beta_t : torch.Tensor
            Borne supérieure, shape [1, H, 1]
        x_full_orig : torch.Tensor
            Série complète originale, shape [1, seq_len, n_features]
        x_mark : torch.Tensor
            Time features, shape [1, seq_len, time_features]
        
        Returns
        -------
        loss : torch.Tensor
            Loss totale (scalaire)
        loss_validity : torch.Tensor
            Loss de validity (scalaire)
        loss_proximity : torch.Tensor
            Loss de proximity (scalaire)
        """
        # Reconstruire x_full avec le nouveau x_cf (complètement indépendant)
        # Ne pas utiliser x_full_orig.clone() car cela peut hériter du graphe
        x_full_cf = torch.cat([x_full_orig[:, :, :-1].detach(), x_cf], dim=-1)
        
        # Prédire le forecast AVEC GRADIENTS
        # Utiliser predict_with_grad() pour permettre le backprop
        y_cf = self.forecaster.predict_with_grad(x_full_cf, x_mark, x_mark)  # [1, H, n_features]
        y_cf = y_cf[:, :, -1:]  # Extraire OT channel [1, H, 1]
        
        # Loss de validity: hinge loss sur TOUS les timesteps
        # L_validity = mean(relu(α - y_cf) + relu(y_cf - β))
        lower_violation = F.relu(alpha_t - y_cf)  # [1, H, 1]
        upper_violation = F.relu(y_cf - beta_t)   # [1, H, 1]
        loss_validity = (lower_violation + upper_violation).mean()
        
        # Loss de proximity: L1 distance
        loss_proximity = torch.abs(x_cf - x_orig).mean()
        
        # Loss totale
        loss = self.w_validity * loss_validity + self.w_proximity * loss_proximity
        
        return loss, loss_validity, loss_proximity
    
    def transform_sample(self, x_np, alpha_np, beta_np, x_full_orig, x_mark):
        """
        Génère un counterfactual pour un seul sample via gradient descent.
        
        Parameters
        ----------
        x_np : np.ndarray
            Série originale (OT channel), shape [1, BH, 1]
        alpha_np : np.ndarray
            Borne inférieure (bornes RL-MCF), shape [H]
        beta_np : np.ndarray
            Borne supérieure (bornes RL-MCF), shape [H]
        x_full_orig : torch.Tensor
            Série complète originale, shape [1, seq_len, n_features]
        x_mark : torch.Tensor
            Time features, shape [1, seq_len, time_features]
        
        Returns
        -------
        x_cf_np : np.ndarray
            Série contrefactuelle, shape [1, BH, 1]
        y_cf_np : np.ndarray
            Forecast contrefactuel, shape [1, H, 1]
        """
        # Convertir en tensors
        x_orig = torch.tensor(x_np, dtype=torch.float32, device=self.device)
        alpha_t = torch.tensor(alpha_np, dtype=torch.float32, device=self.device).reshape(1, -1, 1)
        beta_t = torch.tensor(beta_np, dtype=torch.float32, device=self.device).reshape(1, -1, 1)
        
        # Détacher x_full_orig pour éviter les problèmes de graphe de calcul
        x_full_orig = x_full_orig.detach()
        x_mark = x_mark.detach()
        
        # Initialiser x_cf = x_orig (avec gradient)
        x_cf = x_orig.clone()
        
        # Calculer les bornes de clipping
        x_min = float(x_np.min())
        x_max = float(x_np.max())
        sigma = float(x_np.std())
        if sigma < 1e-6:
            sigma = 1.0
        clip_min = x_min - 3.0 * sigma
        clip_max = x_max + 3.0 * sigma
        
        # Adam parameters
        m = torch.zeros_like(x_cf)
        v = torch.zeros_like(x_cf)
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        
        # Gradient descent avec approximation numérique pour iTransformer
        for it in range(self.max_iter):
            with torch.no_grad():
                # Évaluer la loss actuelle
                x_full_cf = torch.cat([x_full_orig[:, :, :-1], x_cf], dim=-1)
                y_cf = self.forecaster.predict(x_full_cf, x_mark, x_mark)[:, :, -1:]
                lower_violation = F.relu(alpha_t - y_cf)
                upper_violation = F.relu(y_cf - beta_t)
                loss_validity = (lower_violation + upper_violation).mean()
                loss_proximity = torch.abs(x_cf - x_orig).mean()
                loss_curr = self.w_validity * loss_validity + self.w_proximity * loss_proximity
                
                # Gradient numérique par perturbation aléatoire (SPSA - Simultaneous Perturbation Stochastic Approximation)
                epsilon = 0.01
                delta = torch.randn_like(x_cf) * epsilon
                
                # Évaluer loss avec perturbation positive
                x_cf_plus = x_cf + delta
                x_full_cf_plus = torch.cat([x_full_orig[:, :, :-1], x_cf_plus], dim=-1)
                y_cf_plus = self.forecaster.predict(x_full_cf_plus, x_mark, x_mark)[:, :, -1:]
                lower_violation_plus = F.relu(alpha_t - y_cf_plus)
                upper_violation_plus = F.relu(y_cf_plus - beta_t)
                loss_validity_plus = (lower_violation_plus + upper_violation_plus).mean()
                loss_proximity_plus = torch.abs(x_cf_plus - x_orig).mean()
                loss_plus = self.w_validity * loss_validity_plus + self.w_proximity * loss_proximity_plus
                
                # Approximation du gradient: g ≈ (f(x+δ) - f(x)) / δ
                grad = (loss_plus - loss_curr) * delta / (epsilon ** 2)
                
                # Mise à jour avec Adam
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad ** 2)
                m_hat = m / (1 - beta1 ** (it + 1))
                v_hat = v / (1 - beta2 ** (it + 1))
                x_cf = x_cf - self.lr * m_hat / (torch.sqrt(v_hat) + eps)
                x_cf = x_cf.clamp(clip_min, clip_max)
                
                # Early stopping
                if loss_validity.item() < 1e-4:
                    break
        
        # Calculer le forecast final
        with torch.no_grad():
            x_full_cf = x_full_orig.clone()
            x_full_cf[:, :, -1:] = x_cf
            y_cf = self.forecaster.predict_ot(x_full_cf, x_mark)
        
        return x_cf.cpu().numpy(), y_cf.cpu().numpy()
    
    def transform(self, X_np, alphas, betas, X_full, X_mark):
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
        X_full : torch.Tensor
            Séries complètes, shape [N, seq_len, n_features]
        X_mark : torch.Tensor
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
                x_full_orig=X_full[i:i+1],
                x_mark=X_mark[i:i+1]
            )
            X_cf_list.append(x_cf)
            Y_cf_list.append(y_cf)
        
        return np.concatenate(X_cf_list, axis=0), np.concatenate(Y_cf_list, axis=0)

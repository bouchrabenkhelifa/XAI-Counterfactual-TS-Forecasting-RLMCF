import numpy as np
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------------------
# 1. Rate of Change Feasibility
# ---------------------------------------------------------------------------

def rate_of_change_feasibility(x, x_cf, x_train, percentile=95):
    """
    Mesure si la vitesse de variation du delta (x_cf - x) est physiquement
    réalisable, en la comparant aux vitesses observées dans x_train.

    Parameters
    ----------
    x       : np.ndarray, shape [B, T] or [B, T, C]
    x_cf    : np.ndarray, shape [B, T] or [B, T, C]
    x_train : np.ndarray, shape [N, T] or [N, T, C]
    percentile : int, borne haute des vitesses réalistes (défaut : 95)

    Returns
    -------
    np.ndarray, shape [B] — proportion de pas de temps où le delta varie
    de façon réaliste (1.0 = parfaitement actionnable).
    """
    x      = np.asarray(x,      dtype=np.float64)
    x_cf   = np.asarray(x_cf,   dtype=np.float64)
    x_train = np.asarray(x_train, dtype=np.float64)

    # Vitesses observées dans x_train (différences absolues consécutives)
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    train_velocities = np.abs(np.diff(x_train_flat, axis=1)).reshape(-1)
    max_realistic_rate = np.percentile(train_velocities, percentile)

    B = x.shape[0]
    x_flat    = x.reshape(B, -1)
    x_cf_flat = x_cf.reshape(B, -1)

    scores = []
    for i in range(B):
        delta          = x_cf_flat[i] - x_flat[i]
        delta_velocity = np.abs(np.diff(delta))
        feasible_steps = np.mean(delta_velocity <= max_realistic_rate)
        scores.append(feasible_steps)

    return np.asarray(scores, dtype=np.float32)


# ---------------------------------------------------------------------------
# 2. Action Efficiency Ratio (AER)
# ---------------------------------------------------------------------------

def action_efficiency_ratio(x, x_cf, y_hat, y_cf, eps=1e-8):
    """
    Rapport entre le gain de forecast obtenu et l'effort d'intervention.
    AER élevé → petit changement, grand effet (actionnable et efficient).
    AER faible → grand changement, petit effet (CF force le modèle).

    Parameters
    ----------
    x     : np.ndarray, shape [B, T] or [B, T, C]
    x_cf  : np.ndarray, shape [B, T] or [B, T, C]
    y_hat : np.ndarray, shape [B, H] or [B, H, 1]  — forecast original
    y_cf  : np.ndarray, shape [B, H] or [B, H, 1]  — forecast contrefactuel

    Returns
    -------
    np.ndarray, shape [B]
    """
    x     = np.asarray(x,     dtype=np.float64)
    x_cf  = np.asarray(x_cf,  dtype=np.float64)
    y_hat = np.asarray(y_hat, dtype=np.float64)
    y_cf  = np.asarray(y_cf,  dtype=np.float64)

    if y_hat.ndim == 3:
        y_hat = y_hat[..., 0]
    if y_cf.ndim == 3:
        y_cf = y_cf[..., 0]

    B = x.shape[0]
    x_flat    = x.reshape(B, -1)
    x_cf_flat = x_cf.reshape(B, -1)

    delta_forecast = np.abs(y_hat.mean(axis=1) - y_cf.mean(axis=1))
    effort         = np.linalg.norm(x_cf_flat - x_flat, axis=1)

    aer = delta_forecast / (effort + eps)
    return aer.astype(np.float32)


# ---------------------------------------------------------------------------
# 3. Reachability Score
# ---------------------------------------------------------------------------

def reachability_score(x, x_cf, x_train, n_neighbors=10, n_steps=50,
                       random_state=None):
    """
    Estime si x_cf est atteignable depuis x par des transitions
    qui ressemblent à celles observées dans x_train.

    Parameters
    ----------
    x           : np.ndarray, shape [B, T] or [B, T, C]
    x_cf        : np.ndarray, shape [B, T] or [B, T, C]
    x_train     : np.ndarray, shape [N, T] or [N, T, C]
    n_neighbors : int, nombre de voisins pour le graphe de transitions
    n_steps     : int, nombre de trajectoires simulées
    random_state : int or None

    Returns
    -------
    np.ndarray, shape [B] — score d'atteignabilité ∈ (0, 1]
    """
    rng = np.random.default_rng(random_state)

    x       = np.asarray(x,       dtype=np.float64)
    x_cf    = np.asarray(x_cf,    dtype=np.float64)
    x_train = np.asarray(x_train, dtype=np.float64)

    B = x.shape[0]
    x_flat      = x.reshape(B, -1)
    x_cf_flat   = x_cf.reshape(B, -1)
    x_train_flat = x_train.reshape(x_train.shape[0], -1)

    # Transitions observées dans x_train
    transitions = np.diff(x_train_flat, axis=0)  # (N-1, D)

    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(x_train_flat)

    scores = []
    for i in range(B):
        xi    = x_flat[i]
        xi_cf = x_cf_flat[i]

        # Simuler n_steps trajectoires depuis xi
        reached = []
        for _ in range(n_steps):
            _, idx = nbrs.kneighbors(xi.reshape(1, -1))
            # Choisir une transition réaliste aléatoire dans le voisinage
            t_idx = rng.choice(idx[0])
            if t_idx < len(transitions):
                transition = transitions[t_idx]
            else:
                transition = transitions[-1]
            reached.append(xi + transition)

        reached = np.array(reached)  # (n_steps, D)
        dists   = np.linalg.norm(reached - xi_cf, axis=1)
        score   = 1.0 / (1.0 + np.min(dists))
        scores.append(score)

    return np.asarray(scores, dtype=np.float32)


# ---------------------------------------------------------------------------
# 4. Causal Compactness
# ---------------------------------------------------------------------------

def causal_compactness(x, x_cf, threshold_factor=0.1):
    """
    Une CF actionnable modifie des segments contigus (intervention localisée),
    pas des points éparpillés aléatoirement.

    Parameters
    ----------
    x                : np.ndarray, shape [B, T] or [B, T, C]
    x_cf             : np.ndarray, shape [B, T] or [B, T, C]
    threshold_factor : float, fraction de la magnitude moyenne pour définir
                       un changement significatif (défaut : 0.1)

    Returns
    -------
    np.ndarray, shape [B] — compacité ∈ (0, 1] (1 = 1 seul segment contigu)
    """
    x    = np.asarray(x,    dtype=np.float64)
    x_cf = np.asarray(x_cf, dtype=np.float64)

    B = x.shape[0]
    x_flat    = x.reshape(B, -1)
    x_cf_flat = x_cf.reshape(B, -1)

    scores = []
    for i in range(B):
        delta     = np.abs(x_cf_flat[i] - x_flat[i])
        threshold = np.mean(delta) * threshold_factor
        changed   = delta > threshold

        segments   = 0
        in_segment = False
        for c in changed:
            if c and not in_segment:
                segments  += 1
                in_segment = True
            elif not c:
                in_segment = False

        total_changed = np.sum(changed)
        if total_changed == 0:
            scores.append(1.0)
        else:
            compactness = 1.0 / segments if segments > 0 else 1.0
            scores.append(compactness)

    return np.asarray(scores, dtype=np.float32)


# ---------------------------------------------------------------------------
# 5. Forecast Monotonicity
# ---------------------------------------------------------------------------

def forecast_monotonicity(x, x_cf, forecaster, threshold=5e-2):
    """
    Pour chaque timestep modifié, vérifie que sa contribution marginale
    va dans la bonne direction (réduction du forecast).

    Parameters
    ----------
    x          : np.ndarray, shape [B, T] or [B, T, C]
    x_cf       : np.ndarray, shape [B, T] or [B, T, C]
    forecaster : callable, prend un tableau de même shape que x et renvoie
                 les forecasts [B, H] ou [B, H, 1]
    threshold  : float, seuil de détection des timesteps modifiés

    Returns
    -------
    np.ndarray, shape [B] — proportion de modifications dans le bon sens
    """
    x    = np.asarray(x,    dtype=np.float64)
    x_cf = np.asarray(x_cf, dtype=np.float64)

    B = x.shape[0]
    scores = []

    for i in range(B):
        xi    = x[i:i+1]       # (1, T, ...)
        xi_cf = x_cf[i:i+1]

        diff    = np.abs(xi_cf - xi)
        # Indice temporel des timesteps modifiés
        if diff.ndim == 3:
            mask = np.where(np.max(diff[0], axis=-1) > threshold)[0]
        else:
            mask = np.where(diff[0] > threshold)[0]

        if len(mask) == 0:
            scores.append(1.0)
            continue

        baseline_forecast = np.asarray(forecaster(xi))
        if baseline_forecast.ndim == 3:
            baseline_forecast = baseline_forecast[..., 0]
        baseline_mean = baseline_forecast.mean()

        contributions = []
        for t in mask:
            x_ablated = xi_cf.copy()
            if x_ablated.ndim == 3:
                x_ablated[0, t, :] = xi[0, t, :]
            else:
                x_ablated[0, t] = xi[0, t]

            ablated_forecast = np.asarray(forecaster(x_ablated))
            if ablated_forecast.ndim == 3:
                ablated_forecast = ablated_forecast[..., 0]

            # Contribution positive = ce timestep réduit le forecast
            contribution = baseline_mean - ablated_forecast.mean()
            contributions.append(contribution)

        monotonicity = np.mean(np.array(contributions) > 0)
        scores.append(float(monotonicity))

    return np.asarray(scores, dtype=np.float32)


__all__ = [
    "rate_of_change_feasibility",
    "action_efficiency_ratio",
    "reachability_score",
    "causal_compactness",
    "forecast_monotonicity",
]
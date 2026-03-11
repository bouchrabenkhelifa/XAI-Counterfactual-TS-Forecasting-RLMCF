import torch
import torch.nn as nn
import numpy as np


# ══════════════════════════════════════════════════════════════
# 1) State builder
# ══════════════════════════════════════════════════════════════

def build_state(z: torch.Tensor,
                y_hat: torch.Tensor,
                direction: float = -1.0) -> torch.Tensor:
   
    if y_hat.dim() == 3:
        ot = y_hat[:, :, -1]    # (B, pred_len)
    else:
        ot = y_hat              # (B, pred_len)

    # forecast summary : 4 stats
    f_mean = ot.mean(dim=1, keepdim=True)        # (B, 1)
    f_std  = ot.std(dim=1,  keepdim=True) + 1e-8 # (B, 1)
    f_min  = ot.min(dim=1).values.unsqueeze(1)   # (B, 1)
    f_max  = ot.max(dim=1).values.unsqueeze(1)   # (B, 1)

    summary = torch.cat([f_mean, f_std, f_min, f_max], dim=1)  # (B, 4)

    # direction token
    B   = z.shape[0]
    dir_token = torch.full((B, 1), direction,
                           dtype=z.dtype, device=z.device)     # (B, 1)

    return torch.cat([z, summary, dir_token], dim=1)            # (B, 69)


# ══════════════════════════════════════════════════════════════
# 2) Actor
# ══════════════════════════════════════════════════════════════

class Actor(nn.Module):
    """
    Actor network : s → (mu, log_std) → a ~ N(mu, std)

    Uses stochastic policy (Gaussian) for exploration.
    Action bounded in [-1, 1] via Tanh.

    Architecture :
        Linear(69→256) → LayerNorm → GELU
        Linear(256→256) → LayerNorm → GELU
        Linear(256→128) → LayerNorm → GELU
        Linear(128→64)  → (mu head + log_std head)

    Args:
        state_dim  : input state dimension (default 69)
        action_dim : latent space dimension (default 64)
        log_std_min: minimum log std for numerical stability
        log_std_max: maximum log std
    """

    def __init__(self, state_dim:   int   = 69,
                       action_dim:  int   = 64,
                       log_std_min: float = -4.0,
                       log_std_max: float = 0.5):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        self.mu_head      = nn.Linear(128, action_dim)
        self.log_std_head = nn.Linear(128, action_dim)

    def forward(self, s: torch.Tensor):
        """
        Args:
            s : (B, state_dim)

        Returns:
            mu      : (B, action_dim)  mean action
            log_std : (B, action_dim)  log standard deviation
        """
        h       = self.net(s)
        mu      = torch.tanh(self.mu_head(h))
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std,
                              self.log_std_min,
                              self.log_std_max)
        return mu, log_std

    def sample(self, s: torch.Tensor):
        """
        Sample action with reparameterization trick.

        Args:
            s : (B, state_dim)

        Returns:
            a        : (B, action_dim)  sampled action ∈ [-1,1]
            log_prob : (B,)             log probability of action
            mu       : (B, action_dim)  deterministic action
        """
        mu, log_std = self.forward(s)
        std         = log_std.exp()
        dist        = torch.distributions.Normal(mu, std)

        # reparameterization : a = tanh(mu + std * eps)
        x_t      = dist.rsample()
        a        = torch.tanh(x_t)

        # log prob with tanh correction
        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(1 - a.pow(2) + 1e-6)
        log_prob  = log_prob.sum(dim=-1)    # (B,)

        return a, log_prob, mu

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad)


# ══════════════════════════════════════════════════════════════
# 3) Critic
# ══════════════════════════════════════════════════════════════

class Critic(nn.Module):
    """
    Critic network : s → V(s) ∈ R

    Estimates the expected cumulative reward from state s.
    Used to compute advantage = R - V(s).

    Architecture :
        Linear(69→256) → LayerNorm → GELU
        Linear(256→128) → LayerNorm → GELU
        Linear(128→1)

    Args:
        state_dim : input state dimension (default 69)
    """

    def __init__(self, state_dim: int = 69):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s : (B, state_dim)

        Returns:
            V : (B, 1)  state value
        """
        return self.net(s)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad)


# ══════════════════════════════════════════════════════════════
# 4) Full Actor-Critic agent
# ══════════════════════════════════════════════════════════════

class ActorCritic(nn.Module):
    """
    Full Actor-Critic agent for CF-RL.

    Combines Actor + Critic with shared state builder.

    Args:
        latent_dim  : AE latent dimension (default 64)
        pred_len    : forecaster prediction length (default 96)
        eta         : perturbation scale in latent space (default 0.05)
        entropy_coef: entropy regularization coefficient
        direction   : -1.0 = want forecast down, +1.0 = up
    """

    def __init__(self,
                 latent_dim:   int   = 64,
                 pred_len:     int   = 96,
                 eta:          float = 0.05,
                 entropy_coef: float = 0.01,
                 direction:    float = -1.0):
        super().__init__()

        # state dim = latent_dim + 4 (summary) + 1 (direction)
        state_dim = latent_dim + 4 + 1

        self.actor       = Actor(state_dim=state_dim,
                                 action_dim=latent_dim)
        self.critic      = Critic(state_dim=state_dim)

        self.latent_dim  = latent_dim
        self.eta         = eta
        self.entropy_coef= entropy_coef
        self.direction   = direction

    def build_state(self, z: torch.Tensor,
                    y_hat: torch.Tensor) -> torch.Tensor:
        return build_state(z, y_hat, self.direction)

    def act(self, z: torch.Tensor,
            y_hat: torch.Tensor):
        """
        Given z and y_hat, sample action and compute z_cf.

        Args:
            z     : (B, latent_dim)
            y_hat : (B, pred_len, C)

        Returns:
            z_cf     : (B, latent_dim)  perturbed latent
            a        : (B, latent_dim)  sampled action
            log_prob : (B,)             log prob of action
            s        : (B, state_dim)   state vector
        """
        s              = self.build_state(z, y_hat)
        a, log_prob, _ = self.actor.sample(s)
        z_cf           = torch.clamp(z + self.eta * a, -1.0, 1.0)
        return z_cf, a, log_prob, s

    def evaluate(self, s: torch.Tensor) -> torch.Tensor:
        """
        Estimate state value V(s).

        Args:
            s : (B, state_dim)

        Returns:
            V : (B, 1)
        """
        return self.critic(s)

    def compute_loss(self,
                     log_prob: torch.Tensor,
                     reward:   torch.Tensor,
                     value:    torch.Tensor,
                     entropy:  torch.Tensor = None) -> dict:
        """
        Compute actor + critic losses.

        advantage = R - V(s).detach()
        loss_actor  = -advantage * log_prob - entropy_coef * H
        loss_critic = MSE(V(s), R)
        loss_total  = loss_actor + 0.5 * loss_critic

        Args:
            log_prob : (B,)   log probability of taken action
            reward   : (B,)   total reward
            value    : (B,1)  critic estimate V(s)
            entropy  : (B,)   optional entropy bonus

        Returns:
            dict with losses
        """
        V         = value.squeeze(1)              # (B,)
        advantage = (reward - V.detach())         # (B,)

        # normalize advantage for stability
        if advantage.shape[0] > 1:
            advantage = ((advantage - advantage.mean())
                        / (advantage.std() + 1e-8))

        loss_actor  = -(advantage * log_prob).mean()
        loss_critic = nn.functional.mse_loss(V, reward.detach())

        # entropy bonus : encourage exploration
        if entropy is not None:
            loss_actor = loss_actor - self.entropy_coef * entropy.mean()

        loss_total = loss_actor + 0.5 * loss_critic

        return {
            "total"   : loss_total,
            "actor"   : loss_actor,
            "critic"  : loss_critic,
            "advantage": float(advantage.mean().item()),
        }

    def count_parameters(self):
        return {
            "actor" : self.actor.count_parameters(),
            "critic": self.critic.count_parameters(),
            "total" : (self.actor.count_parameters()
                     + self.critic.count_parameters()),
        }
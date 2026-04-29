# Random Latent Perturbation (RLP) Baseline

## Principe

Même pipeline que le framework RL, mais l'action est un **bruit gaussien aléatoire** au lieu de la politique apprise.

```
x → AE.encode() → z → z + eta * N(0,1) → AE.decode() → masque temporel → forecaster → y_cf
```

## Usage

Depuis la racine du projet :

```bash
python baselines/RLP/run_rlp.py \
    --forecast_config assets/configs/models/etth1_dataset/forecasters/itransformer/etth1_96_48_S.json \
    --ae_config       assets/configs/models/etth1_dataset/ae/tcn_ae.json \
    --rl_config       assets/configs/models/etth1_dataset/RL_ablations/config_final.json \
    --eval_batches    20 \
    --n_trials        10 \
    --seed            42 \
    --output          baselines/RLP/results/rlp_etth1.json
```

## Paramètres

| Paramètre | Description |
|---|---|
| `--rl_config` | Config RL dont on réutilise `eta`, `mask_last_k`, `mask_ramp_k`, `rho`, `fr` |
| `--eval_batches` | Nombre de batches du test set évalués par trial |
| `--n_trials` | Nombre de runs aléatoires (résultats moyennés) |
| `--seed` | Graine de base (chaque trial utilise `seed + trial`) |

## Résultats

Sauvegardés dans `baselines/RLP/results/` au format JSON :
- `avg_metrics` : moyenne sur les n_trials
- `std_metrics` : écart-type sur les n_trials
- `all_trials` : détail de chaque trial

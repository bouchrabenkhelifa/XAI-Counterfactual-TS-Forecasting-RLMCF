# 📑 Index des fichiers - Adaptation ForecastCF

## 🎯 Par objectif

### Je veux démarrer rapidement
→ `QUICKSTART.md` - Guide de démarrage en 3 étapes

### Je veux comprendre ce qui a été fait
→ `SUMMARY.md` - Résumé complet de l'adaptation

### Je veux voir la documentation complète
→ `README.md` - Vue d'ensemble détaillée

### Je veux vérifier ma configuration
→ `check_setup.py` - Script de vérification

### Je veux lancer la comparaison
→ `run_comparison.sh` - Pipeline automatique complet

### Je veux comparer les résultats
→ `compare_results.py` - Script de comparaison avec graphiques

### Je veux installer les dépendances
→ `requirements_baseline.txt` - Liste des packages Python

## 📂 Par type de fichier

### Documentation (📖)
- `README.md` - Documentation principale
- `QUICKSTART.md` - Guide rapide
- `SUMMARY.md` - Résumé de l'adaptation
- `INDEX.md` - Ce fichier
- `ForecastCF/README_PYTORCH.md` - Détails techniques

### Scripts Python (🐍)
- `compare_results.py` - Comparaison RL vs ForecastCF
- `check_setup.py` - Vérification de configuration
- `ForecastCF/src/pytorch_adapter.py` - Wrapper PyTorch
- `ForecastCF/src/cf_search_pytorch.py` - Exécution avec PyTorch

### Scripts Bash (⚡)
- `run_comparison.sh` - Pipeline complet
- `ForecastCF/run_etth1_itransformer.sh` - ForecastCF + iTransformer

### Configuration (⚙️)
- `requirements_baseline.txt` - Dépendances Python

## 🔄 Workflow typique

```
1. Vérifier la configuration
   └─> python baselines/check_setup.py

2. Installer les dépendances manquantes
   └─> pip install -r baselines/requirements_baseline.txt

3. Lancer ForecastCF baseline
   └─> cd baselines/ForecastCF
   └─> bash run_etth1_itransformer.sh

4. Comparer avec vos résultats RL
   └─> python baselines/compare_results.py \
         --rl-results <vos_resultats> \
         --fcf-results baselines/ForecastCF/results/forecastcf_etth1_itransformer.csv

5. Analyser les graphiques
   └─> Voir baselines/comparison_plots/
```

## 📊 Fichiers générés après exécution

### Résultats ForecastCF
- `ForecastCF/results/forecastcf_etth1_itransformer.csv`

### Graphiques de comparaison
- `comparison_plots/comparison_barplot.png`
- `comparison_plots/comparison_boxplot.png`
- `comparison_plots/comparison_radar.png`

## 🎓 Par niveau d'expertise

### Débutant
1. `QUICKSTART.md` - Commencez ici
2. `check_setup.py` - Vérifiez votre setup
3. `run_comparison.sh` - Lancez tout automatiquement

### Intermédiaire
1. `README.md` - Comprenez l'architecture
2. `ForecastCF/run_etth1_itransformer.sh` - Personnalisez les paramètres
3. `compare_results.py` - Analysez les résultats

### Avancé
1. `SUMMARY.md` - Détails de l'implémentation
2. `ForecastCF/README_PYTORCH.md` - Architecture technique
3. `ForecastCF/src/pytorch_adapter.py` - Code du wrapper
4. `ForecastCF/src/cf_search_pytorch.py` - Code d'exécution

## 🔍 Recherche rapide

### "Comment installer ?"
→ `requirements_baseline.txt` + `pip install -r`

### "Comment vérifier que tout est prêt ?"
→ `python baselines/check_setup.py`

### "Comment lancer rapidement ?"
→ `bash baselines/run_comparison.sh`

### "Comment personnaliser les paramètres ?"
→ Éditez `ForecastCF/run_etth1_itransformer.sh`

### "Comment comparer les résultats ?"
→ `python baselines/compare_results.py --help`

### "Où sont les résultats ?"
→ `ForecastCF/results/` et `comparison_plots/`

### "Comment tester un autre modèle ?"
→ Voir section "Autres modèles" dans `README.md`

### "Ça ne marche pas, que faire ?"
→ Section "Troubleshooting" dans `QUICKSTART.md`

## 📞 Aide

### Problème de configuration
→ `check_setup.py` vous dira exactement ce qui manque

### Problème d'exécution
→ Voir "Troubleshooting" dans `QUICKSTART.md`

### Question sur l'adaptation
→ Voir `SUMMARY.md` pour les détails techniques

### Question sur ForecastCF original
→ Voir `ForecastCF/README.md` (documentation originale)

## 🗺️ Carte mentale

```
baselines/
│
├─ 🚀 DÉMARRAGE RAPIDE
│  ├─ QUICKSTART.md
│  ├─ check_setup.py
│  └─ run_comparison.sh
│
├─ 📖 DOCUMENTATION
│  ├─ README.md (principal)
│  ├─ SUMMARY.md (résumé)
│  ├─ INDEX.md (ce fichier)
│  └─ ForecastCF/README_PYTORCH.md
│
├─ 🔧 OUTILS
│  ├─ compare_results.py
│  ├─ check_setup.py
│  └─ requirements_baseline.txt
│
└─ 💻 CODE
   └─ ForecastCF/
      ├─ src/
      │  ├─ pytorch_adapter.py (wrapper)
      │  └─ cf_search_pytorch.py (exécution)
      └─ run_etth1_itransformer.sh
```

## ✅ Checklist

Avant de commencer:
- [ ] Lire `QUICKSTART.md`
- [ ] Exécuter `check_setup.py`
- [ ] Installer dépendances manquantes
- [ ] Vérifier que le checkpoint iTransformer existe

Pour lancer:
- [ ] Exécuter `run_comparison.sh` OU
- [ ] Exécuter manuellement les étapes du QUICKSTART

Après exécution:
- [ ] Vérifier les résultats CSV
- [ ] Analyser les graphiques
- [ ] Comparer avec vos résultats RL

## 🎯 Fichiers essentiels (top 5)

1. **QUICKSTART.md** - Pour démarrer
2. **check_setup.py** - Pour vérifier
3. **run_comparison.sh** - Pour lancer
4. **compare_results.py** - Pour comparer
5. **README.md** - Pour comprendre

---

**Astuce**: Commencez toujours par `QUICKSTART.md` !

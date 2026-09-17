# Index of every reported number, and the script that produces it

Kept so that any figure in the manuscript can be traced to the code that generated it without
re-reading the whole repository.

| Manuscript location | Value | Produced by |
|---|---|---|
| Table 1, LightGBM | 0.6888 [0.6837, 0.6936] | `02_nested_benchmark.py` |
| Table 1, XGBoost | 0.6827 [0.6773, 0.6876] | `02_nested_benchmark.py` |
| Table 1, logistic regression | 0.6758 [0.6706, 0.6808] | `02_nested_benchmark.py` |
| §3.1 paired difference | +0.0062 [+0.0041, +0.0082] | `02_nested_benchmark.py` |
| Table 2, all rows | 0.6895 down to 0.6768 | `04_preprocessing_ablation.py` |
| §3.2 terminal-encounter effect | 0.6824 against 0.6895 | `04_preprocessing_ablation.py` |
| Table 3, splitting comparison | +0.0014, p = 0.45 | `05_trajectory_features.py` |
| Table 4, trajectory contribution | +0.0112 | `05_trajectory_features.py` |
| §3.4 repeat-encounter rates | 16.74% against 8.80% | `05_trajectory_features.py` |
| Table 5, balancing | 0.6895 to 0.6756 | `03_ablations.py` |
| Table 6, deep models | 0.6662, 0.6463, 0.6282 | `06_deep_models.py` |
| §3.7 calibration | 0.0953 against 0.0957 | `07_subgroup_and_utility.py` |
| Table 7, subgroups | 0.7329 down to 0.6312 | `07_subgroup_and_utility.py` |
| Table 8, capacity targeting | 3.00x down to 1.76x | `07_subgroup_and_utility.py` |
| §Synthetic ceiling, Bayes AUC 0.5814 | derived in code, not quoted | `01_synthetic_signal_ceiling.py` |
| §Synthetic ceiling, logistic 0.5809, p = 0.72 vs oracle | 15-fold repeated CV | `01_synthetic_signal_ceiling.py` |
| Figures 1 through 7 | — | `08_figures.py` |

Console output from `01_synthetic_signal_ceiling.py` is committed under `results/tables/`, so the ceiling argument can be checked without running anything.

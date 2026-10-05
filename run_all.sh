#!/usr/bin/env bash
# Reproduces every number in the manuscript, in order. Expect roughly 40 minutes on a
# laptop CPU — experiment 06 accounts for most of that, and can be skipped safely.
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

echo "==> 01  signal ceiling of the synthetic dataset"
python src/experiments/01_synthetic_signal_ceiling.py | tee results/tables/01_synthetic_ceiling.txt

echo "==> 02  nested cross-validated benchmark"
python src/experiments/02_nested_benchmark.py        | tee results/tables/02_nested_benchmark.txt

echo "==> 03  feature-family and class-balancing ablations"
python src/experiments/03_ablations.py               | tee results/tables/03_ablations.txt

echo "==> 04  preprocessing ablation against the published pipeline"
python src/experiments/04_preprocessing_ablation.py  | tee results/tables/04_preprocessing.txt

echo "==> 05  patient-trajectory features"
python src/experiments/05_trajectory_features.py     | tee results/tables/05_trajectory.txt

echo "==> 06  deep-learning comparison (slow; optional)"
python src/experiments/06_deep_models.py             | tee results/tables/06_deep_models.txt

echo "==> 07  subgroup and decision-curve analysis"
python src/experiments/07_subgroup_and_utility.py    | tee results/tables/07_subgroup.txt

echo "==> 09  dashboard aggregates (docs/index.html)"
python src/experiments/09_dashboard_export.py

echo "==> 08  figures"
python src/experiments/08_figures.py

echo "done — tables in results/tables, figures in results/figures"

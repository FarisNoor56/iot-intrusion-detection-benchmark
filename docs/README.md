# CICIoT2023 Benchmark Comparison

This notebook represents the project workflow for training and evaluating a machine learning model on the CICIoT2023 IoT intrusion dataset.

The repository contains the reproducible code in `src/train_and_evaluate.py` and the benchmark summary in `results/benchmark_summary.md`.

## Workflow

1. Download the dataset from the official CICIoT2023 website.
2. Place the CSV file in `data/`.
3. Run:

```bash
python src/train_and_evaluate.py --data_path data/<your_csv_file>.csv
```

4. Review output metrics and comparison results under `results/`.

## Reported benchmark

The paper reports that a Random Forest / LightGBM style model achieves approximately 0.99 F1-score and >99% accuracy on the CICIoT2023 benchmark.

## Current implementation

The repository uses a robust Random Forest pipeline with:
- median imputation for numerical data,
- most-frequent imputation for categorical data,
- OneHot encoding for categorical features,
- train/test split with stratification,
- weighted precision/recall/F1 metrics.

## Repository files

- `src/train_and_evaluate.py` — training and evaluation logic
- `results/benchmark_summary.md` — summary of benchmark vs implementation
- `results/benchmark_metrics.json` — exact metric values from the implementation
- `results/benchmark_comparison.json` — comparison object aligned with the paper

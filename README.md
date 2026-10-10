# IoT Intrusion Detection Benchmark

This repository reproduces a practical IoT intrusion detection workflow using the CICIoT2023 dataset and compares the implementation against the published benchmark reported in the CICIoT2023 paper.

## Objective

The goal is to:
- load and preprocess the selected public IoT intrusion detection dataset,
- implement a standard machine learning pipeline for classification,
- evaluate the model on the same dataset,
- compare the results against the benchmark metrics reported in the original CICIoT2023 publication.

## Selected dataset

- Dataset: CICIoT2023
- Source: Canadian Institute for Cybersecurity (UNB)
- Reference: https://www.unb.ca/cic/datasets/iotdataset-2023.html
- Paper: "CICIoT2023: A Real-time Dataset and Benchmark for Large-Scale Attacks in IoT Environment"

## Benchmark used for comparison

The original paper reports that machine learning models, especially Random Forest and LightGBM, perform very strongly on this dataset, with:
- Accuracy: often > 99%
- F1-score: around 0.99 for top models
- Strong performance across multiple attack families

This project attempts to reproduce that trend using a standard Random Forest classifier on the same data.

## Repository structure

- `src/` — source code for preprocessing, training, and evaluation
- `results/` — output metrics and comparison summaries
- `data/` — local dataset storage (not committed; place CSV files here)
- `README.md` — project overview
- `requirements.txt` — Python dependencies

## Quick start

1. Download the CICIoT2023 CSV data into `data/`.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the benchmark script:

```bash
python src/train_and_evaluate.py --data_path data/your_dataset.csv
```

4. Inspect generated metrics in `results/benchmark_summary.md` and JSON outputs.

## Expected model behavior

The training pipeline includes:
- CSV parsing
- missing-value handling
- categorical encoding
- feature filtering for non-informative columns
- train/test split with stratification
- Random Forest classification
- metrics: accuracy, precision, recall, F1-score, macro-F1, class report

## Notes

This dataset is large and may require a machine with sufficient RAM and CPU resources. If the CSV is too large for a single in-memory load, the code can be adapted to incremental training or chunked processing.

## References

- CICIoT2023 dataset: https://www.unb.ca/cic/datasets/iotdataset-2023.html
- CICIoT2023 paper: https://www.mdpi.com/1424-8220/23/13/5941
- IoT-Related-Dataset-and-Resources: https://github.com/mnazhoq/IoT-Related-Dataset-and-Resources

## License

MIT

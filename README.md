# Self-Pruning Neural Network for CIFAR-10

This repository implements a sparsity-aware neural network for CIFAR-10 using learnable gates in prunable linear layers. The project is designed for engineering readability, reproducibility, and submission quality.

## 1. Problem Statement

The model learns both:
- classification parameters for CIFAR-10
- pruning decisions through gate parameters

Each weight is multiplied by a learnable gate:

$$
\tilde{W}_{ij} = W_{ij} \cdot \sigma(s_{ij})
$$

with an L1 sparsity term on gates added to the loss:

$$
\mathcal{L} = \mathcal{L}_{CE} + \lambda \sum_{i,j} |\sigma(s_{ij})|
$$

This encourages many gates to move toward zero while preserving predictive performance.

## 2. Visualizing Sparsity

Below is the gate distribution for the best performing model ($\lambda = 1e-3$). The spike at zero indicates successful pruning of redundant connections.

![Gate Distribution](results/gate_distributions.png)

## 3. Installation Instructions

### 2.1 Prerequisites

- Python 3.13+
- pip

### 2.2 Setup

```bash
cd tredence-analytics
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Alternative editable install:

```bash
pip install -e .
```

## 4. Usage Guide

### 3.1 Run Training

End-to-end enhanced training + hyperparameter sweep:

```bash
python optimization/run_enhanced_experiments.py
```

This runs:
- hyperparameter tuning experiments
- visualization generation
- summary reporting of best configuration

### 3.2 Run Evaluation Script

Baseline evaluation table (three lambda settings) and gate histogram plot:

```bash
python core/evaluate.py
```

This produces:
- console table: lambda, test accuracy, sparsity level (%), active weights
- plot: results/gate_distributions.png

### 3.3 Expected Artifacts

- results/model_lambda_*.pt
- results/history_lambda_*.pt
- results/gate_distributions.png
- results/hyperparams/results_summary.json
- results/hyperparams/curves_*.png
- results/hyperparams/comparison_matrix.png

## 5. Repository Structure

```text
tredence-analytics/
├── core/
│   ├── prunable_linear.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── optimization/
│   ├── hyperparameter_tuning.py
│   ├── visualize_results.py
│   └── run_enhanced_experiments.py
├── data/
│   └── cifar-10-batches-py/
├── results/
│   ├── model_lambda_*.pt
│   ├── history_lambda_*.pt
│   ├── gate_distributions.png
│   └── hyperparams/
├── requirements.txt
├── pyproject.toml
├── .gitignore
└── report.md
```

## 6. Code Quality and Builder Mindset

Engineering choices made for maintainability and extensibility:

- modular layer design via PrunableLinear
- callback-based training flow with early stopping support
- configurable regularization (dropout, batch norm, lambda)
- experiment orchestration separated from model code
- explicit artifact saving for reproducibility and review

## 7. API Documentation (Optional Bonus)

This repository does not include a FastAPI inference server by default, but if you add one,
document and run it with the following pattern:

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Example inference trigger:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"image": [0.0, 0.1, ...]}'
```

Recommended API contract:
- endpoint: POST /predict
- input: flattened, normalized CIFAR-10 image tensor
- output: predicted class index and confidence scores

## 8. Report

See report.md for:
- mathematical explanation of gate sparsity
- results table for three lambda experiments
- gate-distribution visualization discussion
- implementation quality notes and next steps

## 9. License

Provided as-is for educational and research use.

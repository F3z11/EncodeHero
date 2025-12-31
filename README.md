# EncodeHero

A benchmarking tool for comparing categorical encoding methods in machine learning using the Analytic Hierarchy Process (AHP).

This tool is part for the paper *"Categorical Variable Encoding Methods for Tabular Data: A Benchmarking Study"* by F. Clerici and N. Nobani.

## Overview

EncodeHero helps you identify the best categorical encoding technique for your dataset by evaluating multiple encoders across various performance and cost metrics. It uses AHP to rank encoders based on your priorities, making it easier to choose the right encoding strategy for your ML pipeline.

## Features

- **Comprehensive Encoder Testing**: Benchmarks 20+ categorical encoders including One-Hot, Target, Binary, CatBoost, and more
- **Multi-Criteria Evaluation**: Analyzes encoders based on performance metrics (Accuracy, Precision, Recall, F1, AUC, MCC) and computational cost (Time)
- **Interactive GUI**: User-friendly Streamlit interface for setting priorities and visualizing results
- **Support for Multiple Task Types**: Binary classification, multiclass classification, and regression
- **AHP-Based Ranking**: Systematic decision-making framework for encoder selection

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Using the Interactive GUI

Launch the Streamlit application:
```bash
streamlit run EncodeHero/gui.py
```

In the GUI, you can:
1. Select your task type (binary, multiclass, or regression)
2. Set priorities for different criteria using pairwise comparisons
3. View the ranked encoders based on your preferences

## Project Structure

```
EncodeHero/
├── EncodeHero/
│   ├── ahp_computation.py   # AHP ranking logic
│   ├── experiment.py         # ML experiment runner
│   ├── gui.py               # Streamlit interface
│   ├── res_analysis.py      # Results analysis
│   └── results/             # Benchmark results
└── requirements.txt
```

### Running Experiments Programmatically

To benchmark encoders on your own dataset:

```python
import pandas as pd
from EncodeHero import MLExperiment

# Load your dataset
df = pd.read_csv("your_dataset.csv")

# Create and run experiment
experiment = MLExperiment(
    data=df,
    target='target_column_name',  # Name of your target column
    task='binary',                # 'binary', 'multiclass', or 'regression'
    seed=42,                      # Random seed for reproducibility
    dataset_name='my_dataset'     # Name for results folder
)

# Run the experiment (results saved in my_dataset_42/ folder)
results = experiment.run_experiment(
    export=True,      # Save results to CSV files
    dir='.',          # Directory where results folder will be created
    use_cache=True    # Cache encoder results for faster re-runs
)

# Access results
print(f"Top encoders by median performance:")
print(results['all_median'].head())
print(f"\nExecution times:")
print(results['time_dim'])
```

**Output Structure:**
- Results are saved in a folder named `{dataset_name}_{seed}/`
- CSV files include:
  - `results_{seed}.csv` - Detailed metrics for all encoder-model combinations
  - `time_dim_{seed}.csv` - Execution time for each encoder
  - `model_median_{seed}.csv` - Median performance across models per encoder
  - `all_median_{seed}.csv` - Overall median performance per encoder
- Cache files are stored in `{dataset_name}_{seed}/cache/` for faster re-runs

**Requirements for your dataset:**
- Must be a pandas DataFrame
- Target column should contain the values to predict
- Categorical columns will be automatically detected
- For classification tasks, target will be automatically encoded to numeric labels

## How It Works

1. **Benchmarking**: Tests each encoder with multiple ML models on your dataset
2. **Metric Collection**: Gathers performance metrics (accuracy, F1, etc.) and timing information
3. **AHP Ranking**: Applies pairwise comparison preferences to calculate encoder priorities
4. **Recommendation**: Presents ranked encoders to help you make an informed choice

## License

See [LICENSE](LICENSE) file for details.

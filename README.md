🚀 Features

This tool is designed to perform pre-training dataset diagnostics, helping practitioners detect critical data issues before investing time in model development.

🖼️ Image Dataset Analysis

For image-based classification datasets, the tool performs an automated inspection of class balance:

Class Distribution Analysis

Counts the number of samples per class based on directory structure

Provides a clear overview of majority and minority classes

Class Imbalance Detection

Automatically flags underrepresented classes using a configurable threshold

Helps identify scenarios where accuracy may be misleading

Visual Diagnostics

Generates and saves a bar chart illustrating class distribution

Enables quick visual inspection of imbalance severity

📌 Why this matters:
Severely imbalanced image datasets often require techniques such as class weighting, resampling, or specialized evaluation metrics. Detecting this early prevents biased model training.

📊 Tabular Dataset Analysis

For tabular (CSV-based) datasets, the tool performs a deeper, feature-aware analysis:

Target Variable Imbalance

Computes normalized class distribution of the target column

Detects minority classes and generates explicit imbalance warnings

Feature-Level Diagnostics

Numerical features

Detects highly skewed distributions using statistical skewness

Highlights features that may benefit from log transformation or robust scaling

Categorical features

Identifies high-cardinality columns that may increase overfitting risk

Helps surface potential feature engineering concerns early

Automated Recommendations

Translates technical findings into human-readable guidance

Suggests common mitigation strategies such as:

Resampling techniques (e.g., SMOTE)

Alternative evaluation metrics (F1-score, recall)

Feature transformation or encoding strategies

📌 Why this matters:
Many tabular ML failures stem from poorly understood feature distributions rather than model choice. This analysis enforces a data-first mindset before modeling.

🧠 Intelligent Dataset Recommendations

Instead of only reporting statistics, the tool provides actionable insights:

Converts detected issues into concise recommendations

Bridges the gap between raw diagnostics and modeling decisions

Acts as a lightweight “data consultant” prior to experimentation

✅ Design Principles

Lightweight & Fast – runs in seconds on small or large datasets

Non-intrusive – does not modify data, only analyzes

Modular – image and tabular analyses are independent

Fail-safe – missing datasets are handled gracefully without crashing

## 📁 Project Structure

```text
dataset-health-check/
├── data/
│   └── train/
│       ├── class_0/
│       ├── class_1/
│       └── sample_tabular.csv
├── reports/
│   └── class_distribution.png
├── src/
│   ├── class_analysis.py
│   └── tabular_analysis.py
├── run.py
├── requirements.txt

▶️ Usage
pip install -r requirements.txt
python run.py

🧠 Example Output
📊 DATASET SUMMARY (IMAGE)
Total samples: 6
Class distribution:
  class_0: 5
  class_1: 1

⚠️ Class 'class_1' is underrepresented (16.67%)

📋 TABULAR DATASET CHECK
Rows: 10, Columns: 5
Target distribution (%): {0: 90.0, 1: 10.0}

🧠 DATASET RECOMMENDATION
- Dataset shows class imbalance. Consider using class weighting or resampling techniques.

📌 Why This Matters

Many machine learning projects fail not because of the model choice, but due to poor dataset quality.

This tool encourages a dataset-first mindset, helping practitioners catch critical issues before training any model.

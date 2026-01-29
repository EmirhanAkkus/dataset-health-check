# Dataset Health Check 🧠📊

A lightweight Python tool to **analyze dataset quality before model training**.

This project helps identify common dataset issues that negatively impact machine learning performance, such as class imbalance and problematic features.

---

## 🚀 Features

### 🖼️ Image Dataset Analysis
- Class distribution analysis
- Automatic class imbalance warnings
- Class distribution visualization (saved as image)

### 📊 Tabular Dataset Analysis
- Target class imbalance detection
- Feature-level diagnostics:
  - Skewed numerical features
- Automatic, human-readable recommendations

---

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

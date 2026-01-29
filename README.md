# Dataset Health Check 🧠📊

A lightweight Python tool to **analyze dataset quality before model training**.

This project focuses on identifying common dataset issues that negatively impact machine learning performance, such as class imbalance and problematic feature distributions.  
The goal is to promote a **dataset-first mindset** before model development begins.

---

## 🚀 Features

This tool performs **pre-training dataset diagnostics** for both image-based and tabular datasets, helping practitioners detect critical data issues early.

---

### 🖼️ Image Dataset Analysis

For image-based classification datasets, the tool performs an automated inspection of class balance based on folder structure.

**Capabilities:**

- **Class Distribution Analysis**
  - Counts the number of samples per class
  - Provides a clear overview of majority and minority classes

- **Class Imbalance Detection**
  - Automatically flags underrepresented classes using a configurable threshold
  - Helps identify cases where accuracy alone may be misleading

- **Visual Diagnostics**
  - Generates and saves a bar chart illustrating class distribution
  - Enables quick visual inspection of imbalance severity

**Why this matters:**  
Imbalanced image datasets often require techniques such as class weighting, resampling, or alternative evaluation metrics. Detecting imbalance early prevents biased model training.

---

### 📊 Tabular Dataset Analysis

For tabular (CSV-based) datasets, the tool performs feature-aware analysis with actionable insights.

**Capabilities:**

- **Target Variable Imbalance**
  - Computes normalized class distribution of the target column
  - Detects minority classes and generates explicit imbalance warnings

- **Feature-Level Diagnostics**
  - **Numerical features**
    - Detects highly skewed distributions using statistical skewness
    - Highlights features that may benefit from log transformation or robust scaling
  - **Categorical features**
    - Identifies high-cardinality columns
    - Flags features that may increase overfitting risk

- **Automated Recommendations**
  - Translates technical findings into **human-readable guidance**
  - Suggests mitigation strategies such as:
    - Class weighting or resampling (e.g., SMOTE)
    - Using recall or F1-score instead of accuracy
    - Feature transformation or encoding strategies

**Why this matters:**  
Many tabular ML failures stem from poorly understood feature distributions rather than model choice. This analysis helps surface those risks early.

---

### 🧠 Intelligent Dataset Recommendations

Instead of only reporting statistics, the tool provides **actionable recommendations**:

- Converts detected issues into concise suggestions
- Bridges the gap between raw diagnostics and modeling decisions
- Acts as a lightweight “data consultant” prior to experimentation

---

### ✅ Design Principles

- **Lightweight & Fast** – runs in seconds on small or large datasets  
- **Non-intrusive** – does not modify data, only analyzes  
- **Modular** – image and tabular analyses are independent  
- **Fail-safe** – missing datasets are handled gracefully  

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
└── README.md

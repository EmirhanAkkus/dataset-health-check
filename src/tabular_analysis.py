import pandas as pd


def analyze_tabular_dataset(
    csv_path,
    target_column,
    imbalance_threshold=0.2,
    high_cardinality_threshold=50
):
    """
    Analyze a tabular dataset for target imbalance and feature-level issues.
    """
    df = pd.read_csv(csv_path)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")

    report = {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "target_distribution": {},
        "warnings": [],
        "feature_insights": []
    }

    # --- Target imbalance analysis ---
    target_ratios = df[target_column].value_counts(normalize=True)

    for cls, ratio in target_ratios.items():
        report["target_distribution"][cls] = round(ratio * 100, 2)
        if ratio < imbalance_threshold:
            report["warnings"].append(
                f"⚠️ Target class '{cls}' is underrepresented ({ratio:.2%})"
            )

    # --- Feature-level analysis ---
    for col in df.columns:
        if col == target_column:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            skewness = df[col].skew()
            if abs(skewness) > 1:
                report["feature_insights"].append(
                    f"📈 Numeric feature '{col}' is highly skewed (skew={skewness:.2f})"
                )
        else:
            unique_count = df[col].nunique()
            if unique_count > high_cardinality_threshold:
                report["feature_insights"].append(
                    f"🧩 Categorical feature '{col}' has high cardinality ({unique_count} unique values)"
                )

    return report


def generate_tabular_recommendations(report):
    """
    Generate human-readable recommendations based on tabular analysis results.
    """
    recommendations = []

    if report["warnings"]:
        recommendations.append(
            "Dataset shows class imbalance. Consider using class weighting, "
            "oversampling techniques (e.g., SMOTE), or focusing on recall/F1-score "
            "instead of accuracy."
        )

    for insight in report["feature_insights"]:
        if "skewed" in insight:
            recommendations.append(
                "Skewed numeric features detected. Applying log transformation "
                "or robust scaling may improve model performance."
            )
        if "high cardinality" in insight:
            recommendations.append(
                "High-cardinality categorical features detected. "
                "Target encoding or feature reduction is recommended to "
                "avoid overfitting."
            )

    if not recommendations:
        recommendations.append(
            "No major issues detected. Dataset appears suitable for baseline model training."
        )

    # remove duplicates while preserving order
    recommendations = list(dict.fromkeys(recommendations))

    return recommendations

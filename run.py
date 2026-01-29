from src.class_analysis import (
    analyze_class_distribution,
    plot_class_distribution
)
from src.tabular_analysis import (
    analyze_tabular_dataset,
    generate_tabular_recommendations
)


def main():
    # ===============================
    # IMAGE DATASET CHECK
    # ===============================
    train_dir = "data/train"

    report = analyze_class_distribution(train_dir)

    print("\n DATASET SUMMARY (IMAGE)")
    print(f"Total samples: {report['total_samples']}")
    print("Class distribution:")

    for cls, count in report["class_counts"].items():
        print(f"  {cls}: {count}")

    if report["warnings"]:
        print("\n WARNINGS")
        for w in report["warnings"]:
            print(w)
    else:
        print("\n No class imbalance detected")

    plot_class_distribution(report["class_counts"])
    print("\n Class distribution plot saved to reports/class_distribution.png")

    # ===============================
    # TABULAR DATASET CHECK
    # ===============================
    tabular_csv = "data/train/sample_tabular.csv"  # senin mevcut yapına göre
    target_col = "target"

    print("\n TABULAR DATASET CHECK")

    try:
        tabular_report = analyze_tabular_dataset(tabular_csv, target_col)

        print(f"Rows: {tabular_report['num_rows']}, Columns: {tabular_report['num_columns']}")
        print("Target distribution (%):", tabular_report["target_distribution"])

        if tabular_report["warnings"]:
            for w in tabular_report["warnings"]:
                print(w)

        if tabular_report["feature_insights"]:
            for insight in tabular_report["feature_insights"]:
                print(insight)

        # ===============================
        # RECOMMENDATIONS
        # ===============================
        recommendations = generate_tabular_recommendations(tabular_report)

        print("\n DATASET RECOMMENDATION")
        for rec in recommendations:
            print(f"- {rec}")

    except FileNotFoundError:
        print(f" Tabular dataset not found at '{tabular_csv}'")
    except Exception as e:
        print(f" Tabular analysis failed: {e}")


if __name__ == "__main__":
    main()

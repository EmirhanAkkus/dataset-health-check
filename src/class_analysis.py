from pathlib import Path
import matplotlib.pyplot as plt


def analyze_class_distribution(data_dir, imbalance_threshold=0.2):
    """
    Analyze class distribution in a classification dataset.

    Expected structure:
    data_dir/
        class_0/
        class_1/
        ...

    Returns:
        dict with class counts and warnings
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"{data_dir} does not exist")

    class_counts = {}
    total_samples = 0

    for class_dir in data_dir.iterdir():
        if class_dir.is_dir():
            count = len([f for f in class_dir.iterdir() if f.is_file()])
            class_counts[class_dir.name] = count
            total_samples += count

    if total_samples == 0:
        raise ValueError("Dataset is empty or incorrect folder structure.")

    warnings = []
    for cls, count in class_counts.items():
        ratio = count / total_samples
        if ratio < imbalance_threshold:
            warnings.append(
                f"⚠️ Class '{cls}' is underrepresented ({ratio:.2%})"
            )

    return {
        "total_samples": total_samples,
        "class_counts": class_counts,
        "warnings": warnings
    }


def plot_class_distribution(class_counts, output_path="reports/class_distribution.png"):
    """
    Plot and save class distribution bar chart.
    Automatically creates output directory if it does not exist.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    plt.figure(figsize=(6, 4))
    plt.bar(classes, counts)
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Samples")

    for i, v in enumerate(counts):
        plt.text(i, v, str(v), ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

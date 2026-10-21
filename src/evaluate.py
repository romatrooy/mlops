"""Сравнение экспериментов: собирает reports/metrics_*.json в одну таблицу,
сохраняет CSV/Markdown-отчёт и график сравнения точности.

Запуск: python src/evaluate.py
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load_metrics(reports_dir="reports"):
    rows = []
    for path in sorted(Path(reports_dir).glob("metrics_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "experiment": path.stem.replace("metrics_", ""),
            "model": data.get("model"),
            "dataset": "augmented" if data.get("augmented") else "original",
            "epochs": data.get("epochs"),
            "lr": data.get("lr"),
            "train_size": data.get("train_size"),
            "best_test_acc": data.get("best_test_acc"),
            "final_test_acc": data.get("final_test_acc"),
        })
    return pd.DataFrame(rows)


def to_markdown(df):
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        values = []
        for col in headers:
            value = row[col]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Сравнение результатов экспериментов")
    parser.add_argument("--reports_dir", type=str, default="reports")
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    df = load_metrics(reports_dir)
    if df.empty:
        print(f"В {reports_dir} нет файлов metrics_*.json — сначала обучите модели "
              f"(python src/models/train.py --model simple_cnn).")
        return

    df = df.sort_values(["model", "dataset"]).reset_index(drop=True)

    print("\n=== Сравнение результатов (Fashion-MNIST) ===")
    print(df.to_string(index=False))

    csv_path = reports_dir / "comparison.csv"
    df.to_csv(csv_path, index=False)
    md_path = reports_dir / "comparison.md"
    md_path.write_text(to_markdown(df), encoding="utf-8")

    # График: точность по моделям на исходных и аугментированных данных
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    pivot = df.pivot_table(index="model", columns="dataset",
                           values="final_test_acc", aggfunc="max")
    ax = pivot.plot(kind="bar", figsize=(8, 5), rot=0)
    ax.set_ylabel("Test accuracy")
    ax.set_xlabel("Модель")
    ax.set_title("Влияние аугментации на качество модели")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8)
    plt.tight_layout()
    plot_path = figures_dir / "accuracy_comparison.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"\nОтчёт сохранён: {csv_path}\n            и : {md_path}")
    print(f"График сохранён: {plot_path}")


if __name__ == "__main__":
    main()
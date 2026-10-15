from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .engine import EvaluationResult


def _style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.direction": "in", "ytick.direction": "in", "axes.linewidth": 0.8,
        "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    })


def save_history(history: list[dict[str, float]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    _style()
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.2), constrained_layout=True)
    axes[0].plot(epochs, [row["loss"] for row in history], "o-", ms=3, lw=1.2, color="#2878B5")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy loss", title="(a) Training loss")
    axes[1].plot(epochs, [100 * row["accuracy"] for row in history], "o-", ms=3, lw=1.2, color="#C82423")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy (%)", title="(b) Training accuracy")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output_dir / "training_curves.png")
    plt.close(fig)


def save_evaluation(result: EvaluationResult, output_dir: Path, name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.json"
    path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return path


def save_comparison(results: dict[str, EvaluationResult], output_dir: Path) -> None:
    payload = {name: asdict(result) for name, result in results.items()}
    (output_dir / "inference_comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _style()
    names = list(results)
    accuracy = [100 * results[name].accuracy for name in names]
    fig, ax = plt.subplots(figsize=(max(3.5, 0.8 * len(names)), 2.6), constrained_layout=True)
    bars = ax.bar(names, accuracy, color=["#2878B5", "#2F9E44"] + ["#C82423"] * max(0, len(names) - 2), width=0.65)
    ax.set_ylabel("MNIST accuracy (%)")
    ax.set_title("LeNet-5 inference accuracy comparison")
    ax.tick_params(axis="x", rotation=20)
    ax.spines[["top", "right"]].set_visible(False)
    lower = max(0.0, min(accuracy) - 2.0)
    ax.set_ylim(lower, min(100.0, max(accuracy) + 1.0))
    for bar, value in zip(bars, accuracy):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.08, f"{value:.2f}%", ha="center", va="bottom", fontsize=7)
    fig.savefig(output_dir / "accuracy_comparison.png")
    plt.close(fig)

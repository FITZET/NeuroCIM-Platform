from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


PLATFORM_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = PLATFORM_ROOT / "outputs" / "full_kernel_experiment"
SOURCE = EXPERIMENT_ROOT / "error_aware_training_accuracy_matrix.csv"
OUTPUT_DIR = EXPERIMENT_ROOT / "report_charts"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 11,
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "legend.fontsize": 11,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )


def plot_grouped(
    frame: pd.DataFrame,
    order_label: str,
    order_column: str,
    stage: str,
    output_name: str,
) -> None:
    nmos_column = f"01_NMOS_{order_column}_{stage}"
    aos_column = f"03_AOS_1_{order_column}_{stage}"
    labels = [f"{model}\n({dataset})" for model, dataset in zip(frame["网络"], frame["数据集"])]
    nmos = frame[nmos_column].astype(float).to_numpy()
    aos1 = frame[aos_column].astype(float).to_numpy()

    x = np.arange(len(frame))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10.8, 6.2), constrained_layout=True)
    bars_nmos = ax.bar(
        x - width / 2,
        nmos,
        width,
        label="NMOS",
        color="#E8873A",
        edgecolor="#9B4E16",
        linewidth=0.8,
        hatch="//",
        zorder=3,
    )
    bars_aos = ax.bar(
        x + width / 2,
        aos1,
        width,
        label="AOS1",
        color="#3678B5",
        edgecolor="#1E4E78",
        linewidth=0.8,
        zorder=3,
    )

    stage_title = "训练前" if stage == "训练前" else "误差感知训练后"
    ax.set_title(
        f"{stage_title}批量推理准确率\n{order_label}拟合 LUT（完整测试集）",
        pad=14,
        weight="bold",
    )
    ax.set_ylabel("Top-1 准确率")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 1.12)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_yticks(np.arange(0.0, 1.01, 0.1))
    ax.grid(axis="y", color="#D9DEE7", linewidth=0.8, alpha=0.85, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", ncols=2, frameon=False)

    for bars in (bars_nmos, bars_aos):
        ax.bar_label(
            bars,
            labels=[f"{value * 100:.2f}%" for value in bars.datavalues],
            padding=4,
            fontsize=9.5,
            rotation=0,
        )

    fig.savefig(OUTPUT_DIR / f"{output_name}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_DIR / f"{output_name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(SOURCE)
    required = {"网络", "数据集"}
    for order in ("低阶", "高阶"):
        for device in ("01_NMOS", "03_AOS_1"):
            for stage in ("训练前", "训练后"):
                required.add(f"{device}_{order}_{stage}")
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"汇总表缺少列：{sorted(missing)}")

    for order in ("低阶", "高阶"):
        report = frame[
            [
                "网络",
                "数据集",
                f"01_NMOS_{order}_训练前",
                f"03_AOS_1_{order}_训练前",
                f"01_NMOS_{order}_训练后",
                f"03_AOS_1_{order}_训练后",
            ]
        ].copy()
        report.to_csv(
            OUTPUT_DIR / f"{order}LUT训练前后批量推理准确率.csv",
            index=False,
            encoding="utf-8-sig",
        )
        plot_grouped(frame, order, order, "训练前", f"{order}LUT_训练前批量推理准确率")
        plot_grouped(frame, order, order, "训练后", f"{order}LUT_训练后批量推理准确率")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()

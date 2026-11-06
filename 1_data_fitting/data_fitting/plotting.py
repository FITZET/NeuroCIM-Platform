from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator
import numpy as np

from .models import FittingConfig, FittingResult


PAPER_COLORS = {
    "blue": "#2878B5",
    "red": "#C82423",
    "green": "#2F9E44",
    "gold": "#F2A104",
    "gray": "#666666",
}


def apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "axes.linewidth": 1.2,
            "lines.linewidth": 1.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
            "legend.fontsize": 10,
            "figure.dpi": 120,
            "savefig.bbox": "tight",
            "savefig.transparent": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, path: Path, config: FittingConfig) -> Path:
    fig.savefig(
        path, dpi=config.plot_dpi, facecolor="white",
        bbox_inches="tight", pad_inches=0.04,
    )
    plt.close(fig)
    return path


def plot_surface_overview(result: FittingResult, output_dir: Path, config: FittingConfig) -> Path:
    frame = result.dataset.frame
    measured = frame.pivot(index="W", columns="X", values="output").to_numpy(float)
    levels = result.dataset.code_max + 1
    fitted_frame = frame[["W", "X"]].copy()
    fitted_frame["fit"] = result.fits["full_surface"].predicted
    fitted = fitted_frame.pivot(index="W", columns="X", values="fit").reindex(index=range(levels), columns=range(levels)).to_numpy(float)
    residual = measured - fitted
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.75), constrained_layout=True)
    images = [measured, fitted, residual]
    titles = ["(a) Measured response", "(b) Bilinear fit", "(c) Residual"]
    cmaps = ["viridis", "viridis", "RdBu_r"]
    for ax, data, title, cmap in zip(axes, images, titles, cmaps):
        options = {}
        if title.endswith("Residual") and np.min(data) < 0 < np.max(data):
            bound = np.max(np.abs(data))
            options["norm"] = TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
        image = ax.imshow(data, origin="lower", aspect="auto", extent=(0, result.dataset.code_max, 0, result.dataset.code_max), cmap=cmap, **options)
        ax.set_title(title)
        ax.set_xlabel("Input code, X")
        ax.set_ylabel("Weight code, W")
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
        colorbar.ax.tick_params(labelsize=9, width=1.0)
    return _save(fig, output_dir / f"surface_overview.{config.image_format}", config)


def plot_fit_quality(result: FittingResult, output_dir: Path, config: FittingConfig) -> Path:
    fit = result.fits["full_surface"]
    actual = result.dataset.frame["output"].to_numpy(float)
    step = max(1, len(actual) // 8000)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 2.75), constrained_layout=True)
    axes[0].scatter(actual[::step], fit.predicted[::step], s=3, alpha=0.35, color=PAPER_COLORS["blue"], rasterized=True)
    low, high = min(actual.min(), fit.predicted.min()), max(actual.max(), fit.predicted.max())
    axes[0].plot([low, high], [low, high], color=PAPER_COLORS["red"], lw=1.8, label="Ideal agreement")
    axes[0].set(xlabel="Measured output", ylabel="Fitted output", title=f"(a) Fit quality ($R^2$ = {fit.metrics.r2:.5f})")
    axes[0].legend(frameon=False)
    axes[1].hist(fit.residual, bins=60, density=True, color=PAPER_COLORS["blue"], alpha=0.85, edgecolor="none")
    axes[1].axvline(0, color=PAPER_COLORS["red"], lw=1.8)
    axes[1].set(xlabel="Residual", ylabel="Probability density", title="(b) Residual distribution")
    axes[1].yaxis.set_major_locator(MaxNLocator(4))
    return _save(fig, output_dir / f"fit_quality.{config.image_format}", config)


def plot_ideal_product_vs_device(result: FittingResult, output_dir: Path, config: FittingConfig) -> Path:
    """Plot all measured code pairs against X*W and their least-squares line."""

    frame = result.dataset.frame
    product = frame["X"].to_numpy(float) * frame["W"].to_numpy(float)
    measured = frame["output"].to_numpy(float)
    fit = result.fits["raw_product_output"]
    if result.dataset.kind.value == "device":
        measured_plot = measured * 1e6
        slope_plot = fit.coefficients["slope"] * 1e6
        intercept_plot = fit.coefficients["intercept"] * 1e6
        ylabel = r"Measured device current, $I_{\mathrm{out}}$ ($\mu$A)"
        equation = rf"$I_{{\mathrm{{out}}}}={slope_plot:.3e}(XW)+{intercept_plot:.3f}$ $\mu$A"
    else:
        measured_plot = measured
        slope_plot = fit.coefficients["slope"]
        intercept_plot = fit.coefficients["intercept"]
        ylabel = "Actual output code"
        equation = rf"$Y={slope_plot:.6f}(XW)+{intercept_plot:.3e}$"

    fig, ax = plt.subplots(figsize=(8.0, 3.05), constrained_layout=True)
    ax.scatter(
        product,
        measured_plot,
        s=2.2,
        alpha=0.16,
        color=PAPER_COLORS["blue"],
        edgecolors="none",
        rasterized=True,
        label="Device data (65,536 code pairs)",
    )
    product_max = result.dataset.code_max**2
    x_line = np.linspace(0, product_max, 500)
    ax.plot(
        x_line,
        intercept_plot + slope_plot * x_line,
        color=PAPER_COLORS["red"],
        lw=2.0,
        label="Least-squares fit",
    )
    ax.text(
        0.04,
        0.96,
        equation + "\n" + rf"$R^2={fit.metrics.r2:.6f}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.9},
    )
    ax.set(
        xlabel=r"Ideal multiplication, $X\times W$ (code$^2$)",
        ylabel=ylabel,
        title=f"{result.dataset.name}: ideal product vs. device response",
        xlim=(0, product_max),
    )
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, output_dir / f"ideal_product_vs_device.{config.image_format}", config)


def plot_transfer_slices(result: FittingResult, output_dir: Path, config: FittingConfig) -> Path:
    frame = result.dataset.frame.copy()
    frame["fit"] = result.fits["full_surface"].predicted
    selected_w = tuple(np.unique(np.rint(np.linspace(0, result.dataset.code_max, 5)).astype(int)))
    fig, ax = plt.subplots(figsize=(8.0, 3.0), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(selected_w)))
    for weight, color in zip(selected_w, colors):
        subset = frame[frame["W"] == weight].sort_values("X")
        ax.plot(subset["X"], subset["output"], color=color, lw=2.0, label=f"W = {weight}")
        ax.plot(subset["X"], subset["fit"], color=color, lw=1.4, ls="--")
    ax.set(xlabel="Input code, X", ylabel="Output response", title="Measured (solid) and fitted (dashed) transfer curves")
    ax.legend(frameon=False, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, output_dir / f"transfer_slices.{config.image_format}", config)


def plot_lut_error(result: FittingResult, output_dir: Path, config: FittingConfig) -> Path:
    ideal = np.multiply.outer(np.arange(256), np.arange(256)).T
    error = result.lut.astype(float) - ideal
    bound = max(1.0, float(np.max(np.abs(error))))
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 2.75), constrained_layout=True)
    image = axes[0].imshow(error, origin="lower", extent=(0, 255, 0, 255), cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound))
    axes[0].set(xlabel="Input code, X", ylabel="Weight code, W", title="(a) Hardware LUT error")
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.03, label="Product-code error")
    for weight, color in zip((64, 128, 192, 255), plt.cm.viridis(np.linspace(0.1, 0.9, 4))):
        axes[1].plot(np.arange(256), error[weight], lw=1.7, color=color, label=f"W = {weight}")
    axes[1].axhline(0, color="black", lw=1.2)
    axes[1].set(xlabel="Input code, X", ylabel="Product-code error", title="(b) Error slices")
    axes[1].legend(frameon=False)
    axes[1].spines[["top", "right"]].set_visible(False)
    return _save(fig, output_dir / f"lut_error.{config.image_format}", config)


def _normalized_axis(result: FittingResult, title: str) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(8.0, 3.15), constrained_layout=True)
    ax.set(
        xlim=(-0.03, 1.03), ylim=(-0.05, 1.08),
        xlabel=r"Normalized ideal product, $p=XW/X_{max}^2$",
        ylabel="Normalized output", title=f"{result.dataset.name}: {title}",
    )
    ax.grid(True, color="#D9DDE3", linewidth=1.0, alpha=0.8)
    return fig, ax


def _metric_box(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.97, 0.06, text, transform=ax.transAxes, ha="right", va="bottom",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#AEB4BD", "alpha": 0.95},
    )


def plot_normalized_linear(result: FittingResult, output_dir: Path, config: FittingConfig) -> Path:
    """Normalized ideal product versus measured output and its linear fit."""

    frame = result.dataset.frame
    product = frame["X"].to_numpy(float) * frame["W"].to_numpy(float)
    product_max = float(result.dataset.code_max**2)
    normalized_product = product / product_max if product_max else product
    output = frame["output"].to_numpy(float)
    full_scale = float(output.max() - output.min())
    normalized_output = (output - output.min()) / full_scale if full_scale else np.zeros_like(output)
    fit = result.fits["linear_product"]
    slope, intercept = fit.coefficients["slope"], fit.coefficients["intercept"]

    fig, ax = _normalized_axis(result, "normalized linear least-squares fit")
    ax.scatter(normalized_product, normalized_output, s=2.5, alpha=0.08,
               color=PAPER_COLORS["blue"], edgecolors="none", rasterized=True,
               label="Normalized measured output")
    line_x = np.linspace(0, 1, 500)
    ax.plot(line_x, slope * line_x + intercept, color=PAPER_COLORS["red"], lw=2.2,
            label=rf"Fit: $T={slope:.4f}p{intercept:+.4f}$")
    ax.plot(line_x, line_x, "--", color="#202124", lw=1.6, label="Ideal y=x")
    ax.legend(loc="upper left", frameon=False)
    _metric_box(ax, f"$R^2$ = {fit.metrics.r2:.6f}\nRMSE = {fit.metrics.rmse*100:.4f}% FS\nNL = {fit.metrics.nl_percent_fs:.4f}% FS")
    return _save(fig, output_dir / f"normalized_linear.{config.image_format}", config)


def plot_normalized_surface(result: FittingResult, output_dir: Path, config: FittingConfig) -> Path:
    """Normalized ideal product with measured and complete-surface predictions."""

    frame = result.dataset.frame
    product = frame["X"].to_numpy(float) * frame["W"].to_numpy(float)
    product_max = float(result.dataset.code_max**2)
    normalized_product = product / product_max if product_max else product
    output = frame["output"].to_numpy(float)
    full_scale = float(output.max() - output.min())
    normalized_output = (output - output.min()) / full_scale if full_scale else np.zeros_like(output)
    fit = result.fits["full_surface"]
    predicted = (fit.predicted - output.min()) / full_scale if full_scale else np.zeros_like(output)

    fig, ax = _normalized_axis(result, "complete bilinear surface fit")
    ax.scatter(normalized_product, normalized_output, s=2.5, alpha=0.045, color="#7A7F87",
               edgecolors="none", rasterized=True, label="Normalized measured output")
    ax.scatter(normalized_product, predicted, s=2.5, alpha=0.10, color=PAPER_COLORS["red"],
               edgecolors="none", rasterized=True, label="Surface prediction")
    ax.plot([0, 1], [0, 1], "--", color="#202124", lw=1.6, label="Ideal y=x")
    ax.legend(loc="upper left", frameon=False)
    c = fit.coefficients
    _metric_box(ax, f"Surface $R^2$ = {fit.metrics.r2:.6f}\nα = {c['alpha']:.5g}\nβ = {c['beta']:.5g}\nγ = {c['gamma']:.5g}\nρ = {c['rho']:.5g}")
    return _save(fig, output_dir / f"normalized_surface.{config.image_format}", config)


def plot_normalized_lut(result: FittingResult, output_dir: Path, config: FittingConfig) -> Path:
    """Canonical four-term differential LUT versus normalized ideal product."""

    ideal = np.multiply.outer(np.arange(256), np.arange(256)).T.astype(float)
    lut_float = result.lut_gain * result.interaction_current
    lut = result.lut.astype(float)
    residual = lut - ideal
    centered = ideal - ideal.mean()
    denominator = float(np.sum(centered**2))
    lut_r2 = 1.0 - float(np.sum((ideal - lut_float)**2)) / denominator
    p = ideal.ravel() / 65025.0

    fig, ax = _normalized_axis(result, "four-term differential multiplication LUT")
    ax.scatter(p, lut.ravel() / 65025.0, s=2.5, alpha=0.10, color=PAPER_COLORS["red"],
               edgecolors="none", rasterized=True, label="Normalized LUT output")
    ax.plot([0, 1], [0, 1], "--", color="#202124", lw=1.8, label="Ideal y=x")
    ax.legend(loc="upper left", frameon=False)
    _metric_box(ax, f"LUT $R^2$ = {lut_r2:.6f}\nRMSE = {np.sqrt(np.mean(residual**2)):.2f} code\nMAE = {np.mean(np.abs(residual)):.2f} code\ngain = {result.lut_gain:.7g}")
    return _save(fig, output_dir / f"normalized_lut.{config.image_format}", config)


def generate_all_plots(result: FittingResult, output_dir: Path, config: FittingConfig) -> dict[str, Path]:
    apply_paper_style()
    return {
        "surface_plot": plot_surface_overview(result, output_dir, config),
        "quality_plot": plot_fit_quality(result, output_dir, config),
        "ideal_product_fit_plot": plot_ideal_product_vs_device(result, output_dir, config),
        "slice_plot": plot_transfer_slices(result, output_dir, config),
        "lut_error_plot": plot_lut_error(result, output_dir, config),
        "normalized_linear_plot": plot_normalized_linear(result, output_dir, config),
        "normalized_surface_plot": plot_normalized_surface(result, output_dir, config),
        "normalized_lut_plot": plot_normalized_lut(result, output_dir, config),
    }

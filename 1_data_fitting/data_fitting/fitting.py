from __future__ import annotations

import numpy as np

from .models import DeviceDataset, FitMetrics, ModelFit


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> FitMetrics:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    residual = actual - predicted
    ss_res = float(residual @ residual)
    centered = actual - actual.mean()
    ss_tot = float(centered @ centered)
    r2 = 1.0 if ss_tot == 0 and ss_res == 0 else (float("nan") if ss_tot == 0 else 1 - ss_res / ss_tot)
    full_scale = float(actual.max() - actual.min())
    max_abs = float(np.max(np.abs(residual)))
    return FitMetrics(
        r2=r2,
        rmse=float(np.sqrt(np.mean(residual**2))),
        mae=float(np.mean(np.abs(residual))),
        max_abs_error=max_abs,
        full_scale=full_scale,
        nl_percent_fs=0.0 if full_scale == 0 and max_abs == 0 else 100 * max_abs / full_scale,
    )


def _least_squares(features: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coefficients, *_ = np.linalg.lstsq(features, target, rcond=None)
    return coefficients, features @ coefficients


def fit_code_voltage(dataset: DeviceDataset) -> dict[str, ModelFit]:
    frame = dataset.frame
    results: dict[str, ModelFit] = {}
    for code, voltage in (("X", "VD"), ("W", "VG")):
        matrix = np.column_stack([np.ones(len(frame)), frame[code]])
        coefficients, predicted = _least_squares(matrix, frame[voltage].to_numpy())
        actual = frame[voltage].to_numpy()
        results[f"{voltage.lower()}_calibration"] = ModelFit(
            name=f"{voltage} code calibration",
            equation=f"{voltage} = offset + slope * {code}",
            coefficients={"offset": float(coefficients[0]), "slope": float(coefficients[1])},
            metrics=calculate_metrics(actual, predicted),
            predicted=predicted,
            residual=actual - predicted,
            metadata={"target_unit": "code" if dataset.kind.value == "ideal" else "V"},
        )
    return results


def fit_linear_product(dataset: DeviceDataset) -> ModelFit:
    frame = dataset.frame
    product = frame["X"].to_numpy(float) * frame["W"].to_numpy(float)
    output = frame["output"].to_numpy(float)
    full_scale = output.max() - output.min()
    normalized_output = (output - output.min()) / full_scale if full_scale else np.zeros_like(output)
    product_max = float(dataset.code_max**2)
    normalized_product = product / product_max if product_max else product
    matrix = np.column_stack([np.ones(len(frame)), normalized_product])
    coefficients, predicted = _least_squares(matrix, normalized_output)
    return ModelFit(
        name="Linear ideal-product fit",
        equation=f"T = intercept + slope * (X/{dataset.code_max}) * (W/{dataset.code_max})",
        coefficients={"intercept": float(coefficients[0]), "slope": float(coefficients[1])},
        metrics=calculate_metrics(normalized_output, predicted),
        predicted=predicted,
        residual=normalized_output - predicted,
        metadata={"target": "normalized raw output", "unit": "normalized FS"},
    )


def fit_raw_product_output(dataset: DeviceDataset) -> ModelFit:
    """Fit physical device output directly against the ideal integer product."""

    frame = dataset.frame
    product = frame["X"].to_numpy(float) * frame["W"].to_numpy(float)
    target = frame["output"].to_numpy(float)
    matrix = np.column_stack([np.ones(len(frame)), product])
    coefficients, predicted = _least_squares(matrix, target)
    target_unit = "code" if dataset.kind.value == "ideal" else "A"
    return ModelFit(
        name="Physical output versus ideal product",
        equation="output = intercept + slope * (X * W)",
        coefficients={"intercept": float(coefficients[0]), "slope": float(coefficients[1])},
        metrics=calculate_metrics(target, predicted),
        predicted=predicted,
        residual=target - predicted,
        metadata={
            "x": "ideal integer product X*W",
            "x_range": [0, dataset.code_max**2],
            "target": "measured device output",
            "target_unit": target_unit,
        },
    )


def fit_quadratic_surface(dataset: DeviceDataset) -> ModelFit:
    frame = dataset.frame
    v = frame["X"].to_numpy(float) / dataset.code_max
    w = frame["W"].to_numpy(float) / dataset.code_max
    output = frame["output"].to_numpy(float)
    full_scale = output.max() - output.min()
    target = (output - output.min()) / full_scale if full_scale else np.zeros_like(output)
    matrix = np.column_stack([np.ones(len(frame)), v, w, v**2, w**2, v * w])
    coefficients, predicted = _least_squares(matrix, target)
    names = ("theta_0", "theta_v", "theta_w", "theta_v2", "theta_w2", "theta_vw")
    return ModelFit(
        name="Six-term quadratic response surface",
        equation="T = theta_0 + theta_v*v + theta_w*w + theta_v2*v^2 + theta_w2*w^2 + theta_vw*v*w",
        coefficients=dict(zip(names, map(float, coefficients))),
        metrics=calculate_metrics(target, predicted),
        predicted=predicted,
        residual=target - predicted,
        metadata={"v": f"X/{dataset.code_max}", "w": f"W/{dataset.code_max}", "unit": "normalized FS"},
    )


def fit_full_surface(dataset: DeviceDataset) -> ModelFit:
    frame = dataset.frame
    vd = frame["VD"].to_numpy(float)
    vg = frame["VG"].to_numpy(float)
    target = frame["output"].to_numpy(float)
    matrix = np.column_stack([vd * vg, vd, vg, np.ones(len(frame))])
    coefficients, predicted = _least_squares(matrix, target)
    names = ("alpha", "beta", "gamma", "rho")
    return ModelFit(
        name="Complete bilinear transfer surface",
        equation="output = alpha*VD*VG + beta*VD + gamma*VG + rho",
        coefficients=dict(zip(names, map(float, coefficients))),
        metrics=calculate_metrics(target, predicted),
        predicted=predicted,
        residual=target - predicted,
        metadata={"target_unit": "code" if dataset.kind.value == "ideal" else "A"},
    )


def build_interaction_lut(
    dataset: DeviceDataset, lut_min: int = 0, lut_max: int = 65535
) -> tuple[np.ndarray, np.ndarray, float, FitMetrics]:
    frame = dataset.frame
    native_levels = dataset.code_max + 1
    output_grid = (
        frame.pivot(index="W", columns="X", values="output")
        .reindex(index=range(native_levels), columns=range(native_levels))
        .to_numpy(float)
    )
    if np.isnan(output_grid).any():
        # A dropped/incomplete acquisition grid is completed in normalized code
        # space for LUT construction. The original frame remains unchanged and
        # records the selected missing-data policy.
        coordinates = np.arange(native_levels)
        for row in range(native_levels):
            valid = np.isfinite(output_grid[row])
            if valid.any():
                output_grid[row] = np.interp(coordinates, coordinates[valid], output_grid[row, valid])
        for column in range(native_levels):
            valid = np.isfinite(output_grid[:, column])
            if valid.any():
                output_grid[:, column] = np.interp(coordinates, coordinates[valid], output_grid[valid, column])
        if np.isnan(output_grid).any():
            raise ValueError("Incomplete acquisition grid cannot be interpolated: an entire row or column is absent")
    native_interaction = output_grid - output_grid[:, [0]] - output_grid[[0], :] + output_grid[0, 0]
    if native_levels == 256:
        interaction = native_interaction
    else:
        # Low-bit inputs describe a coarse acquisition grid. Interpolate that
        # physical surface onto the canonical uint8 inference grid; do not
        # relabel or truncate the measured codes as uint8 samples.
        source_axis = np.linspace(0.0, 1.0, native_levels)
        target_axis = np.linspace(0.0, 1.0, 256)
        along_x = np.vstack([np.interp(target_axis, source_axis, row) for row in native_interaction])
        interaction = np.vstack([
            np.interp(target_axis, source_axis, along_x[:, column]) for column in range(256)
        ]).T
    product = np.multiply.outer(np.arange(256), np.arange(256)).T.astype(float)
    denominator = float(np.sum(interaction**2))
    gain = 0.0 if denominator == 0 else float(np.sum(interaction * product) / denominator)
    lut = np.clip(np.rint(gain * interaction), lut_min, lut_max).astype(np.uint16)
    predicted_interaction = product / gain if gain != 0 else np.zeros_like(product)
    metrics = calculate_metrics(interaction.ravel(), predicted_interaction.ravel())
    return interaction, lut, gain, metrics

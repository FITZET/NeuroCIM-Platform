# DRAM-CIM data-fitting backend

This directory contains a GUI-independent backend for importing complete uint8
device-response grids, fitting device models, extracting interaction current,
building multiplication LUTs, and exporting publication-style figures.

## Command-line verification

```powershell
python 1_data_fitting/cli.py 0_data/01_NMOS.csv -o 1_data_fitting/output/01_NMOS
```

Use `--format pdf` or `--format svg` for vector figures.

## GUI-facing API

```python
from data_fitting import DataFittingService, FittingConfig

service = DataFittingService()
dataset = service.load(csv_path)
result = service.fit(
    dataset,
    output_dir=result_dir,
    config=FittingConfig(image_format="png", plot_dpi=300),
    progress=lambda percent, message: update_progress_bar(percent, message),
)
```

The GUI should run `fit` in a worker thread. The service does not open windows,
and the optional progress callback can be connected to GUI signals.

## Outputs

- `fitting_summary.json`: coefficients, equations, metrics, provenance, and artifacts
- `fitting_points.csv`: canonical input, predictions, and residuals for every model
- `interaction_current.csv`: four-term-difference interaction grid
- `hardware_multiplication_lut.csv`: uint16 256 x 256 LUT
- `surface_overview.*`: measured, fitted, and residual maps
- `fit_quality.*`: agreement plot and residual distribution
- `ideal_product_vs_device.*`: ideal integer product on x-axis and measured device output on y-axis
- `transfer_slices.*`: measured/fitted code sweeps
- `lut_error.*`: LUT product-error map and slices

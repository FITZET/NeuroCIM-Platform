# Disposable layer-sensitivity analysis

Everything in this directory is diagnostic and disposable. No production code
imports this package. Deleting the complete `temp` directory does not affect
training, quantized inference, or fault inference.

Run from the project root:

```powershell
python 3_models/LeNet_5/temp/layer_sensitivity.py
python 3_models/LeNet_5/temp/alternative_error_schemes.py
python 3_models/LeNet_5/temp/outsource_style_eval.py
```

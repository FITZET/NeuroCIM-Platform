# Binary MNIST formal experiment

This backend trains a real BNN on MNIST. Every learned `Conv2d` and `Linear`
weight is `sign(w)`. Hidden activations are `sign(BatchNorm(x))`; the first
image tensor and final class logits stay real. The backward pass uses the
clipped identity STE: gradient 1 on `[-1,1]`, 0 outside.

```powershell
$env:NEUROCIM_MNIST_ROOT = 'path/to/dataset-root'
python 3_models/BNN/main.py train --epochs 15
python 3_models/BNN/main.py evaluate
python 3_models/BNN/main.py differential --differential-samples 4
```

`output/formal/best.pt` and `latest.pt` contain model, optimizer, scheduler,
history, epoch, configuration, and RNG state. Sidecar files contain SHA256.
The differential command freezes binary weights into ordinary Conv/Linear
layers and uses the platform's exact positive/negative signed LUT arithmetic.
Because this exact simulator is expensive, its fixed small-sample report is a
mechanism check and is explicitly not a full-test-set accuracy statistic.

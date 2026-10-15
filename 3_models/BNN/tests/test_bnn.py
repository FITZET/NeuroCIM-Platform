from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
sys.path[:0] = [str(ROOT), str(PROJECT / "5_NeuroCIM_opt")]

import random

from bnn.engine import capture_rng_state, load_checkpoint, save_checkpoint, seed_everything
from bnn.model import BinaryMNIST, binary_sign_ste
from neurocim_opt.layers import DifferentialLinear


def test_binary_forward_values_and_ste_gradient() -> None:
    value = torch.tensor([-2.0, -.5, 0.0, .5, 2.0], requires_grad=True)
    binary_sign_ste(value).sum().backward()
    assert binary_sign_ste(value.detach()).tolist() == [-1, -1, 1, 1, 1]
    assert value.grad.tolist() == [0, 1, 1, 1, 0]


def test_frozen_graph_uses_standard_layers_and_is_equivalent() -> None:
    torch.manual_seed(1); model = BinaryMNIST().eval(); frozen = model.frozen().eval()
    sample = torch.randn(2, 1, 28, 28)
    torch.testing.assert_close(model(sample), frozen(sample))
    assert all(type(m) in (nn.Conv2d, nn.Linear) for m in frozen.modules()
               if isinstance(m, (nn.Conv2d, nn.Linear)))
    weights = torch.cat([m.weight.flatten() for m in frozen.modules()
                         if isinstance(m, (nn.Conv2d, nn.Linear))])
    assert set(weights.unique().tolist()) <= {-1.0, 1.0}


def test_checkpoint_round_trip_includes_training_state(tmp_path: Path) -> None:
    model = BinaryMNIST(); optimizer = torch.optim.Adam(model.parameters()); scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    payload = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
               "scheduler_state_dict": scheduler.state_dict(), "history": [{"epoch": 1}], "epoch": 1,
               "best_validation_accuracy": .5, "config": {}, "rng_state": capture_rng_state()}
    path = tmp_path / "state.pt"; digest = save_checkpoint(path, payload)
    restored = BinaryMNIST(); loaded = load_checkpoint(path, restored, optimizer, scheduler)
    assert loaded["history"] == [{"epoch": 1}]
    for expected, actual in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(expected, actual)
    assert len(digest) == 64 and path.with_suffix(".pt.sha256").is_file()


def test_resume_restores_all_rngs_and_continuous_training_state(tmp_path: Path) -> None:
    seed_everything(19)
    model = nn.Linear(3, 2); optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=.9)
    inputs = torch.tensor([[.2, -.1, .4]]); target = torch.tensor([[.3, -.2]])

    def step(m, o, s):
        o.zero_grad(); loss = (m(inputs) - target).square().mean(); loss.backward(); o.step(); s.step()

    step(model, optimizer, scheduler)
    payload = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
               "scheduler_state_dict": scheduler.state_dict(), "history": [], "epoch": 1,
               "best_validation_accuracy": 0., "config": {}, "rng_state": capture_rng_state()}
    path = tmp_path / "resume.pt"; save_checkpoint(path, payload)
    expected_random = (random.random(), float(np.random.rand()), float(torch.rand(())))
    expected_cuda = float(torch.rand((), device="cuda")) if torch.cuda.is_available() else None
    step(model, optimizer, scheduler)
    continuous = [parameter.detach().clone() for parameter in model.parameters()]

    random.random(); np.random.rand(); torch.rand(7)
    restored = nn.Linear(3, 2); restored_optimizer = torch.optim.Adam(restored.parameters(), lr=.01)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, 1, gamma=.9)
    load_checkpoint(path, restored, restored_optimizer, restored_scheduler)
    actual_random = (random.random(), float(np.random.rand()), float(torch.rand(())))
    actual_cuda = float(torch.rand((), device="cuda")) if torch.cuda.is_available() else None
    assert actual_random == expected_random and actual_cuda == expected_cuda
    step(restored, restored_optimizer, restored_scheduler)
    for expected, actual in zip(continuous, restored.parameters()):
        torch.testing.assert_close(expected, actual)


def test_resume_restores_train_loader_generator_permutation(tmp_path: Path) -> None:
    model = nn.Linear(2, 1); optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    generator = torch.Generator().manual_seed(42)
    torch.randperm(17, generator=generator)
    payload = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
               "scheduler_state_dict": scheduler.state_dict(), "history": [], "epoch": 1,
               "best_validation_accuracy": 0., "config": {}, "rng_state": capture_rng_state(),
               "train_loader_generator_state": generator.get_state()}
    path = tmp_path / "loader.pt"; save_checkpoint(path, payload)
    expected = torch.randperm(17, generator=generator)
    restored_generator = torch.Generator().manual_seed(999)
    load_checkpoint(path, model, optimizer, scheduler,
                    train_loader_generator=restored_generator)
    actual = torch.randperm(17, generator=restored_generator)
    assert torch.equal(actual, expected)


def test_ideal_differential_linear_matches_quantized_signed_sum() -> None:
    codes = torch.arange(256, dtype=torch.int64); lut = codes[:, None] * codes[None, :]
    source = nn.Linear(3, 1, bias=False); source.weight.data[:] = torch.tensor([[1., -1., 1.]])
    layer = DifferentialLinear.from_module(source, lut=lut, input_scale=1/255, weight_scale=1/255)
    value = torch.tensor([[1., -1., .5]])
    expected = torch.tensor([[2.0 + round(.5 * 255) / 255]])
    torch.testing.assert_close(layer(value), expected)


def test_nonideal_lut_changes_signed_differential_result() -> None:
    codes = torch.arange(256, dtype=torch.int64); ideal = codes[:, None] * codes[None, :]
    nonideal = ideal.clone(); nonideal[255, 255] -= 255
    source = nn.Linear(1, 1, bias=False); source.weight.data.fill_(1)
    a = DifferentialLinear.from_module(source, lut=ideal, input_scale=1/255, weight_scale=1/255)
    b = DifferentialLinear.from_module(source, lut=nonideal, input_scale=1/255, weight_scale=1/255)
    assert a(torch.ones(1, 1)).item() == pytest.approx(1.0)
    assert b(torch.ones(1, 1)).item() < 1.0

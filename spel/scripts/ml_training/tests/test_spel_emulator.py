from pathlib import Path

import pytest
import torch

from spel.scripts.ml_training.model import MLPConfig, SpelEmulator


def test_forward_shape():
    config = MLPConfig(
        in_dim=5,
        out_dim=3,
        hidden_dim=16,
        num_layers=2,
    )

    emulator = SpelEmulator(
        config,
        casename="dummy",
    )

    x = torch.randn(10, config.in_dim)

    y = emulator(x)

    assert y.shape == (10, config.out_dim)


def test_default_buffers_are_identity():
    config = MLPConfig(
        in_dim=4,
        out_dim=2,
        hidden_dim=8,
        num_layers=1,
    )

    emulator = SpelEmulator(config, "dummy")

    x = torch.randn(7, config.in_dim)

    with torch.inference_mode():
        expected = emulator.network(x)
        actual = emulator(x)

    torch.testing.assert_close(actual, expected)


def test_checkpoint_roundtrip(tmp_path):
    config = MLPConfig(
        in_dim=5,
        out_dim=3,
        hidden_dim=8,
        num_layers=2,
    )

    emulator = SpelEmulator(config, "dummy")

    with torch.no_grad():
        emulator.x_mean.fill_(2.0)
        emulator.x_std.fill_(3.0)

    emulator.make_checkpoint()
    loaded = SpelEmulator.from_checkpoint(emulator._checkpoint_fn)

    torch.testing.assert_close(
        emulator.x_mean,
        loaded.x_mean,
    )

    torch.testing.assert_close(
        emulator.x_std,
        loaded.x_std,
    )


def make_test_emulator(case_name: str) -> SpelEmulator:
    config = MLPConfig(
        in_dim=4,
        out_dim=1,
        hidden_dim=8,
        num_layers=1,
    )

    emulator = SpelEmulator(
        config,
        case_name,
    ).double()

    with torch.no_grad():
        emulator.x_mean.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        emulator.x_std.copy_(torch.tensor([2.0, 2.0, 4.0, 4.0]))
        emulator.y_mean.copy_(torch.tensor([10.0]))
        emulator.y_std.copy_(torch.tensor([3.0]))

        first = emulator.network[0]
        second = emulator.network[2]

        first.weight.copy_(
            torch.tensor(
                [
                    [0.10, 0.20, 0.30, 0.40],
                    [-0.20, 0.10, 0.40, -0.30],
                    [0.50, -0.10, 0.20, 0.10],
                    [0.30, 0.30, -0.20, 0.20],
                    [-0.40, 0.20, 0.10, 0.50],
                    [0.20, -0.50, 0.30, 0.10],
                    [0.10, 0.40, -0.30, 0.20],
                    [-0.30, 0.10, 0.20, 0.40],
                ],
                dtype=torch.float64,
            )
        )

        first.bias.copy_(
            torch.tensor(
                [0.1, -0.2, 0.3, 0.0, -0.1, 0.2, -0.3, 0.4],
                dtype=torch.float64,
            )
        )
        second.weight.copy_(
            torch.tensor(
                [[0.2, -0.1, 0.3, 0.4, -0.2, 0.1, 0.5, -0.3]],
                dtype=torch.float64,
            )
        )

        second.bias.copy_(torch.tensor([0.25], dtype=torch.float64))

    emulator.eval()
    return emulator


def test_spel_emulator_torchscript():
    import xarray as xr

    from spel.scripts.config import input_data_dir

    case_dir = input_data_dir / "tiny_case"
    case_dir.mkdir(parents=True, exist_ok=True)

    emulator = make_test_emulator("tiny_case")

    test_input = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 4.0, 7.0, 8.0],
            [-1.0, 0.0, 1.0, 2.0],
        ],
        dtype=torch.float64,
    )

    emulator.make_trace()

    trace_path = emulator._trace_fn

    loaded = torch.jit.load(
        case_dir / "spel_emulator_torchscript.pt",
        map_location="cpu",
    )
    loaded.eval()

    with torch.inference_mode():
        python_output = emulator(test_input)
        torchscript_output = loaded(test_input)

    torch.testing.assert_close(
        torchscript_output,
        python_output,
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    comparison_ds = xr.Dataset(
        data_vars={
            "input": (
                ("sample", "input_feature"),
                test_input.cpu().numpy(),
            ),
            "python_output": (
                ("sample", "output_feature"),
                python_output.cpu().numpy(),
            ),
            "torchscript_output": (
                ("sample", "output_feature"),
                torchscript_output.cpu().numpy(),
            ),
        },
        coords={
            "sample": range(test_input.shape[0]),
            "input_feature": range(test_input.shape[1]),
            "output_feature": range(python_output.shape[1]),
        },
        attrs={
            "model_file": trace_path.name,
            "input_dtype": str(test_input.dtype),
            "description": (
                "Reference input and output for validating "
                "SpelEmulator inference through FTorch."
            ),
        },
    )

    output_path = case_dir / "spel_emulator_python_inference.nc"

    comparison_ds.to_netcdf(output_path)
    assert output_path.exists()

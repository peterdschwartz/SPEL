from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from spel.scripts.config import input_data_dir
from spel.scripts.ml_training.mlp import build_mlp


@dataclass(frozen=True)
class MLPConfig:
    in_dim: int
    out_dim: int
    hidden_dim: int
    num_layers: int


class SpelEmulator(nn.Module):
    def __init__(
        self,
        params: MLPConfig,
        casename: str,
    ):
        super().__init__()
        self.case_name = casename
        self._checkpoint_fn = input_data_dir / casename / "spel_emulator_checkpoint.pt"
        self._trace_fn = input_data_dir / casename / "spel_emulator_torchscript.pt"

        self.network = build_mlp(
            in_dim=params.in_dim,
            out_dim=params.out_dim,
            hidden_dim=params.hidden_dim,
            num_layers=params.num_layers,
        )

        self.register_buffer(
            "x_mean",
            torch.zeros(params.in_dim, dtype=torch.float32),
        )
        self.register_buffer(
            "x_std",
            torch.ones(params.in_dim, dtype=torch.float32),
        )
        self.register_buffer(
            "y_mean",
            torch.zeros(params.out_dim, dtype=torch.float32),
        )
        self.register_buffer(
            "y_std",
            torch.ones(params.out_dim, dtype=torch.float32),
        )
        self.config = params

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        device: str | torch.device = "cpu",
    ) -> "SpelEmulator":

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

        params = MLPConfig(
            in_dim=checkpoint["config"]["in_dim"],
            out_dim=checkpoint["config"]["out_dim"],
            hidden_dim=checkpoint["config"]["hidden_dim"],
            num_layers=checkpoint["config"]["num_layers"],
        )

        emulator = cls(params, checkpoint_path.parent.name)

        emulator.load_state_dict(checkpoint["model_state"])
        emulator.to(device)
        emulator.eval()

        return emulator

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_normalized = (x - self.x_mean) / self.x_std
        y_normalized = self.network(x_normalized)
        return y_normalized * self.y_std + self.y_mean

    def make_checkpoint(self):
        """makes checkpoint file for the trained model"""
        self._checkpoint_fn.parent.mkdir(parents=True, exist_ok=True)
        original_device = next(self.parameters()).device

        was_training = self.training

        self.to("cpu")
        self.eval()

        torch.save(
            {"model_state": self.state_dict(), "config": asdict(self.config), "casename":self.case_name},
            self._checkpoint_fn,
        )
        self.to(original_device)
        if was_training:
            self.train()

        return

    def make_trace(self):
        """
        Makes Torchscript file for emulator and validates it by reading comparing
        an inference between the emulator loaded from the checkpoint and from the torchscript
        """

        self.make_checkpoint()

        chk_pt_model = SpelEmulator.from_checkpoint(checkpoint_path=self._checkpoint_fn)
        chk_pt_model.eval()

        example_input = torch.randn(
            (1, self.config.in_dim),
            dtype=torch.float32,
        )

        traced_model = torch.jit.trace(
            chk_pt_model,
            example_input,
        )

        traced_model.save(self._trace_fn)

        # Validation of the trace
        loaded_model = torch.jit.load(self._trace_fn)
        loaded_model.eval()

        test_input = torch.randn(8, self.config.in_dim)

        with torch.inference_mode():
            expected = self(test_input)
            actual = loaded_model(test_input)

        torch.testing.assert_close(
            actual,
            expected,
            rtol=1.0e-6,
            atol=1.0e-6,
        )


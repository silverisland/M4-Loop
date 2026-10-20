"""Executed by test_nextgen in an isolated copied host, not collected by pytest."""
# ruff: noqa: E402 -- install the import guard before importing the copied backbone.

from __future__ import annotations

import importlib.abc
import json
import os
import subprocess
import sys
from pathlib import Path


class RejectOtherBackbones(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"m4", "m1", "m4_loop", "NextGenPVForecast"}:
            raise AssertionError(f"Forbidden legacy import: {fullname}")
        if fullname.startswith("forecast_backbones.") and not (
            fullname == "forecast_backbones.m4_loop"
            or fullname.startswith("forecast_backbones.m4_loop.")
        ):
            raise AssertionError(f"Forbidden other backbone: {fullname}")
        return None


sys.meta_path.insert(0, RejectOtherBackbones())
sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd
import torch
import yaml
from interfaces.modeloutput import TransformersAdaptedModelOutput
from pipelines.model_registry import build_evaluator, build_model
from pipelines.pipeline_wrapper_inference import unified_inference
from safetensors.torch import load_file

from forecast_backbones.m4_loop.predict import predict
from forecast_backbones.m4_loop.train import COLLATE, build_dataset, train
from forecast_backbones.m4_loop.transformer_config import feature_columns, load_config

torch.set_num_threads(1)
cfg = load_config(json.loads(Path("config.json").read_text()))
frame = pd.read_parquet("windows.parquet")
model = build_model("m4_loop", cfg)
assert type(model).__name__ == "TransformersAdaptedM4LoopModel"
dataset = build_dataset(frame, cfg)
batch = COLLATE([dataset[0], dataset[1]])
output = model(**batch)
assert isinstance(output, TransformersAdaptedModelOutput)
assert output.logits.shape == (2, 4)
assert torch.isfinite(output.loss)
output.loss.backward()
assert all(p.grad is not None for p in model.parameters() if p.requires_grad)

evaluator = build_evaluator("m4_loop")
model.train()
evaluated = evaluator(model, dataset, COLLATE, cfg)
assert model.training  # Evaluator must restore the calling mode.
assert evaluated.station.tolist() == frame.station_id.tolist()
np.testing.assert_array_equal(
    evaluated.timestamp_win.to_numpy(), pd.to_datetime(frame.timestamp).to_numpy()
)

trainer = train(cfg, frame, frame, "run", max_steps=2, test_source=frame)
assert trainer.state.global_step == 2
root = Path("run/model_result")
assert (root / "model.safetensors").is_file()
assert (root / "model_config.yaml").is_file()
assert Path("run/result_val.parquet").is_file()
assert Path("run/result_test.parquet").is_file()
restored = build_model("m4_loop", load_config(root / "model_config.yaml")).eval()
restored.load_state_dict(load_file(str(root / "model.safetensors")), strict=True)
trainer.model.eval()
with torch.no_grad():
    torch.testing.assert_close(
        trainer.model(batch["batch_dict"]).logits,
        restored(batch["batch_dict"]).logits,
        rtol=0,
        atol=0,
    )

# Label-free input, no capacity, and a final incomplete batch.
unlabeled = frame.drop(columns=["observe_power_future", "cap_power_on"])
result = unified_inference(
    restored,
    unlabeled,
    timestamp_cols=["timestamp"],
    entity_cols=["timestamp", "station_id"],
    feature_cols=feature_columns(cfg, supervised=False),
    batch_size=2,
)
column = type(restored).__name__
assert len(result) == len(frame)
assert result.station_id.tolist() == frame.station_id.tolist()
assert all(isinstance(row, np.ndarray) and row.shape == (4,) for row in result[column])
assert np.isfinite(np.stack(result[column])).all()
local_result = predict(root, unlabeled, "predictions.parquet", device="cpu")
assert local_result.station.tolist() == result.station_id.tolist()
np.testing.assert_allclose(
    np.stack(local_result.observe_power_predict), np.stack(result[column]), rtol=1e-5, atol=1e-6
)
assert "observe_power_future" not in local_result
for name, module in tuple(sys.modules.items()):
    if name.startswith(("forecast_backbones.", "pipelines.", "interfaces.")):
        location = getattr(module, "__file__", None)
        if location:
            assert Path(location).resolve().is_relative_to(Path.cwd()), location
for entry in ("train", "predict"):
    result = subprocess.run(
        [sys.executable, "-m", f"forecast_backbones.m4_loop.{entry}", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr

# Exercise the optional patch against the real public training script, including
# default config discovery. Its legacy training utility is intentionally absent.
cfg["optimization"]["train_epochs"] = 1
Path("forecast_backbones/m4_loop/config/configs.yaml").write_text(yaml.safe_dump(cfg))
for split in ("train", "valid", "test"):
    frame.to_parquet(f"{split}.parquet", index=False)
legacy_entry = subprocess.run(
    [
        sys.executable,
        "-I",
        "train_shell/train_script.py",
        "--model_name",
        "m4_loop",
        "--dataset_path",
        ".",
        "--run_path",
        "public-entry-run",
    ],
    capture_output=True,
    text=True,
    timeout=60,
    env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"),
)
assert legacy_entry.returncode == 0, legacy_entry.stdout + legacy_entry.stderr
assert Path("public-entry-run/model_result/model.safetensors").is_file()
assert Path("public-entry-run/model_result/model_config.yaml").is_file()
print("NextGen copy integration passed")

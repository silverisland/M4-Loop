"""Load standard NextGen artifacts and predict without target columns."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file

from .train import COLLATE, build_dataset
from .transformer_config import load_config
from .transformer_models import TransformersAdaptedM4LoopModel
from .transformer_trainer import evaluate_model


def load_model(model_dir, device="auto"):
    """Load this backbone's own weights; no other backbone is imported."""
    root = Path(model_dir)
    cfg = load_config(root / "model_config.yaml")
    model = TransformersAdaptedM4LoopModel(cfg)
    model.load_state_dict(load_file(str(root / "model.safetensors")), strict=True)
    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    return model.to(device).eval()


def predict(model_dir, source, out_path=None, device="auto"):
    model = load_model(model_dir, device)
    dataset = build_dataset(source, model.config, supervised=False)
    result = evaluate_model(model, dataset, COLLATE, model.config)
    # Prediction needs no truth column. The evaluator's NaN placeholder is for metrics callers.
    result = result.drop(columns="observe_power_future")
    if out_path is not None:
        destination = Path(out_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(destination, index=False)
    return result


def main():
    parser = argparse.ArgumentParser(description="Predict with a trained NextGen M4-Loop model")
    parser.add_argument("--model-dir", required=True, help="Run's model_result directory")
    parser.add_argument("--data", required=True, help="Input parquet; no target required")
    parser.add_argument("--out", required=True, help="Output parquet")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    predict(args.model_dir, args.data, args.out, args.device)


if __name__ == "__main__":
    main()

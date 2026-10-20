from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader

from m4_loop.checkpoint import load_checkpoint, save_checkpoint
from m4_loop.data import WindowDataset, collate_windows
from m4_loop.model import M4LoopModel
from m4_loop.predict import predict
from m4_loop.train import evaluate, train


def test_evaluation_aggregates_valid_points_independent_of_batch_size(config, frame):
    model = M4LoopModel(config)
    dataset = WindowDataset(frame, config, supervised=True)
    one_batch = evaluate(
        model,
        DataLoader(dataset, batch_size=len(dataset), collate_fn=collate_windows),
        torch.device("cpu"),
    )
    uneven_batches = evaluate(
        model,
        DataLoader(dataset, batch_size=2, collate_fn=collate_windows),
        torch.device("cpu"),
    )
    assert one_batch["valid_points"] == 15
    for metric in ("loss", "mae", "rmse", "normalized_mae", "normalized_rmse"):
        assert np.isfinite(one_batch[metric])
        assert uneven_batches[metric] == pytest.approx(one_batch[metric], rel=2e-6, abs=2e-7)


def test_parquet_training_checkpoint_roundtrip_and_prediction(config, frame, tmp_path):
    parquet_path = tmp_path / "windows.parquet"
    frame.to_parquet(parquet_path, index=False)
    result = train(config, parquet_path, parquet_path, tmp_path / "training")
    checkpoint = Path(result["best_checkpoint"])
    assert checkpoint.is_file()
    assert 1 <= result["best_epoch"] <= config["optimization"]["train_epochs"]
    assert result["global_step"] > 0
    first, metadata = load_checkpoint(checkpoint, device="cpu")
    second, _ = load_checkpoint(checkpoint, device="cpu")
    assert metadata["epoch"] == result["best_epoch"]
    feature_batch = next(
        iter(DataLoader(WindowDataset(frame, config, supervised=False), batch_size=5, collate_fn=collate_windows))
    )
    first.eval()
    second.eval()
    with torch.no_grad():
        first_predictions = first(feature_batch["features"]).predictions
        second_predictions = second(feature_batch["features"]).predictions
    torch.testing.assert_close(first_predictions, second_predictions, rtol=0, atol=0)

    prediction_path = tmp_path / "predictions.parquet"
    output = predict(checkpoint, parquet_path, prediction_path, batch_size=2, device="cpu")
    assert prediction_path.is_file()
    assert len(output) == len(frame)
    np.testing.assert_allclose(
        np.stack(output["observe_power_predict"]),
        first_predictions.numpy(),
        rtol=2e-6,
        atol=2e-6,
    )
    assert output["station_id"].tolist() == frame["station_id"].tolist()


def test_cli_train_then_predict_from_other_working_directory(config, frame, tmp_path):
    """Exercise installed-style module imports and complete public CLI arguments."""
    import os

    root = Path(__file__).resolve().parents[1]
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    parquet_path = tmp_path / "windows.parquet"
    frame.to_parquet(parquet_path, index=False)
    out_dir = tmp_path / "cli-training"
    environment = dict(os.environ, PYTHONPATH=str(root), OMP_NUM_THREADS="1")
    train_process = subprocess.run(
        [
            sys.executable,
            "-m",
            "m4_loop.train",
            "--config",
            str(config_path),
            "--train",
            str(parquet_path),
            "--valid",
            str(parquet_path),
            "--out-dir",
            str(out_dir),
            "--max-steps",
            "2",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert train_process.returncode == 0, train_process.stdout + train_process.stderr
    output_path = tmp_path / "cli-predictions.parquet"
    predict_process = subprocess.run(
        [
            sys.executable,
            "-m",
            "m4_loop.predict",
            "--checkpoint",
            str(out_dir / "best.pt"),
            "--data",
            str(parquet_path),
            "--output",
            str(output_path),
            "--device",
            "cpu",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert predict_process.returncode == 0, predict_process.stdout + predict_process.stderr
    assert output_path.is_file()


def test_early_stopping_preserves_best_weights(config, frame, tmp_path, monkeypatch):
    """The last epoch deliberately scores worse; its weights must not replace best."""
    import importlib

    training = importlib.import_module("m4_loop.train")
    snapshots = []
    losses = iter([0.1, 0.3])

    def controlled_validation(model, loader, device):
        snapshots.append({key: value.detach().clone() for key, value in model.state_dict().items()})
        return {"loss": next(losses), "valid_points": 15}

    monkeypatch.setattr(training, "evaluate", controlled_validation)
    config["optimization"].update(train_epochs=5, patience=1)
    result = train(config, frame, frame, tmp_path / "best-selection")
    assert result["stop_reason"] == "early_stopping"
    assert result["best_epoch"] == 1
    assert result["epochs_completed"] == 2
    loaded, metadata = load_checkpoint(result["best_checkpoint"])
    assert metadata["metric"] == pytest.approx(0.1)
    assert any(not torch.equal(snapshots[0][key], snapshots[1][key]) for key in snapshots[0])
    for key, value in loaded.state_dict().items():
        torch.testing.assert_close(value, snapshots[0][key], rtol=0, atol=0)


def test_checkpoint_rejects_legacy_and_missing_weights(config, tmp_path):
    model = M4LoopModel(config)
    path = tmp_path / "bad.pt"
    torch.save(model.state_dict(), path)
    with pytest.raises(ValueError, match="format"):
        load_checkpoint(path)
    save_checkpoint(path, model, epoch=1, global_step=2, metric=0.5)
    payload = torch.load(path, weights_only=True)
    payload["state_dict"].pop("position_embedding")
    torch.save(payload, path)
    with pytest.raises(ValueError, match="strictly match"):
        load_checkpoint(path)


def test_unlabeled_predict_and_missing_label_filter(config, frame, tmp_path):
    missing_row = frame.iloc[[0]].copy()
    missing_row["observe_power_future"] = [np.full(4, np.nan)]
    dataset = WindowDataset(missing_row, config, supervised=False)
    assert len(dataset) == 1
    with pytest.raises(ValueError, match="no rows"):
        WindowDataset(missing_row, config, supervised=True)
    import pandas as pd

    supervised = WindowDataset(pd.concat([frame, missing_row]), config, supervised=True)
    assert supervised.stats["rows_without_labels"] == 1
    assert supervised.stats["rows_used"] == len(frame)
    path = tmp_path / "model.pt"
    save_checkpoint(path, M4LoopModel(config), epoch=1, global_step=1, metric=0.5)
    unlabeled = frame.drop(columns=["observe_power_future", "cap_power_on"])
    output = predict(path, unlabeled, device="cpu")
    assert len(output) == len(frame)
    assert output.columns.tolist() == ["station_id", "timestamp", "observe_power_predict"]


def test_training_refuses_to_overwrite_an_existing_run(config, frame, tmp_path):
    sentinel = tmp_path / "history.json"
    sentinel.write_text('{"existing_run":true}')
    with pytest.raises(FileExistsError):
        train(config, frame, frame, tmp_path)
    assert sentinel.read_text() == '{"existing_run":true}'

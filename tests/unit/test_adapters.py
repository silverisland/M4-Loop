from __future__ import annotations

import copy

import numpy as np
import pytest

from forecast_backbones.m4_loop.train import build_dataset
from forecast_backbones.m4_loop.transformer_config import feature_columns, load_config
from forecast_backbones.m4_loop.transformer_trainer import M4Loop_build_training_args


def test_dataset_derives_columns_for_nextgen_and_rejects_mismatch():
    cfg = load_config({"dataset": {"timestamp_col": "time", "entity_col": "station_id"}})
    assert cfg["dataset"]["ts_col"] == "time"
    assert cfg["dataset"]["entity_cols"] == ["time", "station_id"]
    assert cfg["dataset"]["feature_cols"] == feature_columns(cfg)
    assert "cap_power_on" not in feature_columns(cfg, supervised=False)
    with pytest.raises(ValueError, match="inconsistent"):
        load_config({"dataset": {"feature_cols": ["observe_power_future"]}})
    with pytest.raises(ValueError, match="inconsistent"):
        load_config({"dataset": {"label_cols": ["chronos"]}})


def test_m4_style_flags_and_depth_semantics():
    cfg = load_config({"model": {"e_layers": 1, "loop_steps": 4, "revin": 1, "activation": "ReLU"}})
    assert cfg["model"]["e_layers"] == 1
    assert cfg["model"]["loop_steps"] == 4
    assert cfg["model"]["revin"] is True
    assert cfg["model"]["activation"] == "relu"
    with pytest.raises(ValueError, match="block_layers"):
        load_config({"model": {"block_layers": 2}})


def test_training_args_honor_configuration_on_cpu(config, tmp_path):
    cfg = copy.deepcopy(config)
    cfg["optimization"].update(fp16=True, gradient_accumulation_steps=2, logging_steps=3)
    args = M4Loop_build_training_args(cfg, tmp_path, max_steps=2)
    assert args.device.type == "cpu"
    assert not args.fp16
    assert args.gradient_accumulation_steps == 2
    assert args.logging_steps == 3
    assert args.max_steps == 2
    assert args.label_names == ["labels"]
    assert args.load_best_model_at_end
    assert args.prediction_loss_only


def test_dataset_filters_only_fully_invalid_target_rows(config, frame):
    altered = frame.copy(deep=True)
    altered.at[0, "observe_power_future"] = np.full(4, np.nan)
    with pytest.warns(UserWarning, match="Dropped 1 rows"):
        dataset = build_dataset(altered, config)
    assert len(dataset) == len(frame) - 1
    with pytest.raises(ValueError, match="required columns"):
        build_dataset(frame.drop(columns=["cap_power_on"]), config)
    invalid = frame.copy(deep=True)
    invalid.at[0, "observe_power_future"] = np.zeros(3)
    with pytest.raises(ValueError, match="shape"):
        build_dataset(invalid, config)
    unlabeled = frame.drop(columns=["observe_power_future", "cap_power_on"])
    assert len(build_dataset(unlabeled, config, supervised=False)) == len(frame)

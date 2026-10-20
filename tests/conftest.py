"""Small deterministic windows keep architecture and end-to-end tests on CPU."""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def single_torch_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def config():
    from m4_loop.config import load_config

    return load_config(
        {
            "inference": {"input_len": 8, "pred_len": 4, "seq_len": 12},
            "dataset": {
                "conv_cols": ["general_ghi", "general_temp"],
                "conv_predict_cols": ["general_ghi_predict", "general_temp_predict"],
                "entity_col": "station_id",
                "timestamp_col": "timestamp",
            },
            "model": {
                "patch_len": 4,
                "stride": 4,
                "d_model": 16,
                "n_heads": 4,
                "d_ff": 32,
                "embedding_dim": 12,
                "head_hidden_dim": 8,
                "dropout": 0.0,
                "attn_dropout": 0.0,
                "head_dropout": 0.0,
            },
            "optimization": {
                "device": "cpu",
                "batch_size": 2,
                "train_epochs": 2,
                "num_workers": 0,
                "random_seed": 7,
            },
        }
    )


@pytest.fixture
def batch():
    generator = torch.Generator().manual_seed(13)
    return {
        "general_ghi": torch.rand(2, 8, generator=generator) * 600,
        "general_temp": torch.rand(2, 8, generator=generator) * 20 + 10,
        "general_ghi_predict": torch.rand(2, 4, generator=generator) * 600,
        "general_temp_predict": torch.rand(2, 4, generator=generator) * 20 + 10,
        "observe_power": torch.rand(2, 8, generator=generator) * 10,
        "chronos": torch.rand(2, 4, generator=generator) * 10,
        "observe_power_future": torch.rand(2, 4, generator=generator) * 10,
        "cap_power_on": torch.tensor([10.0, 20.0]),
    }


@pytest.fixture
def frame(batch):
    import numpy as np
    import pandas as pd

    rows = []
    for index in range(5):
        row = {
            name: value[index % 2].numpy().copy()
            for name, value in batch.items()
            if name != "cap_power_on"
        }
        row["cap_power_on"] = float(batch["cap_power_on"][index % 2])
        row["station_id"] = f"station-{index % 2}"
        row["timestamp"] = f"2026-01-0{index + 1}T00:00:00"
        if index == 1:
            row["observe_power_future"][1:] = np.nan
        if index == 3:
            row["observe_power_future"][2:] = np.inf
        rows.append(row)
    return pd.DataFrame(rows)

"""M4-style configuration for a self-contained NextGen forecasting backbone."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).with_name("config") / "configs.yaml"
_DEFAULT_CONFIG: dict[str, dict[str, Any]] = yaml.safe_load(
    DEFAULT_CONFIG.read_text(encoding="utf-8")
)


def _merge(base: dict[str, Any], override: Mapping[str, Any], prefix: str = "") -> None:
    for key, value in override.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if key not in base:
            raise ValueError(f"Unsupported configuration key: {path}")
        if isinstance(base[key], dict):
            if not isinstance(value, Mapping):
                raise ValueError(f"{path} must be a mapping")
            _merge(base[key], value, path)
        else:
            base[key] = deepcopy(value)


def _integer(value: Any, name: str, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")


def _number(value: Any, name: str, *, minimum: float, inclusive: bool = True) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value) or (value < minimum if inclusive else value <= minimum):
        comparison = ">=" if inclusive else ">"
        raise ValueError(f"{name} must be finite and {comparison} {minimum}")


def _validate(config: dict[str, Any]) -> None:
    dataset, inference, model, optimization = (
        config[name] for name in ("dataset", "inference", "model", "optimization")
    )
    for name in ("conv_cols", "conv_predict_cols"):
        values = dataset[name]
        if not isinstance(values, list) or len(values) != 2:
            raise ValueError(f"dataset.{name} must contain exactly two weather columns")
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError(f"dataset.{name} must contain non-empty column names")
    for name in ("pv_col", "tsfm_col", "target_col", "capacity_col", "timestamp_col", "entity_col"):
        if not isinstance(dataset[name], str) or not dataset[name].strip():
            raise ValueError(f"dataset.{name} must be a non-empty column name")
    columns = [
        *dataset["conv_cols"],
        *dataset["conv_predict_cols"],
        *(
            dataset[name]
            for name in (
                "pv_col",
                "tsfm_col",
                "target_col",
                "capacity_col",
                "timestamp_col",
                "entity_col",
            )
        ),
    ]
    if len(columns) != len(set(columns)):
        raise ValueError("dataset column names must be distinct")
    for name, value in inference.items():
        _integer(value, f"inference.{name}")
    if inference["seq_len"] != inference["input_len"] + inference["pred_len"]:
        raise ValueError("inference.seq_len must equal input_len + pred_len")
    for name in (
        "e_layers",
        "loop_steps",
        "d_model",
        "n_heads",
        "d_ff",
        "patch_len",
        "stride",
        "embedding_dim",
        "head_hidden_dim",
    ):
        _integer(model[name], f"model.{name}")
    if model["d_model"] % model["n_heads"]:
        raise ValueError("model.d_model must be divisible by model.n_heads")
    if model["patch_len"] > inference["seq_len"]:
        raise ValueError("model.patch_len must not exceed inference.seq_len")
    if model["stride"] > model["patch_len"]:
        raise ValueError("model.stride must not exceed model.patch_len (no uncovered gaps)")
    if (inference["seq_len"] - model["patch_len"]) % model["stride"]:
        raise ValueError(
            "patch_len and stride must cover seq_len exactly without dropping its tail"
        )
    for name in ("dropout", "attn_dropout", "head_dropout"):
        _number(model[name], f"model.{name}", minimum=0.0)
        if model[name] >= 1:
            raise ValueError(f"model.{name} must be < 1")
    if model["activation"] not in ("relu", "gelu"):
        raise ValueError("model.activation must be 'relu' or 'gelu'")
    if model["loss_type"] not in ("mae", "mse", "huber"):
        raise ValueError("model.loss_type must be 'mae', 'mse', or 'huber'")
    if not isinstance(model["revin"], bool):
        raise ValueError("model.revin must be a boolean")
    _number(model["revin_eps"], "model.revin_eps", minimum=0.0, inclusive=False)
    for name in (
        "train_epochs",
        "batch_size",
        "eval_batch_size",
        "patience",
        "gradient_accumulation_steps",
        "logging_steps",
    ):
        _integer(optimization[name], f"optimization.{name}")
    for name in ("random_seed", "num_workers"):
        _integer(optimization[name], f"optimization.{name}", minimum=0)
    _number(
        optimization["learning_rate"], "optimization.learning_rate", minimum=0.0, inclusive=False
    )
    _number(optimization["weight_decay"], "optimization.weight_decay", minimum=0.0)
    _number(optimization["grad_clip"], "optimization.grad_clip", minimum=0.0, inclusive=False)
    if optimization["random_seed"] >= 2**32:
        raise ValueError("optimization.random_seed must be less than 2**32")
    device = optimization["device"]
    if (
        not isinstance(device, str)
        or re.fullmatch(r"auto|cpu|mps|cuda(?::[0-9]+)?", device) is None
    ):
        raise ValueError("optimization.device must be auto, cpu, mps, cuda, or cuda:N")
    if not isinstance(optimization["fp16"], bool):
        raise ValueError("optimization.fp16 must be a boolean")
    _number(optimization["warmup_ratio"], "optimization.warmup_ratio", minimum=0.0)
    if optimization["warmup_ratio"] > 1:
        raise ValueError("optimization.warmup_ratio must be <= 1")


def _normalize(config: dict[str, Any], override: Mapping[str, Any]) -> None:
    """Derive the fields consumed by NextGen's common dataset loader."""
    dataset = config["dataset"]
    supplied = override.get("dataset", {})
    features = [
        *dataset["conv_cols"],
        *dataset["conv_predict_cols"],
        dataset["pv_col"],
        dataset["tsfm_col"],
        dataset["capacity_col"],
    ]
    expected = {
        "ts_col": dataset["timestamp_col"],
        "feature_cols": features,
        "label_cols": [dataset["target_col"]],
    }
    for name, value in expected.items():
        if name in supplied and dataset[name] != value:
            raise ValueError(f"dataset.{name} is inconsistent with the model column names")
        dataset[name] = value
    if "entity_cols" in supplied:
        entities = dataset["entity_cols"]
        valid = (
            isinstance(entities, list)
            and all(isinstance(value, str) for value in entities)
            and len(entities) == len(set(entities))
            and dataset["entity_col"] in entities
            and set(entities) <= {dataset["entity_col"], dataset["timestamp_col"]}
        )
        if not valid:
            raise ValueError(
                "dataset.entity_cols must contain entity_col and optionally timestamp_col"
            )
    else:
        dataset["entity_cols"] = [dataset["timestamp_col"], dataset["entity_col"]]
    model = config["model"]
    if isinstance(model["activation"], str):
        model["activation"] = model["activation"].lower()
    if isinstance(model["revin"], int) and model["revin"] in (0, 1):
        model["revin"] = bool(model["revin"])


def load_config(source: str | Path | Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Merge with local defaults and validate canonical and NextGen dataset fields.

    ``model.e_layers`` is the number of independently parameterized blocks;
    ``loop_steps`` repeats that block stack with shared weights. The encoder
    always uses Pre-LayerNorm and never carries residual attention scores.
    """
    config = deepcopy(_DEFAULT_CONFIG)
    if source is None:
        override: Mapping[str, Any] = {}
    elif isinstance(source, Mapping):
        override = source
    elif isinstance(source, (str, Path)):
        with Path(source).open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
        override = {} if loaded is None else loaded
    else:
        raise TypeError("config source must be a YAML path, mapping, or None")
    if not isinstance(override, Mapping):
        raise ValueError("configuration document must be a mapping")
    _merge(config, override)
    _normalize(config, override)
    _validate(config)
    return config


def feature_columns(
    config: str | Path | Mapping[str, Any] | None = None,
    supervised: bool = True,
) -> list[str]:
    """Return features, including capacity only for supervised use.

    Capacity is needed for supervised loss. Label-free model inference does not
    require a capacity value.
    """
    dataset = load_config(config)["dataset"]
    columns = [
        *dataset["conv_cols"],
        *dataset["conv_predict_cols"],
        dataset["pv_col"],
        dataset["tsfm_col"],
    ]
    if supervised:
        columns.append(dataset["capacity_col"])
    return columns

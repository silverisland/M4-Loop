"""Run from the NextGen root: python -m forecast_backbones.m4_loop.train."""

from __future__ import annotations

import argparse
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pipelines.transformer_dataset import UnifiedDataset
from transformers import EarlyStoppingCallback, Trainer, set_seed

from .transformer_config import feature_columns, load_config
from .transformer_models import TransformersAdaptedM4LoopModel
from .transformer_trainer import M4Loop_build_training_args, evaluate_model

COLLATE = partial(
    UnifiedDataset.collate_fn,
    timestamp_output="epoch_s",
    drop_str_type=False,
    for_hf_trainer=True,
)


def build_dataset(source, config, supervised=True):
    """Use NextGen's shared dataset and keep targets out of model features."""
    cfg = load_config(config)
    ds, horizon = cfg["dataset"], cfg["inference"]["pred_len"]
    frame = source if isinstance(source, pd.DataFrame) else pd.read_parquet(source)
    required = [ds["timestamp_col"], ds["entity_col"], *feature_columns(cfg, supervised)]
    if supervised:
        required.append(ds["target_col"])
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if supervised:
        usable = []
        for index, row in enumerate(frame[ds["target_col"]]):
            values = np.asarray(row, dtype=float)
            if values.shape != (horizon,):
                raise ValueError(f"{ds['target_col']} row {index} must have shape ({horizon},)")
            usable.append(bool(np.isfinite(values).any()))
        removed = len(usable) - sum(usable)
        if removed:
            warnings.warn(f"Dropped {removed} rows with no finite target values", stacklevel=2)
        frame = frame.loc[usable].reset_index(drop=True)
    if frame.empty:
        raise ValueError("Dataset has no usable rows")
    return UnifiedDataset(
        frame,
        timestamp_cols=[ds["timestamp_col"]],
        entity_cols=ds["entity_cols"],
        feature_cols=feature_columns(cfg, supervised),
        label_cols=[ds["target_col"]] if supervised else [],
        timestamp_output="epoch_s",
        for_hf_trainer=True,
    )


def train(config, train_source, valid_source, out_dir, max_steps=-1, test_source=None):
    """Train and save model_result/model.safetensors plus model_config.yaml."""
    cfg = load_config(config)
    args = M4Loop_build_training_args(cfg, out_dir, max_steps)
    set_seed(cfg["optimization"]["random_seed"])
    model = TransformersAdaptedM4LoopModel(cfg)
    train_ds = build_dataset(train_source, cfg)
    valid_ds = build_dataset(valid_source, cfg)
    trainer = Trainer(
        model=model,
        args=args,
        data_collator=COLLATE,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg["optimization"]["patience"])],
    )
    out = Path(out_dir)
    config_text = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    if trainer.is_world_process_zero():
        out.mkdir(parents=True, exist_ok=True)
        (out / "model_config.yaml").write_text(config_text, encoding="utf-8")
    result = trainer.train()
    trainer.save_model(str(out / "model_result"))
    trainer.save_state()
    trainer.save_metrics("train", result.metrics)
    trainer.save_metrics("eval", trainer.evaluate())
    if trainer.is_world_process_zero():
        (out / "model_result" / "model_config.yaml").write_text(config_text, encoding="utf-8")
        evaluate_model(model, valid_ds, COLLATE, cfg).to_parquet(
            out / "result_val.parquet", index=False
        )
        if test_source is not None:
            evaluate_model(model, build_dataset(test_source, cfg), COLLATE, cfg).to_parquet(
                out / "result_test.parquet", index=False
            )
    trainer.accelerator.wait_for_everyone()
    return trainer


def main():
    parser = argparse.ArgumentParser(description="Train M4-Loop inside NextGen")
    parser.add_argument("--config", default=None, help="Defaults to local config/configs.yaml")
    parser.add_argument("--train", required=True, help="Training parquet")
    parser.add_argument("--valid", required=True, help="Validation parquet")
    parser.add_argument("--test", default=None, help="Optional labeled test parquet")
    parser.add_argument("--out", required=True, help="Run output directory")
    parser.add_argument("--max-steps", type=int, default=-1, help="Short integration check limit")
    args = parser.parse_args()
    train(args.config, args.train, args.valid, args.out, args.max_steps, args.test)


if __name__ == "__main__":
    main()

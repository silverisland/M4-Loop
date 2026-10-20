"""NextGen training-argument and evaluator entry points for M4-Loop."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import TrainingArguments

from .transformer_config import load_config


def M4Loop_build_training_args(model_config, result_dir, max_steps=-1):
    """Use HF Trainer while retaining the entry-point convention used by M4."""
    opt = load_config(model_config)["optimization"]
    requested = opt["device"]
    cuda = torch.cuda.is_available()
    mps = torch.backends.mps.is_available()
    if requested.startswith("cuda") and not cuda:
        raise ValueError("CUDA was requested but is unavailable; use device: auto or cpu")
    if requested == "mps" and not mps:
        raise ValueError("MPS was requested but is unavailable; use device: auto or cpu")
    use_cpu = requested == "cpu" or (requested == "auto" and not (cuda or mps))
    eval_key = (
        "eval_strategy"
        if "eval_strategy" in inspect.signature(TrainingArguments).parameters
        else "evaluation_strategy"
    )
    args = TrainingArguments(
        output_dir=str(result_dir),
        per_device_train_batch_size=opt["batch_size"],
        per_device_eval_batch_size=opt["eval_batch_size"],
        gradient_accumulation_steps=opt["gradient_accumulation_steps"],
        num_train_epochs=opt["train_epochs"],
        learning_rate=opt["learning_rate"],
        weight_decay=opt["weight_decay"],
        warmup_ratio=opt["warmup_ratio"],
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        max_grad_norm=opt["grad_clip"],
        max_steps=max_steps,
        seed=opt["random_seed"],
        data_seed=opt["random_seed"],
        use_cpu=use_cpu,
        fp16=bool(opt["fp16"] and cuda and not use_cpu and requested != "mps"),
        dataloader_num_workers=opt["num_workers"],
        dataloader_pin_memory=not use_cpu,
        save_strategy="epoch",
        save_total_limit=2,
        save_safetensors=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        prediction_loss_only=True,
        label_names=["labels"],
        remove_unused_columns=False,
        report_to=[],
        logging_steps=opt["logging_steps"],
        **{eval_key: "epoch"},
    )
    if requested != "auto":
        desired = torch.device(requested)
        if args.device.type != desired.type or (
            desired.index is not None and args.device.index != desired.index
        ):
            raise ValueError(
                f"Trainer selected {args.device}, but {requested} was requested. "
                "For CUDA, select visible GPUs with CUDA_VISIBLE_DEVICES and use device: cuda."
            )
    return args


@torch.no_grad()
def evaluate_model(model, test_dataset, HF_collate_fn, config):
    """Return the standard M4 evaluation columns, in dataset order."""
    cfg = load_config(config)
    ds, horizon = cfg["dataset"], cfg["inference"]["pred_len"]
    loader = DataLoader(
        test_dataset,
        batch_size=cfg["optimization"]["eval_batch_size"],
        shuffle=False,
        collate_fn=HF_collate_fn,
    )
    timestamps = test_dataset.timestamp
    rows, offset = [], 0
    was_training = model.training
    model.eval()
    try:
        for batch in loader:
            bd = batch.get("batch_dict", batch)
            pred = model(bd).logits.detach().float().cpu().numpy()
            truth = batch.get("labels", {}).get(ds["target_col"])
            truth = (
                np.full((len(pred), horizon), np.nan)
                if truth is None
                else torch.as_tensor(truth).cpu().numpy()
            )
            stations = bd.get(ds["entity_col"], [None] * len(pred))
            for index in range(len(pred)):
                station = stations[index]
                rows.append(
                    {
                        "timestamp_win": timestamps[offset + index],
                        "station": station.item() if isinstance(station, torch.Tensor) else station,
                        "observe_power_predict": pred[index],
                        "observe_power_future": truth[index],
                    }
                )
            offset += len(pred)
    finally:
        model.train(was_training)
    return pd.DataFrame(
        rows,
        columns=["timestamp_win", "station", "observe_power_predict", "observe_power_future"],
    )

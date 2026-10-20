"""NextGen-compatible M4-Loop model, implemented entirely in this directory."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from interfaces.modeloutput import TransformersAdaptedModelOutput
from torch import Tensor, nn

from .layers.loop_transformer import FlattenPredictionHead, LoopedTransformerEncoder, RevIN
from .layers.losses import normalized_point_loss
from .transformer_config import load_config


class TransformersAdaptedM4LoopModel(nn.Module):
    """Joint weather/power patch forecaster with a shared Transformer stack.

    Forward inputs are named raw columns, each shaped [batch,time]. Weather and
    power include the complete history plus known future predictions before
    window normalization. Future targets are supplied only through ``labels``.
    """

    def __init__(
        self,
        config: str | Path | Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.config = load_config(config)
        self.labels_name = [self.config["dataset"]["target_col"]]
        inference, model = self.config["inference"], self.config["model"]
        self.input_len = inference["input_len"]
        self.prediction_length = inference["pred_len"]
        self.context_length = self.input_len
        self.seq_len = inference["seq_len"]
        self.channels = 3
        self.patch_len = model["patch_len"]
        self.stride = model["stride"]
        self.patch_num = (self.seq_len - self.patch_len) // self.stride + 1
        self.loss_type = model["loss_type"]
        self.use_revin = model["revin"]
        self.revin = RevIN(model["revin_eps"])
        self.patch_projection = nn.Linear(self.patch_len, model["d_model"])
        self.position_embedding = nn.Parameter(
            torch.empty(1, self.channels * self.patch_num, model["d_model"])
        )
        nn.init.uniform_(self.position_embedding, -0.02, 0.02)
        self.input_dropout = nn.Dropout(model["dropout"])
        self.encoder = LoopedTransformerEncoder(
            d_model=model["d_model"],
            n_heads=model["n_heads"],
            d_ff=model["d_ff"],
            block_layers=model["e_layers"],
            loop_steps=model["loop_steps"],
            dropout=model["dropout"],
            attn_dropout=model["attn_dropout"],
            activation=model["activation"],
        )
        self.head = FlattenPredictionHead(
            nf=self.channels * self.patch_num * model["d_model"],
            target_window=self.prediction_length,
            embedding_dim=model["embedding_dim"],
            head_hidden_dim=model["head_hidden_dim"],
            head_dropout=model["head_dropout"],
        )

    def _tensor(self, value: Any, name: str) -> Tensor:
        weight = self.patch_projection.weight
        try:
            return torch.as_tensor(value, dtype=weight.dtype, device=weight.device)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(f"{name} must contain numeric tensor values") from exc

    def _column(self, batch: Mapping[str, Any], name: str, length: int) -> Tensor:
        if name not in batch:
            raise ValueError(f"Missing required input column: {name}")
        value = self._tensor(batch[name], name)
        if value.ndim != 2 or value.shape[1] != length or value.shape[0] < 1:
            raise ValueError(
                f"Input {name} must have shape [B,{length}] with B >= 1; got {tuple(value.shape)}"
            )
        if not torch.isfinite(value).all():
            raise ValueError(f"Input {name} contains non-finite values")
        return value

    def _prepare_input(self, batch: Mapping[str, Any]) -> Tensor:
        dataset = self.config["dataset"]
        histories = [
            self._column(batch, name, self.input_len)
            for name in [*dataset["conv_cols"], dataset["pv_col"]]
        ]
        futures = [
            self._column(batch, name, self.prediction_length)
            for name in [*dataset["conv_predict_cols"], dataset["tsfm_col"]]
        ]
        batch_sizes = {value.shape[0] for value in [*histories, *futures]}
        if len(batch_sizes) != 1:
            raise ValueError("All input columns must have the same batch size")
        return torch.stack(
            [torch.cat([past, future], dim=-1) for past, future in zip(histories, futures)], dim=1
        )

    def compute_loss(self, pred: Tensor, target: Any, capacity: Any) -> Tensor:
        """Apply the configured fixed-capacity loss to explicit targets."""
        return normalized_point_loss(
            pred, target, capacity, horizon=self.prediction_length, loss_type=self.loss_type
        )

    def forward(
        self,
        batch_dict: Mapping[str, Any],
        labels: Mapping[str, Any] | Tensor | None = None,
    ) -> TransformersAdaptedModelOutput:
        if not isinstance(batch_dict, Mapping):
            raise TypeError("batch_dict must map configured column names to tensors")
        x = self._prepare_input(batch_dict)
        stats = None
        if self.use_revin:
            x, stats = self.revin.normalize(x)
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        tokens = self.patch_projection(patches)
        batch_size = tokens.shape[0]
        tokens = tokens.reshape(batch_size, self.channels * self.patch_num, -1)
        tokens = self.input_dropout(tokens + self.position_embedding)
        encoded = self.encoder(tokens)
        encoded = encoded.reshape(batch_size, self.channels, self.patch_num, -1)
        encoded = encoded.permute(0, 1, 3, 2)
        predictions, embeddings = self.head(encoded)
        if stats is not None:
            # The final channel is historical PV plus the future power prior.
            predictions = predictions * stats.std[:, -1, :] + stats.mean[:, -1, :]
        loss = None
        if labels is not None:
            dataset = self.config["dataset"]
            if isinstance(labels, Mapping):
                if dataset["target_col"] not in labels:
                    raise ValueError(f"labels is missing target column {dataset['target_col']}")
                target = labels[dataset["target_col"]]
            else:
                target = labels
            if dataset["capacity_col"] not in batch_dict:
                raise ValueError(f"Missing fixed capacity column: {dataset['capacity_col']}")
            loss = self.compute_loss(predictions, target, batch_dict[dataset["capacity_col"]])
        return TransformersAdaptedModelOutput(
            loss=loss, logits=predictions, predictions=predictions, embeddings=embeddings
        )

from __future__ import annotations

import copy

import pytest
import torch

from forecast_backbones.m4_loop.layers.loop_transformer import RevIN
from forecast_backbones.m4_loop.transformer_models import (
    TransformersAdaptedM4LoopModel as M4LoopModel,
)


def test_loops_reuse_parameters_and_execute_every_block(config):
    model = M4LoopModel(config)
    encoder = model.encoder
    assert len(encoder.blocks) == 2
    assert encoder.loop_steps == 2
    assert encoder.blocks[0] is not encoder.blocks[1]
    parameter_ids = {id(parameter) for parameter in encoder.parameters()}
    parameter_count = sum(parameter.numel() for parameter in encoder.parameters())
    calls = [0, 0]

    def count_call(index):
        def hook(_module, _inputs, _output):
            calls[index] += 1

        return hook

    hooks = [
        block.register_forward_hook(count_call(index)) for index, block in enumerate(encoder.blocks)
    ]
    try:
        encoder(torch.randn(2, 5, config["model"]["d_model"]))
        assert calls == [2, 2]
        encoder.loop_steps = 5
        encoder(torch.randn(2, 5, config["model"]["d_model"]))
        assert calls == [7, 7]
    finally:
        for hook in hooks:
            hook.remove()
    assert {id(parameter) for parameter in encoder.parameters()} == parameter_ids
    assert sum(parameter.numel() for parameter in encoder.parameters()) == parameter_count


def test_loop_gradients_match_explicit_unroll(config):
    """A detached loop boundary may look correct but loses earlier gradients."""
    encoder = M4LoopModel(config).encoder
    explicit = copy.deepcopy(encoder)
    source = torch.randn(2, 5, config["model"]["d_model"], requires_grad=True)
    explicit_source = source.detach().clone().requires_grad_()
    probe = torch.randn_like(source)

    result = encoder(source)
    (result * probe).sum().backward()

    expected = explicit_source
    for _ in range(explicit.loop_steps):
        for block in explicit.blocks:
            expected = block(expected)
    expected = explicit.final_norm(expected)
    (expected * probe).sum().backward()

    torch.testing.assert_close(result, expected)
    assert source.grad is not None
    torch.testing.assert_close(source.grad, explicit_source.grad)
    for (name, actual), (reference_name, reference) in zip(
        encoder.named_parameters(), explicit.named_parameters(), strict=True
    ):
        assert name == reference_name
        assert actual.grad is not None, f"{name} was detached from the loss"
        torch.testing.assert_close(actual.grad, reference.grad)


def test_prenorm_keeps_unnormalized_residual(config):
    block = M4LoopModel(config).encoder.blocks[0]
    source = torch.randn(2, 5, config["model"]["d_model"]) * 4 + 7
    with torch.no_grad():
        for parameter in block.self_attn.parameters():
            parameter.zero_()
        for parameter in block.ff.parameters():
            parameter.zero_()
    # With both residual branches zero, standard Pre-LN is exactly identity.
    torch.testing.assert_close(block(source), source, rtol=0, atol=0)


def test_forward_shapes_explicit_labels_and_finite_backward(config, batch):
    model = M4LoopModel(config)
    output = model(batch)
    assert output.loss is None
    assert output.predictions.shape == (2, 4)
    assert output.embeddings.shape == (2, 12)
    torch.testing.assert_close(output.logits, output.predictions)
    assert torch.isfinite(output.predictions).all()

    changed_targets = dict(batch, observe_power_future=batch["observe_power_future"] + 5000)
    # Even a target column accidentally retained in a feature batch cannot leak.
    torch.testing.assert_close(model(changed_targets).predictions, output.predictions)
    trained_output = model(batch, labels={"observe_power_future": batch["observe_power_future"]})
    assert trained_output.loss.ndim == 0
    assert torch.isfinite(trained_output.loss)
    trained_output.loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert all(
        block.self_attn.in_proj_weight.grad is not None
        and block.self_attn.in_proj_weight.grad.abs().sum() > 0
        for block in model.encoder.blocks
    )


@pytest.mark.parametrize("capacity_shape", ["scalar", "batch", "batch_column"])
def test_capacity_broadcast_is_per_station_even_when_batch_equals_horizon(config, capacity_shape):
    config["model"]["loss_type"] = "mae"
    model = M4LoopModel(config)
    # B == H exposes the common bug where [B] silently broadcasts along H.
    prediction = torch.tensor([[2.0, 4.0, 6.0, 8.0]]).repeat(4, 1)
    target = torch.zeros_like(prediction)
    per_station = torch.tensor([2.0, 4.0, 8.0, 16.0])
    if capacity_shape == "scalar":
        capacity = torch.tensor(2.0)
        expected = (prediction / 2.0).abs().mean()
    else:
        capacity = {
            "batch": per_station,
            "batch_column": per_station[:, None],
        }[capacity_shape]
        expected = (prediction / per_station[:, None]).abs().mean()
    torch.testing.assert_close(model.compute_loss(prediction, target, capacity), expected)


@pytest.mark.parametrize("loss_type", ["mae", "mse", "huber"])
def test_nonfinite_targets_are_masked_without_poisoning_gradients(config, loss_type):
    config["model"]["loss_type"] = loss_type
    model = M4LoopModel(config)
    prediction = torch.tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    target = torch.tensor([[0.0, float("nan"), float("inf"), 2.0]])
    capacity = torch.tensor([2.0])
    residual = torch.tensor([0.5, 1.0])
    expected = {
        "mae": residual.abs().mean(),
        "mse": residual.square().mean(),
        "huber": torch.nn.functional.smooth_l1_loss(residual, torch.zeros_like(residual)),
    }[loss_type]
    loss = model.compute_loss(prediction, target, capacity)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    torch.testing.assert_close(prediction.grad[0, 1:3], torch.zeros(2))


def test_invalid_loss_inputs_fail_instead_of_silent_broadcast(config):
    model = M4LoopModel(config)
    prediction = torch.ones(2, 4)
    target = torch.zeros(2, 4)
    for invalid_capacity in (
        None,
        torch.tensor(0.0),
        torch.tensor(-2.0),
        torch.tensor(float("nan")),
    ):
        with pytest.raises((ValueError, TypeError)):
            model.compute_loss(prediction, target, invalid_capacity)
    with pytest.raises(ValueError):
        model.compute_loss(prediction, target[:, :1], torch.tensor([10.0, 20.0]))
    with pytest.raises(ValueError, match="Time-varying"):
        model.compute_loss(prediction, target, torch.full_like(prediction, 10.0))
    with pytest.raises(ValueError):
        model.compute_loss(prediction, torch.full_like(target, float("nan")), torch.tensor(10.0))


@pytest.mark.parametrize(
    ("column", "invalid"),
    [
        ("general_ghi", torch.ones(2, 7)),
        ("chronos", torch.ones(2, 8)),
        ("observe_power", torch.full((2, 8), float("nan"))),
    ],
)
def test_malformed_features_fail_with_column_context(config, batch, column, invalid):
    model = M4LoopModel(config)
    with pytest.raises((ValueError, TypeError), match=column):
        model(dict(batch, **{column: invalid}))


def test_capacity_only_required_for_explicit_supervision(config, batch):
    model = M4LoopModel(config)
    no_capacity = {key: value for key, value in batch.items() if key != "cap_power_on"}
    assert model(no_capacity).loss is None
    with pytest.raises(ValueError, match="capacity"):
        model(no_capacity, labels=batch["observe_power_future"])


def test_revin_statistics_survive_interleaved_windows():
    layer = RevIN()
    first = torch.randn(2, 3, 12) * 4 + 20
    second = torch.randn(2, 3, 12) * 40 - 100
    normalized_first, first_stats = layer.normalize(first)
    normalized_second, second_stats = layer.normalize(second)
    torch.testing.assert_close(layer.denormalize(normalized_first, first_stats), first)
    torch.testing.assert_close(layer.denormalize(normalized_second, second_stats), second)
    assert not first_stats.mean.requires_grad
    assert not first_stats.std.requires_grad


@pytest.mark.parametrize(
    "override",
    [
        {"model": {"e_layers": 0}},
        {"model": {"pre_norm": True}},
        {"model": {"res_attention": True}},
        {"model": {"loop_steps": 0}},
        {"model": {"stride": 97}},
        {"dataset": {"tsfm_col": "observe_power_future"}},
        {"optimization": {"grad_clip": 0}},
        {"inference": {"seq_len": 865}},
    ],
)
def test_invalid_configuration_is_not_silently_ignored(override):
    with pytest.raises(ValueError):
        M4LoopModel(override)

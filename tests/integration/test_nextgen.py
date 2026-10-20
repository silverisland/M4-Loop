"""Exercise an actual copied NextGen host without any original backbone present."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_copy_into_nextgen_registry_training_reload_and_inference(config, frame, tmp_path):
    host = os.environ.get("NEXTGEN_ROOT")
    if not host:
        pytest.skip("Set NEXTGEN_ROOT to test against the real NextGen framework")
    # Copy only host interfaces/pipeline utilities. No existing M4 or other model is available.
    for name in (
        "pipelines/model_registry.py",
        "pipelines/transformer_dataset.py",
        "pipelines/pipeline_wrapper_inference.py",
        "interfaces/modeloutput.py",
        "train_shell/train_script.py",
    ):
        source = Path(host) / name
        assert source.is_file(), f"Required NextGen interface missing: {source}"
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    patch = ROOT / "integration/nextgen_train_script.patch"
    checked = subprocess.run(
        ["git", "apply", "--check", str(patch)], cwd=tmp_path, capture_output=True, text=True
    )
    assert checked.returncode == 0, checked.stderr
    applied = subprocess.run(
        ["git", "apply", str(patch)], cwd=tmp_path, capture_output=True, text=True
    )
    assert applied.returncode == 0, applied.stderr
    shutil.copytree(
        ROOT / "forecast_backbones/m4_loop",
        tmp_path / "forecast_backbones/m4_loop",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    frame.to_parquet(tmp_path / "windows.parquet", index=False)
    result = subprocess.run(
        [sys.executable, "-I", str(Path(__file__).with_name("nextgen_probe.py"))],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "NextGen copy integration passed" in result.stdout

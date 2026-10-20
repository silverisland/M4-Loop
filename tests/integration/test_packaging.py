"""Installed-package entrypoints, resources and dependency direction."""

from __future__ import annotations

import subprocess
import sys
from importlib.resources import files
from pathlib import Path

import pytest
import yaml

from m4_loop.config import load_config


def test_default_configuration_is_a_packaged_resource():
    resource = files("m4_loop").joinpath("resources/default.yaml")
    assert resource.is_file()
    assert yaml.safe_load(resource.read_text(encoding="utf-8")) == load_config()


@pytest.mark.parametrize(
    "arguments",
    [
        ["-m", "m4_loop", "--help"],
        ["-m", "m4_loop", "train", "--help"],
        ["-m", "m4_loop", "predict", "--help"],
        ["-m", "m4_loop.train", "--help"],
        ["-m", "m4_loop.predict", "--help"],
    ],
)
def test_module_entrypoints_work_outside_checkout(arguments, tmp_path):
    result = subprocess.run(
        [sys.executable, "-I", *arguments],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage:" in result.stdout


@pytest.mark.parametrize("script", ["m4-loop", "m4-loop-train", "m4-loop-predict"])
def test_installed_console_entrypoints(script, tmp_path):
    executable = Path(sys.executable).parent / script
    assert executable.is_file(), "Install this project before running its test suite"
    result = subprocess.run(
        [str(executable), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage:" in result.stdout


@pytest.mark.parametrize(
    ("module", "forbidden"),
    [
        (
            "m4_loop.inference.predictor",
            ["m4_loop.training", "m4_loop.train", "m4_loop.cli"],
        ),
        (
            "m4_loop.models",
            ["m4_loop.training", "m4_loop.inference", "m4_loop.data", "m4_loop.cli"],
        ),
        ("m4_loop.training.trainer", ["m4_loop.cli", "m4_loop.inference"]),
    ],
)
def test_application_dependency_direction(module, forbidden, tmp_path):
    script = f"""
import importlib
import importlib.abc
import sys
forbidden = {forbidden!r}
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == prefix or fullname.startswith(prefix + '.') for prefix in forbidden):
            raise AssertionError('Disallowed dependency: ' + fullname)
sys.meta_path.insert(0, Guard())
importlib.import_module({module!r})
assert not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules for prefix in forbidden)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_public_model_imports_share_one_implementation():
    from m4_loop import M4LoopModel
    from m4_loop.model import M4LoopModel as CompatibilityModel
    from m4_loop.models import M4LoopModel as CanonicalModel

    assert M4LoopModel is CompatibilityModel is CanonicalModel

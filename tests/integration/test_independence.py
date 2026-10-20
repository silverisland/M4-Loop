"""Run a copied package without allowing imports from any legacy project."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import m4_loop


def test_copied_package_forward_backward_without_legacy_imports(config, tmp_path):
    source = Path(m4_loop.__file__).resolve().parent
    shutil.copytree(source, tmp_path / "m4_loop", ignore=shutil.ignore_patterns("__pycache__"))
    isolated_script = r"""
import importlib.abc
import json
import sys
from pathlib import Path

class RejectLegacyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        forbidden = {'m4', 'm1', 'forecast_backbones', 'NextGenPVForecast', 'pipelines', 'interfaces'}
        if fullname.split('.')[0] in forbidden:
            raise AssertionError(f'Independent package imported legacy module: {fullname}')
        return None

sys.meta_path.insert(0, RejectLegacyImports())
sys.path.insert(0, str(Path.cwd()))
import torch
from m4_loop.config import load_config
from m4_loop.models import M4LoopModel
from m4_loop.training.trainer import train
from m4_loop.inference.predictor import predict
import pandas as pd

torch.set_num_threads(1)
model = M4LoopModel(load_config(json.loads(sys.argv[1])))
batch = {
    'general_ghi': torch.rand(2, 8),
    'general_temp': torch.rand(2, 8),
    'general_ghi_predict': torch.rand(2, 4),
    'general_temp_predict': torch.rand(2, 4),
    'observe_power': torch.rand(2, 8),
    'chronos': torch.rand(2, 4),
    'cap_power_on': torch.tensor([10., 20.]),
}
output = model(batch, labels={'observe_power_future': torch.rand(2, 4)})
assert output.predictions.shape == (2, 4)
assert torch.isfinite(output.loss)
output.loss.backward()
assert model.encoder.blocks[0].self_attn.in_proj_weight.grad is not None
cfg = model.config
cfg['optimization'].update(train_epochs=1, batch_size=2, device='cpu')
records = []
for index in range(2):
    row = {name: value[index].tolist() for name, value in batch.items()}
    row['observe_power_future'] = [1., 2., 3., 4.]
    records.append(row)
frame = pd.DataFrame(records)
frame.to_parquet('windows.parquet', index=False)
run = train(cfg, 'windows.parquet', 'windows.parquet', 'run', max_steps=1)
predictions = predict(run['best_checkpoint'], 'windows.parquet', 'predictions.parquet', device='cpu')
assert len(predictions) == 2
assert Path('predictions.parquet').is_file()
for name, module in tuple(sys.modules.items()):
    path = getattr(module, '__file__', None)
    if path:
        assert 'NextGenPVForecast' not in str(path), path
        if name == 'm4_loop' or name.startswith('m4_loop.'):
            assert Path(path).resolve().is_relative_to(Path.cwd().resolve()), path
print('isolated forward and backward passed')
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", isolated_script, json.dumps(config)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "isolated forward and backward passed" in completed.stdout

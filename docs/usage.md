# 在 NextGen 中配置 M4-Loop

## 复制

将本仓库的 `forecast_backbones/m4_loop/` 整个目录复制到工作环境的 `NextGen/forecast_backbones/m4_loop/`，保留这个小写目录名。无需复制整个仓库，也无需执行 `pip install -e .`。以下代码和命令均在 NextGen 根目录运行。

默认配置是 `forecast_backbones/m4_loop/config/configs.yaml`；可以复制为实验配置后传给 `--config`。模块入口会加载默认值，再覆盖实验 YAML 中明确给出的字段。

## 数据列

每行是一条预测窗口。历史 672 点、未来 192 点为默认长度，所有数值输入必须有限、等长。

| 配置项 | 默认列名 | 每行内容 |
| --- | --- | --- |
| `conv_cols` | `general_ghi`, `general_temp` | 两路气象历史，各 input_len 点 |
| `conv_predict_cols` | `general_ghi_predict`, `general_temp_predict` | 两路气象未来预报，各 pred_len 点 |
| `pv_col` | `observe_power` | 历史功率，input_len 点 |
| `tsfm_col` | `chronos` | 外部功率先验，pred_len 点 |
| `target_col` | `observe_power_future` | 真实标签，pred_len 点，仅训练/验证需要 |
| `capacity_col` | `cap_power_on` | 固定装机容量，正数，仅训练/验证需要 |
| `timestamp_col` | `timestamp_win` | 窗口时间标识 |
| `entity_col` | `station` | 电站标识 |

容量与功率须使用一致单位。容量支持 batch 中的标量、`[B]` 或 `[B,1]`，不接受随时间变化的容量序列。标签中部分 NaN/Inf 会掩码；训练数据中整条标签均无效的行会被过滤并提示。没有有效标签的数据集会报错。

`dataset.ts_col/entity_cols/feature_cols/label_cols` 是 NextGen 通用数据加载字段，默认 YAML 同时列出。若修改列名，完整配置中的对应列表也需同步修改；若实验 YAML 只覆盖 `timestamp_col` 等模型字段，模块加载器会自动生成通用字段。两组字段显式冲突会报错，避免读错特征。预测通过 `feature_columns(cfg, supervised=False)` 排除标签与容量。

## 模型与训练设置

- `seq_len` 必须等于 `input_len + pred_len`，Patch 必须完整覆盖窗口。
- `e_layers` 为不同参数的层数，`loop_steps` 为共享循环次数。默认 2×2；消融配置提供 1×4 与 4×1。
- `activation` 支持 ReLU/GELU，不区分大小写；`revin` 支持 true/false 或 1/0。
- `batch_size` 和 `eval_batch_size` 是每设备 batch 大小；梯度累积由 `gradient_accumulation_steps` 控制。
- `device: auto` 由 Trainer 选择可用设备；CPU 可显式设置 `cpu`。`fp16` 仅在 CUDA 使用。选择 GPU 时优先设置 `CUDA_VISIBLE_DEVICES` 并使用 `device: cuda`。
- 每轮验证并保存，按 `eval_loss` 选优，`patience` 控制早停。损失为容量归一化误差，HF Trainer 按样本聚合验证 batch 损失，不是按全数据集有效时间点重新加权的指标。

```bash
python -m forecast_backbones.m4_loop.train \
  --config forecast_backbones/m4_loop/config/configs.yaml \
  --train /data/train.parquet --valid /data/valid.parquet \
  --test /data/test.parquet --out runs/m4_loop/exp1
```

输出包括 HF checkpoint、训练/验证指标、`result_val.parquet`，以及可选 `result_test.parquet`；最终加载目录为 `model_result/`，其中包含 `model.safetensors`、`model_config.yaml`。

## 使用原 train_shell 入口（可选）

现有 NextGen `train_shell/train_script.py` 硬编码模型名称，单复制目录不会增加其训练分支。本仓库提供 `integration/nextgen_train_script.patch`，增加 m4_loop 的默认配置、训练参数与专用训练路由。它基于本次检查的 NextGen 文件生成；其他版本请按补丁内容合并。

在 NextGen 根目录检查并应用：

```bash
git apply --check /path/to/M4-Loop/integration/nextgen_train_script.patch
git apply /path/to/M4-Loop/integration/nextgen_train_script.patch
python train_shell/train_script.py \
  --model_name m4_loop \
  --model_config_path forecast_backbones/m4_loop/config/configs.yaml \
  --dataset_path /data/split \
  --run_path runs/m4_loop/exp1
```

`/data/split` 下需要 `train.parquet`、`valid.parquet`、`test.parquet`。补丁让该入口调用模块的训练函数，以兼容当前公共训练工具的缺失依赖，不修改原 M4 的代码或行为。只使用模块训练入口时不必应用此补丁。

## 加载与统一推理

```python
from forecast_backbones.m4_loop.predict import load_model
from forecast_backbones.m4_loop.transformer_config import feature_columns
from pipelines.pipeline_wrapper_inference import unified_inference

model = load_model("runs/m4_loop/exp1/model_result", device="auto")
ds = model.config["dataset"]
result = unified_inference(
    model,
    "/data/future.parquet",
    timestamp_cols=[ds["timestamp_col"]],
    entity_cols=[ds["timestamp_col"], ds["entity_col"]],
    feature_cols=feature_columns(model.config, supervised=False),
)
```

统一推理结果列名为 `TransformersAdaptedM4LoopModel`。也可以用模块内预测命令生成 `timestamp_win / station / observe_power_predict` 三列：

```bash
python -m forecast_backbones.m4_loop.predict \
  --model-dir runs/m4_loop/exp1/model_result \
  --data /data/future.parquet --out runs/m4_loop/exp1/predictions.parquet
```

两种入口均保持输入行序，预测是原功率量纲，不做非负或容量截断。注册器方式同样支持 `build_model("m4_loop", cfg)` 再严格加载 safetensors；评估通过 `build_evaluator("m4_loop")` 获取函数，返回标准 M4 评估列。

## 人工数据检查

本仓库 `examples/make_demo_data.py` 生成 16 点历史/8 点未来的人工数据，不代表真实精度。在本仓库运行生成脚本后，在 NextGen 中使用 `config/smoke.yaml` 和生成的数据，传 `--max-steps 2` 即可短训检查。生产数据继续使用默认窗口配置。

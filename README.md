# M4-Loop

按 NextGen 中 M4 的接入方式组织的独立预测骨干。把 **`forecast_backbones/m4_loop/` 整个文件夹**复制到工作环境的 `NextGen/forecast_backbones/` 即可配置使用，无需安装本仓库为 Python 包。

模型、循环 Transformer、归一化和损失函数都在这个文件夹内实现，**不导入、不调用 `forecast_backbones.m4` 或其他骨干**。沿用 NextGen 的公共 `interfaces.modeloutput`、`pipelines.transformer_dataset` 接口。

## 要复制的结构

```text
forecast_backbones/m4_loop/
├── transformer_models.py       # TransformersAdaptedM4LoopModel：模型注册入口
├── transformer_config.py       # 配置加载和字段校验
├── transformer_trainer.py      # M4Loop_build_training_args、evaluate_model
├── train.py                    # NextGen 环境内的训练入口
├── predict.py                  # 标准权重加载、无标签预测入口
├── layers/
│   ├── loop_transformer.py     # 本地 RevIN、共享循环编码器、预测头
│   └── losses.py               # 本地容量归一化损失
└── config/
    ├── configs.yaml            # 默认配置，672 点历史 → 192 点未来
    ├── shared_1x4.yaml         # 1 个独立层 × 4 次循环
    ├── untied_4.yaml           # 4 个独立层 × 1 次循环
    └── smoke.yaml              # 人工数据小窗口检查
```

其余 `tests/`、`docs/`、`examples/` 是本仓库的验证与说明，不必复制进骨干目录。

## 在工作环境使用

先把上面的 `m4_loop` 文件夹放入 NextGen；以下命令都从 **NextGen 根目录**运行：

```bash
python -m forecast_backbones.m4_loop.train \
  --config forecast_backbones/m4_loop/config/configs.yaml \
  --train /path/to/train.parquet \
  --valid /path/to/valid.parquet \
  --test /path/to/test.parquet \
  --out runs/m4_loop/exp1

python -m forecast_backbones.m4_loop.predict \
  --model-dir runs/m4_loop/exp1/model_result \
  --data /path/to/future.parquet \
  --out runs/m4_loop/exp1/predictions.parquet
```

`--test` 可省略。训练使用 HF Trainer，结果保存为 `model_result/model.safetensors` 和 `model_result/model_config.yaml`。预测输入无需真实未来功率，也无需容量列。

NextGen 的模型注册器可直接使用 `build_model("m4_loop", config)`；评估入口可用 `build_evaluator("m4_loop")`。原 `train_shell/train_script.py` 采用硬编码分支，若要继续用它的 `--model_name m4_loop` 方式，需要应用 [可选接入补丁](integration/nextgen_train_script.patch)，详见 [使用说明](docs/usage.md)。仅复制目录时，请使用上面的模块训练命令。

## 配置原则

保留 M4 的 `dataset / inference / model / optimization` 四段。默认 `model.e_layers: 2`、`model.loop_steps: 2`：2 个独立 Transformer 层重复执行 2 次，总计 4 次层计算。循环复用同一组参数，梯度穿过所有循环。

默认输入为两路气象历史/未来预报、历史功率、`chronos` 未来功率先验；标签为 `observe_power_future`，训练容量为 `cap_power_on`。列名和窗口长度均从配置读取。损失为容量归一化 MAE，可切换 MSE / Huber；它不包含旧 M4 的 NW 辅助损失。

- [数据、配置、训练与推理细节](docs/usage.md)
- [模块职责及 NextGen 接口](docs/architecture.md)
- [验证方式和范围](docs/validation.md)

Python ≥3.10。第三方依赖列在 `requirements.txt`；工作环境已有依赖时无需重复安装。这里的“独立”指模型代码与原 M4 独立；运行依赖 NextGen 公共接口和第三方库。

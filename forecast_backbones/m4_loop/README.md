# m4_loop：NextGen 预测骨干

将本文件夹直接放到 NextGen 的 `forecast_backbones/m4_loop/`，从 NextGen 根目录运行：

```bash
python -m forecast_backbones.m4_loop.train --config forecast_backbones/m4_loop/config/configs.yaml --train /data/train.parquet --valid /data/valid.parquet --out runs/m4_loop/exp1
python -m forecast_backbones.m4_loop.predict --model-dir runs/m4_loop/exp1/model_result --data /data/future.parquet --out runs/m4_loop/exp1/predictions.parquet
```

注册名：`m4_loop`；模型类：`TransformersAdaptedM4LoopModel`。默认 `e_layers=2`、`loop_steps=2`，表示两个独立层循环两次。完整字段见 `config/configs.yaml`，以这个配置为起点调整，不能直接套用所有旧 M4 参数。

模型实现不调用原 M4。共享 NextGen 的 `interfaces.modeloutput` 与 `pipelines.transformer_dataset`；训练使用 HF Trainer，保存 `model_result/model.safetensors` 与 `model_config.yaml`。

现有公共 `train_shell/train_script.py` 对模型名有硬编码；上面的目录内入口无需改它。若要用 `--model_name m4_loop`，参见仓库的可选接入补丁与[完整说明](https://github.com/silverisland/M4-Loop/blob/main/docs/usage.md)。

# 目录与接口

交付单元是 `forecast_backbones/m4_loop/`。它与 NextGen 的 `m4/` 平级，自己的模块全部使用相对导入。仓库不提供另一套顶层 `m4_loop` 安装包，也不复制 NextGen 公共框架。

| 文件 | 责任 | NextGen 约定 |
| --- | --- | --- |
| `transformer_models.py` | 组装输入、模型前向与损失 | 本文件定义 `TransformersAdaptedM4LoopModel(nn.Module)`；构造参数为完整配置 |
| `transformer_config.py` | 本地 YAML 默认值、合并、校验 | `dataset / inference / model / optimization` |
| `transformer_trainer.py` | TrainingArguments、按行序评估 | `M4Loop_build_training_args`、`evaluate_model` |
| `layers/loop_transformer.py` | RevIN、Pre-LN Transformer、预测头 | 完全本地实现 |
| `layers/losses.py` | 容量归一化 MAE/MSE/Huber | 仅显式标签参与损失 |
| `train.py` | 构建 UnifiedDataset、调用 HF Trainer、保存结果 | 标准 `model_result` 目录 |
| `predict.py` | 严格加载权重并预测 | 输入不需要标签；输出标准预测表 |

## 数据流

```text
历史气象(2路) + 历史功率       [B,3,input_len]
未来气象(2路) + 未来功率先验   [B,3,pred_len]
                 ↓ 时间轴拼接
          全窗口 RevIN → Patch 投影
                 ↓
       合并通道与 Patch 形成 token 序列
                 ↓
       e_layers 个独立的 Pre-LN 层
          重复 loop_steps 次，共享参数
                 ↓
       最终 LayerNorm → 展平预测头
                 ↓
       功率通道反归一化 → [B,pred_len]
```

默认 3 通道、864 点总窗口、96 点 Patch，形成 27 个 token。输入中的 `chronos` 是外部已生成的未来预测特征；本项目不会调用 Chronos 模型。真实未来标签只经 `labels` 参数进入损失。

## 对接边界

`forward(batch_dict, labels=None)` 返回 NextGen 公共 `TransformersAdaptedModelOutput`：`loss` 为可选标量，`logits/predictions` 为 `[B,pred_len]`，`embeddings` 为 `[B,embedding_dim]`。数值特征会转换到模型自身所在设备，兼容统一推理入口直接传入 CPU batch 的行为。

模型注册器按文件本地类发现模型，评估注册器按函数名发现 `evaluate_model`。模型代码不反向导入注册器，也不导入旧 M4、其他骨干或旧训练工具。

共用的 NextGen 代码只有 `interfaces.modeloutput` 和 `pipelines.transformer_dataset`；使用宿主注册器及统一推理函数时，由宿主进行调度。

## 与之前版本的区别

此前的 `src/m4_loop`、console 命令和自定义 `best.pt` 训练产物已退出当前目录方案。当前训练保存标准 safetensors 加 YAML；不把旧 `best.pt` 直接当作新模型目录加载。核心网络参数名与计算仍沿用原 M4-Loop 实现，配置中的 `block_layers` 改为 M4 风格的 `e_layers`。

原 M4 的所有配置字段并不自动适用于新结构。当前不支持 NW 辅助损失、residual attention score 累积、affine RevIN、padding_patch 等旧选项；未知配置项明确报错。应以这里的 `config/configs.yaml` 为起点修改。

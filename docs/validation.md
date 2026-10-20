# 验证范围

验证针对“只复制 m4_loop 到 NextGen”这一部署方式。需要真实 NextGen 的公共接口源文件；不使用伪造框架来代替接入验证。

在本仓库的测试环境安装 `requirements-dev.txt` 后运行：

```bash
NEXTGEN_ROOT=/path/to/NextGen python -m pytest -q
python -m ruff check forecast_backbones tests examples
```

`NEXTGEN_ROOT` 只用于测试定位宿主文件，生产复制部署不需要这个变量。

## 覆盖内容

- 共享参数循环的调用次数、显式展开与梯度一致性、Pre-LN 残差。
- 输入尺寸、标签隔离、容量广播、无效标签掩码和有限梯度。
- M4 风格配置、通用数据字段一致性、CPU 训练参数和标签过滤。
- 把骨干文件夹及真实 NextGen 的注册器、数据集、统一推理和 ModelOutput 复制到临时环境。该环境没有原 M4 或其他骨干，并且导入拦截器禁止加载它们。
- 实际 `build_model` / `build_evaluator`、两步 HF 训练、safetensors 与 YAML 保存、严格重载后预测一致、无标签/无容量的统一推理、最后不足一批数据的行序与结果。
- 训练和预测模块的命令入口。
- 可选补丁应用到真实 `train_shell/train_script.py` 后，实际运行 `--model_name m4_loop`，验证默认配置发现、训练和标准产物保存。

这些检查验证工程接入与数值行为，不代表真实电站精度提升。CUDA、多卡训练和生产规模吞吐需要在工作环境验证；当前本地训练验收使用 CPU。

## 本次执行结果

本次本地验收使用 Python 3.12、PyTorch 2.5.1 和 Transformers 4.40.1：

- 全部 29 项测试通过，包括真实复制接入和补丁后的公共训练脚本。
- Ruff 检查、格式检查和 Git diff 空白检查通过。
- 默认 672/192 窗口前向通过，输出 `[2,192]`，参数量 3,477,376。
- 另将调整前保存的 M4-Loop 参数严格载入新类；固定输入的预测、embedding、损失及所有梯度逐项完全一致。
- 原 M4 目录的 9 个文件与此前保存的 SHA-256 一致。

以上一致性检查使用本地保存的先前 M4-Loop 测试权重，没有调用原 M4 模型。未向仓库提交测试权重或业务数据。

# M4-Loop

独立的光伏功率预测项目：用历史功率、历史/未来气象和外部功率预测构造联合 Patch Transformer，
默认两个共享 block 循环两次。自带数据校验、训练、验证选优、权重保存和预测入口，
不依赖原 M4 或 NextGenPVForecast。

## 项目结构

```text
M4-Loop/
├── pyproject.toml            # 安装、命令入口、测试和代码风格配置
├── src/m4_loop/
│   ├── models/               # 网络、基础层、损失、输出类型
│   ├── data/                 # Parquet 窗口校验、Dataset、batch 整理
│   ├── training/             # 训练循环、验证指标、最佳权重选点
│   ├── inference/            # 批量预测及结果导出
│   ├── cli/                  # 命令解析与调度
│   ├── resources/            # 随安装包发布的默认 YAML
│   ├── config.py             # 配置加载、合并和校验
│   ├── checkpoint.py         # 版本化权重保存与严格加载
│   └── runtime.py            # 设备、随机种子和 Tensor 转移
├── configs/                  # 训练/消融/小样本配置覆盖项
├── examples/                 # 人工数据生成示例
├── tests/unit/               # 模型、损失和配置测试
├── tests/integration/        # 训练、安装包、命令行和独立性测试
└── docs/                     # 工程架构、详细用法和验证记录
```

## 安装

Python ≥3.10，推荐 3.12。在仓库根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

项目采用 `src` 布局，先安装再运行。部署时可用 `python -m pip install .` 安装普通包。

## 训练和预测

```bash
m4-loop train --train /path/to/train.parquet --valid /path/to/valid.parquet --out runs/exp1
m4-loop predict --checkpoint runs/exp1/best.pt --data /path/to/test.parquet --out runs/exp1/predictions.parquet
```

`python -m m4_loop train ...` / `predict ...` 与上述命令等价。
默认输入 672 点历史、192 点未来预报，输出 192 点功率。输入字段与窗口必须准确匹配；
真实数据请按 [数据契约与配置说明](docs/usage.md) 提供。

软件版本 `0.2.0` 调整工程组织，模型计算与 v1 checkpoint 格式保持兼容。
旧 `python -m m4_loop.train`、`python -m m4_loop.predict` 和两个旧 console 命令仍可使用。

## 验证

```bash
python -m pytest -q
python -m ruff check src tests examples
python -m ruff format --check src tests examples
```

测试使用安装后的包；主测试环境不通过 `PYTHONPATH` 或 `sys.path` 注入工作区源码。
真实电站预测精度需要独立实验验证。

- [工程架构与模块依赖](docs/architecture.md)
- [模型、数据、配置、人工数据示例及 Python API](docs/usage.md)
- [验证记录与适用范围](docs/validation.md)

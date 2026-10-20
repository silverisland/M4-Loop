# M4-Loop

独立的光伏功率预测项目。`m4_loop` 自带模型、配置、数据读取、损失、训练、
checkpoint 和预测入口；不导入原 M4，也不依赖 `NextGenPVForecast`、
`forecast_backbones`、`pipelines`、`interfaces` 或其他模型目录。
整个 `M4-Loop` 文件夹可以单独复制到其他机器。

## 模型约定

默认结构：

```text
历史气象 2×672 + 未来气象 2×192 ─┐
历史功率 672  + Chronos 192 ────┴─ [B,3,864]
              ↓ 每个样本、每个通道的全窗 RevIN
              ↓ patch_len=stride=96，Linear(96,256)
              ↓ 变量×时间 patch 展平 + 可学习位置嵌入
                       [B,27,256]
              ↓ A → B → A → B（两个共享 block，循环两次）
              ↓ 最终 LayerNorm
              ↓ 按 [B,3,256,9] 展平
              ↓ 6912 → 256 → 256 → 64 → 192
              ↓ 功率 RevIN 还原
                       [B,192]
```

具体处理方式：

- **真正共享权重：** `block_layers=2`，`loop_steps=2`。循环次数变化不会增加
  block 参数。所有循环都参与反向传播，没有 `detach` 或截断梯度。
- **标准 Pre-LN：** 每个 block 使用 `x + Attention(LN(x))`，
  再使用 `x + FFN(LN(x))`；编码器末尾额外做一次 LayerNorm。不使用 BatchNorm。
- **attention：** 三个通道的所有 patch 做双向联合注意力。每层重新计算
  attention 分数，不跨层或跨轮次累加 softmax 前的分数。
- **固定循环：** 训练和预测使用 checkpoint 保存的相同循环次数。
  不使用动态退出、循环编号嵌入或输入重复注入。Dropout 只在训练时启用。
- **RevIN：** 保留历史与未来已知输入一起计算统计量的方式；均值和标准差只存在于
  当前 forward 的局部变量中，统计量不反传，不在不同样本批次之间缓存。
  默认不使用可学习仿射参数。未来真实标签不参与归一化。
- **输出：** 一次输出整段功率；`logits` 与 `predictions` 均为原功率单位。
  `embeddings` 是预测头第一层线性映射的结果。默认不做负值截断或容量封顶。
- **损失：** 固定容量归一化 MAE；可配置 MSE 或 Huber（归一化误差的阈值为 1）。
  不调用原 M4 的外部南网损失，也不从标签反推容量。验证按相同损失选择最佳权重。

因此，本项目与原 M4 的差异包括循环权重共享、归一化、attention 分数处理和损失。
若要判断循环本身的作用，应使用本项目中的四层非共享配置作对照：

| 配置 | 执行顺序 | 独立 block | 执行次数 |
| --- | --- | ---: | ---: |
| 默认 | A → B → A → B | 2 | 4 |
| `configs/untied_4.yaml` | A → B → C → D | 4 | 4 |
| `configs/shared_1x4.yaml` | A → A → A → A | 1 | 4 |

三者都使用同一套 Pre-LN、预测头和损失。减少参数不等于同比减少计算量或训练激活显存。
修改循环配置后应重新训练；原 M4 checkpoint 不能直接加载。

## 安装

推荐 Python 3.12；最低 Python 3.10。在本目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

仅依赖 PyTorch、NumPy、Pandas、PyArrow 和 PyYAML，不需要 Transformers、
Hugging Face 数据集服务或下载预训练权重。Chronos 预测是数据表中已有的输入列；
本项目不运行 Chronos 模型。

## 数据契约

输入是预先切分好的 Parquet 文件，每行代表一个起报窗口：

| 默认列名 | 每行内容 |
| --- | --- |
| `general_ghi`、`general_temp` | 各 672 个历史值 |
| `general_ghi_predict`、`general_temp_predict` | 各 192 个未来天气预报值 |
| `observe_power` | 672 个历史功率值 |
| `chronos` | 192 个外部功率预测值 |
| `cap_power_on` | 一个正且有限的固定容量，与功率同单位；监督训练必须提供 |
| `observe_power_future` | 192 个真实未来功率值；训练、验证必须提供 |
| `timestamp_win`、`station` | 可选元数据，预测结果原样保留 |

`capacity_col` 默认沿用 M4 的列名 `cap_power_on`，**实际必须提供该窗口采用的固定
归一化容量**。如果该列在你的数据中是时变开机容量，应改用固定装机容量列，例如
`cap_mw`。不接受每个时刻一份容量，也不会用未来标签估计容量。

所有模型输入都必须有限且长度准确；NaN/Inf 输入会报错。标签中的 NaN/Inf 会被掩码，
损失只计算有效点；训练入口过滤整行标签均无效的样本并记录数量。
原始长度不一致会报错，不会自动截断历史或未来序列。

无标签预测不需要容量。模型只在显式传入 `labels` 时计算损失；
即使 `batch_dict` 含有真实标签列，也不会把它当作输入或自动计算监督损失。

训练和验证由调用方按时间或站点提前切分。存在重叠窗口时，应在切分边界按业务设置
隔离区；入口不会随机划分数据，也不会替调用方证明切分不存在泄漏。
未来天气与外部功率预测必须是在起报时刻可获得的版本。

## 配置

默认配置随包放在 `m4_loop/config/default.yaml`。自定义 YAML 只需填写覆盖项：

```yaml
dataset:
  tsfm_col: observe_power_chronos_pred
  capacity_col: cap_mw
optimization:
  device: cuda
  batch_size: 64
```

`input_len + pred_len` 必须等于 `seq_len`；patch 必须完整覆盖整个输入窗口。
默认支持 `mae`、`mse`、`huber` 三种 `model.loss_type`。
未知配置键会报错，旧 M4 的 `e_layers`、`pre_norm`、`res_attention` 等键不能
直接沿用。可用 `block_layers` 和 `loop_steps` 明确控制结构。

## 训练和预测

```bash
python -m m4_loop.train \
  --train /path/to/train.parquet \
  --valid /path/to/valid.parquet \
  --out runs/loop_2x2
```

使用覆盖配置时添加 `--config /path/to/overrides.yaml`。
使用 `--max-steps 2` 可以做短训练检查。训练前设置随机种子，使用 AdamW 和梯度裁剪；
每轮验证，按验证损失保存最佳权重，`patience` 按验证次数计。
训练入口使用单进程、单设备、FP32；`device` 支持 `auto`、`cpu`、`cuda`、`mps`。

```bash
python -m m4_loop.predict \
  --checkpoint runs/loop_2x2/best.pt \
  --data /path/to/test.parquet \
  --out runs/loop_2x2/predictions.parquet
```

checkpoint 同时保存完整配置与权重，预测以其中的配置构造模型，严格检查权重匹配。
预测输出列 `observe_power_predict` 为整段功率序列；标签和站点、时间元数据可随结果保留。
预测默认自动选择当前机器可用设备；可添加 `--device cpu` 显式指定。
训练输出目录必须为空，以免覆盖已有实验。

每次训练输出：

- `best.pt`：验证损失最低的模型，包含架构版本、完整配置和权重。
- `model_config.yaml`：本次实际使用的完整配置。
- `history.json`：各轮训练损失、验证指标及是否刷新最佳。
- `metrics.json`：最佳轮次、指标、总步数、停止原因和设备。
- `data_stats.json`：训练/验证行数、过滤数量和有效标签点数。

在 Python 中加载：

```python
from m4_loop.checkpoint import load_checkpoint

model, metadata = load_checkpoint("runs/loop_2x2/best.pt", device="cpu")
# batch_dict 的序列列为 [B,L] Tensor；load_checkpoint 已将模型设为 eval。
output = model(batch_dict)
power = output.predictions  # [B,192]，原功率单位
```

## 人工数据验证

下面的示例仅验证安装、训练、保存和重载链路，不用于判断真实预测精度：

```bash
python examples/make_demo_data.py --out demo_data
python -m m4_loop.train \
  --config examples/smoke.yaml \
  --train demo_data/train.parquet \
  --valid demo_data/valid.parquet \
  --out runs/smoke \
  --max-steps 2
python -m m4_loop.predict \
  --checkpoint runs/smoke/best.pt \
  --data demo_data/test.parquet \
  --out runs/smoke/predictions.parquet
python -m pytest -q
```

测试覆盖参数共享、循环梯度、标准残差路径、容量广播、缺失标签、无标签泄漏、
checkpoint 重载，以及将包复制到隔离目录、禁止导入原仓库后的运行。

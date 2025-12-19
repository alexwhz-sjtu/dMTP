# MTP Head Training Pipeline

基于 `pipeline.md` 的扩散头训练完整实现。

## 📁 文件结构

```
train_eaglestyle/
├── README.md                           # 本文件
├── TRAINING_GUIDE.md                   # 详细训练指南
├── pipeline.md                         # 原始训练思路文档
├── quick_start.sh                      # 快速启动脚本
├── config_examples.py                  # 配置示例
│
├── data_collection.py                  # Step 1: 数据收集脚本
├── mtp_dataset.py                      # Step 2: 数据加载器
├── train_mtp_head.py                   # Step 3: 主训练脚本
├── test_pipeline.py                    # 流程测试脚本
│
├── mtphead_trainer.py                  # 自定义 Trainer（离散扩散训练）
└── schedulers/                         # Alpha 调度器
    ├── __init__.py
    ├── alpha.py                        # 线性 Alpha 调度器
    └── kappa.py                        # Kappa Alpha 调度器
```

## 🚀 快速开始

### 0. 环境准备

确保已安装必要的依赖：
```bash
pip install transformers torch tqdm
```

### 1. 数据收集 (Data Collection)

从基础模型推理过程中收集隐藏状态和 token：

```bash
python data_collection.py \
    --base_model_path Qwen/Qwen2-7B \
    --input_data_path /path/to/text/file.txt \
    --output_dir ./mtp_collected_data \
    --max_samples 50000 \
    --sample_size 512 \
    --stride 256
```

**输入格式**：文本文件，每行一个文档

**输出**：
```
./mtp_collected_data/collected_data_final/
├── hidden_states.pt       # [num_samples, seq_len, hidden_size]
├── tokens.pt              # [num_samples, seq_len]
├── input_ids.pt           # [num_samples, seq_len]
└── sample_ids.pt          # [num_samples]
```

### 2. 测试流程 (Optional)

验证数据加载和训练流程：

```bash
python test_pipeline.py \
    --data_dir ./mtp_collected_data/collected_data_final \
    --block_length 4
```

### 3. 训练 MTP 头

```bash
python train_mtp_head.py \
    --train_data_dir ./mtp_collected_data/collected_data_final \
    --output_dir ./mtp_checkpoint \
    --block_length 4 \
    --per_device_train_batch_size 32 \
    --num_train_epochs 5 \
    --learning_rate 2e-4 \
    --warmup_steps 1000 \
    --logging_steps 50 \
    --save_steps 500
```

## 📊 训练流程详解

### Step 1: 数据收集 (`data_collection.py`)

**目的**：从基础 LLM 的推理过程中收集条件隐藏状态和预测 token

**工作流**：
1. 加载基础模型和分词器
2. 对每个输入文本进行推理
3. 记录：
   - `h_l`: 每个位置的隐藏状态（base model 最后一层）
   - `t_l`: 每个位置的预测 token（greedy 解码）
4. 使用滑动窗口创建训练样本
5. 分块保存数据

**关键参数**：
- `sample_size`: 训练样本长度（推荐 512）
- `stride`: 滑动窗口步长（推荐 256）
- `max_length`: 单次推理的最大长度（默认 2048）

### Step 2: 数据加载 (`mtp_dataset.py`)

**目的**：将收集的数据转换为 MTP 头训练样本

**样本格式**：
- `in_hidden_states`: 条件隐藏状态 `h_l` [hidden_size]
- `input_ids`: 掩码 token 序列 [block_length]（全为 MASK token ID）
- `labels`: 目标 token `t_l, t_(l+1), ..., t_(l+L-1)` [block_length]
- `attention_mask`: 注意力掩码 [1 + block_length]

**数据流**：
```
原始隐藏状态和 token
    ↓
MTPHeadDataset（每个样本包含条件和目标）
    ↓
MTPHeadCollator（批处理）
    ↓
DataLoader（训练）
```

### Step 3: 模型训练 (`train_mtp_head.py`)

**目的**：使用离散扩散原理训练 MTP 头，预测多个 token

**训练流程**（在 `mtphead_trainer.py` 的 `compute_loss` 中实现）：

1. **时间采样**: 随机采样 $t \in [\epsilon, 1)$
2. **掩码概率计算**: $p_{mask} = 1 - \alpha(t)$，其中 $\alpha(t)$ 来自调度器
3. **随机掩码**: 独立地以概率 $p_{mask}$ 掩码化每个 token（已掩码的 input_ids）
4. **前向传播**: 将掩码 token 和条件隐藏状态输入 MTP 头
5. **损失计算**: 交叉熵损失，仅在掩码位置计算
6. **加权**: 按调度器权重加权（可选）
7. **反向传播**: 更新模型参数

**关键参数**：
- `block_length`: 预测的 token 数（L，推荐 4-8）
- `scheduler_type`: Alpha 调度器类型（'linear' 或 'kappa'）
- `time_epsilon`: 最小时间步，避免退化（推荐 0.01）
- `loss_weight_type`: 损失权重计算（'scheduler' 或 'ones'）

## ⚙️ 配置选项

### 快速测试配置
```bash
python train_mtp_head.py \
    --train_data_dir ./data \
    --output_dir ./test_ckpt \
    --per_device_train_batch_size 8 \
    --num_train_epochs 1 \
    --logging_steps 5
```

### 标准训练配置
```bash
python train_mtp_head.py \
    --train_data_dir ./data \
    --output_dir ./checkpoint \
    --block_length 4 \
    --per_device_train_batch_size 32 \
    --num_train_epochs 5 \
    --learning_rate 2e-4 \
    --warmup_steps 1000 \
    --scheduler_type linear
```

### 大规模训练配置
```bash
python -m torch.distributed.launch --nproc_per_node 8 train_mtp_head.py \
    --train_data_dir ./data \
    --output_dir ./large_ckpt \
    --block_length 8 \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 10
```

查看 `config_examples.py` 获取更多预设配置。

## 📝 文件详细说明

### data_collection.py
**类**：
- `DataCollector`: 从基础模型收集数据

**方法**：
- `collect_from_text()`: 从单个文本收集
- `collect_from_file()`: 从文件批量收集
- `_save_data()`: 保存为 PyTorch 张量

**输入**：
- 基础模型路径
- 文本文件（一行一个文档）

**输出**：
- PyTorch 张量（hidden_states, tokens, input_ids, sample_ids）

---

### mtp_dataset.py
**类**：
- `MTPHeadDataset`: PyTorch Dataset
- `MTPHeadCollator`: 批处理器

**函数**：
- `create_dataloaders()`: 创建训练和验证 DataLoader

**数据格式**：
```python
{
    "in_hidden_states": [hidden_size],
    "input_ids": [block_length],
    "labels": [block_length],
    "attention_mask": [1 + block_length],
}
```

---

### train_mtp_head.py
**类**：
- `ModelArguments`: 模型配置参数
- `DataArguments`: 数据配置参数
- `TrainingExtraArguments`: 额外训练参数
- `LoggingCallback`: 日志回调

**函数**：
- `setup_model()`: 初始化模型
- `main()`: 训练入口

**输出**：
- 训练好的模型（HuggingFace 格式）
- 训练信息 JSON 文件
- 检查点文件

---

### test_pipeline.py
**测试函数**：
- `test_dataset_loading()`: 验证数据加载
- `test_model_initialization()`: 验证模型初始化
- `test_forward_pass()`: 验证前向传播
- `test_training_step()`: 验证训练步骤

---

### mtphead_trainer.py
**继承自 HuggingFace Trainer**

**核心方法**：
- `compute_loss()`: 实现离散扩散损失计算
  - 时间采样
  - 随机掩码
  - 前向传播
  - 加权交叉熵

---

### schedulers/
**Alpha 调度器**：决定时间步 $t$ 对应的掩码率 $\alpha(t)$

- `LinearAlphaScheduler`: 线性衰减
- `KappaAlphaScheduler`: 基于 kappa 参数的衰减

## 🔍 常见问题

### Q: 数据收集太慢怎么办？
**A**：
- 减小 `max_samples`
- 增加 `stride`（减少每个文本的样本数）
- 使用更小的 `max_length`
- 使用多进程处理（需要修改脚本）

### Q: 训练显存不足？
**A**：
- 减小 `per_device_train_batch_size`
- 增加 `gradient_accumulation_steps`
- 使用 DeepSpeed Zero-2 或 Zero-3
- 减小 `block_length`

### Q: 损失不下降？
**A**：
- 检查数据加载是否正确（验证 shape 和值）
- 尝试增加 `warmup_steps`
- 尝试不同的 `learning_rate`
- 检查 `time_epsilon` 是否合理

### Q: 如何使用多 GPU？
**A**：
```bash
python -m torch.distributed.launch --nproc_per_node 4 train_mtp_head.py ...
```

或使用 DeepSpeed：
```bash
deepspeed train_mtp_head.py --deepspeed ds_config.json ...
```

## 📚 参考文献

1. **Simple and Effective Masked Diffusion Language Models**
   - https://arxiv.org/abs/2406.07524
   - 离散扩散训练的理论基础

2. **Large Language Diffusion Models**
   - https://arxiv.org/abs/2502.09992
   - 扩散在 LLM 中的应用

## 💾 检查清单

- [ ] 准备输入文本文件
- [ ] 验证基础模型可正常加载
- [ ] 运行数据收集脚本
- [ ] 验证收集的数据形状和大小
- [ ] 运行测试脚本验证流程
- [ ] 配置训练参数
- [ ] 运行训练脚本
- [ ] 监控损失曲线
- [ ] 验证检查点文件
- [ ] 评估模型性能

## 🤝 贡献

欢迎提出问题和改进建议！

## 📄 许可证

参考基础模型和相关论文的许可证。

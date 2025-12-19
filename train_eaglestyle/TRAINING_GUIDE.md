# MTP Head Training Guide

本文档详细说明如何按照 `pipeline.md` 中的训练思路，完成扩散头的训练。

## 训练流程概览

根据 `pipeline.md`，MTP 头的训练分为三个步骤：

### Step 1: 数据收集 (Data Collection)
从基础 LLM 推理的每个位置收集隐藏状态和输出 token。

```bash
python data_collection.py \
    --base_model_path /share/public/public_models/Qwen2.5-7B-Instruct \
    --input_data_path /share/wanghanzhen/MTP/dMTP/dataset/LongWriter/evaluation/longbench_write_en.jsonl \
    --output_dir /share/wanghanzhen/MTP/dMTP/training_data/ \
    --max_samples 8192 \
    --sample_size 512 \
    --stride 512 \
    --max_new_tokens 8192
```

**参数说明：**
- `base_model_path`: 基础大模型路径（如 Qwen/Qwen2-7B）
- `input_data_path`: 输入文本文件路径，每行一个文档
- `output_dir`: 输出目录，保存收集的数据
- `max_samples`: 最大样本数（可选）
- `sample_size`: 每个训练样本的长度（默认512）
- `stride`: 滑动窗口的步长（默认256）

**输出格式：**
```
/path/to/collected/data/
├── collected_data_final/
│   ├── hidden_states.pt      # [num_samples, seq_len, hidden_size]
│   ├── tokens.pt             # [num_samples, seq_len]
│   ├── input_ids.pt          # [num_samples, seq_len]
│   └── sample_ids.pt         # [num_samples]
```

### Step 2: 数据加载 (Data Loading)
`mtp_dataset.py` 提供了数据集加载器。

数据加载的关键点：
- 对于每个样本位置 `l`，我们用位置 `l` 的隐藏状态 `h_l` 作为条件
- 输入到 MTP 头的是 `L` 长的 mask token 序列
- 标签是原始序列中的后续 `L` 个 token: `t_l, t_(l+1), ..., t_(l+L-1)`

```python
from mtp_dataset import create_dataloaders

train_dl, val_dl, dataset = create_dataloaders(
    data_dir="/path/to/collected/data/collected_data_final",
    block_length=4,          # L: 预测的 token 数
    batch_size=32,
    train_ratio=0.9,
)

# 查看单个样本
sample = dataset[0]
print(sample.keys())  # 'in_hidden_states', 'input_ids', 'labels', 'attention_mask'
```

### Step 3: 模型训练 (Model Training)
使用 `train_mtp_head.py` 来训练 MTP 头。

#### 3.1 基础训练
```bash
torchrun --nproc_per_node=8 /share/wanghanzhen/MTP/dMTP/train_eaglestyle/train_mtp_head.py \
    --model_name_or_path /share/wanghanzhen/MTP/dMTP/mtpmodel \
    --train_data_dir /share/wanghanzhen/MTP/dMTP/train_eaglestyle/data_collection/training_data/prompt_000001 \
    --output_dir /share/wanghanzhen/MTP/dMTP/train_eaglestyle/mtp_checkpoints_whead \
    --block_length 4 \
    --per_device_train_batch_size 128 \
    --dataloader_num_workers 8 \
    --num_train_epochs 100 \
    --learning_rate 5e-5 \
    --warmup_step 500 \
    --logging_steps 100 \
    --save_steps 10000
    # --test-mode

```

#### 3.2 从头训练
```bash
python train_mtp_head.py \
    --num_hidden_layers 1 \
    --vocab_size 151936 \
    --train_data_dir /path/to/collected/data/collected_data_final \
    --output_dir ./mtp_checkpoints \
    --block_length 4 \
    --per_device_train_batch_size 32 \
    --num_train_epochs 3 \
    --learning_rate 1e-4 \
    --scheduler_type linear \
    --loss_weight_type scheduler
```

**主要参数：**

模型参数：
- `model_name_or_path`: 预训练模型路径（如果为 None 则从头初始化）
- `hidden_size`: 隐藏层大小（需与基础模型匹配）
- `num_hidden_layers`: MTP 头的 transformer 层数（通常为 1）
- `vocab_size`: 词汇表大小
- `mask_token_id`: MASK token ID
- `pad_token_id`: PAD token ID

数据参数：
- `train_data_dir`: 收集的数据目录（必需）
- `block_length`: 预测的 token 数量 (L)
- `max_samples`: 最大样本数（用于快速测试）
- `train_ratio`: 训练/验证数据比例

训练参数：
- `output_dir`: 模型保存目录
- `per_device_train_batch_size`: 批大小
- `num_train_epochs`: 训练轮数
- `learning_rate`: 学习率
- `warmup_steps`: 预热步数
- `logging_steps`: 日志记录间隔
- `save_steps`: 模型保存间隔
- `scheduler_type`: alpha 调度器类型 ('linear' 或 'kappa')
- `loss_weight_type`: 损失权重计算方式 ('scheduler' 或 'ones')

## 完整训练示例

假设你已经准备好了输入文本文件 `/data/train_texts.txt`。

### 1. 数据收集（~2-4 小时，取决于文本大小）
```bash
python data_collection.py \
    --base_model_path Qwen/Qwen2-7B \
    --input_data_path /data/train_texts.txt \
    --output_dir /data/mtp_collected \
    --max_samples 50000 \
    --sample_size 512 \
    --stride 256 \
    --device cuda:0
```

### 2. 训练 MTP 头（~1-2 小时）
```bash
python /share/wanghanzhen/MTP/dMTP/train_eaglestyle/train_mtp_head.py \
    --block_length 4 \
    --per_device_train_batch_size 32 \
    --num_train_epochs 5 \
    --learning_rate 5e-05 \
    --warmup_steps 1000 \
    --logging_steps 1000 \
    --save_steps 100000 \
    --scheduler_type linear \
    --time_epsilon 0.01
```

## 训练详解

### 损失函数设计

根据 `pipeline.md` step3，损失函数计算：

1. **时间采样**：对每个样本随机采样时间步 $t \in [\epsilon, 1)$
2. **随机掩码**：根据调度器的 $\alpha(t)$ 计算掩码概率 $p_{mask} = 1 - \alpha(t)$
3. **掩码应用**：将每个位置的 token 独立地以概率 $p_{mask}$ 掩码化
4. **前向传播**：将掩码的 token 和条件隐藏状态输入 MTP 头
5. **损失计算**：计算预测 logits 和真实 token 的交叉熵

### 数据流

```
基础模型隐藏状态 h_l [hidden_size]
       ↓
   [条件]
       ↓
   MTP 头输入：h_l + [MASK] token 序列
       ↓
   离散扩散前向过程（随机掩码）
       ↓
   MTP 头反向推理（去掩码）
       ↓
   预测 L 个 token: t_l, t_(l+1), ..., t_(l+L-1)
       ↓
   与真实 token 对比，计算损失
```

## 常见问题

### Q1: 数据收集时 OOM
**解决**：减小 `max_samples`、`sample_size`，或增加采样 `stride`

### Q2: 训练速度慢
**解决**：
- 增加 `per_device_train_batch_size`
- 减少 `logging_steps` 和 `save_steps`
- 使用 DeepSpeed Zero-2 或 Zero-3
- 减少 `num_train_epochs`

### Q3: 损失不下降
**解决**：
- 检查数据加载是否正确（验证 shape）
- 尝试不同的学习率
- 增加预热步数 `warmup_steps`
- 尝试不同的调度器类型

### Q4: 如何使用多 GPU？
**解决**：
```bash
python -m torch.distributed.launch --nproc_per_node 4 train_mtp_head.py \
    --train_data_dir ... \
    --output_dir ... \
    --per_device_train_batch_size 16 \
    ...
```

## 文件说明

- `data_collection.py`: 从基础模型收集隐藏状态和 token
- `mtp_dataset.py`: 数据加载器，将收集的数据转换为训练样本
- `train_mtp_head.py`: 主训练脚本
- `mtphead_trainer.py`: 自定义 Trainer（实现离散扩散训练逻辑）
- `schedulers/`: Alpha 调度器实现

## 训练检查清单

- [ ] 准备输入文本文件
- [ ] 确认基础模型路径和配置
- [ ] 运行数据收集脚本
- [ ] 验证收集的数据格式和大小
- [ ] 配置训练参数
- [ ] 运行训练脚本
- [ ] 监控训练进度（loss 下降趋势）
- [ ] 验证输出模型

## 参考文献

1. Simple and Effective Masked Diffusion Language Models (https://arxiv.org/abs/2406.07524)
2. Large Language Diffusion Models (https://arxiv.org/abs/2502.09992)

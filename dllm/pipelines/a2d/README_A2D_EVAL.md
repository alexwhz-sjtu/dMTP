# A2D Evaluation Script

本目录包含A2D模型的评估脚本，用于在各种基准数据集上评估模型性能。

## 文件说明

- **eval.py**: A2D模型的lm-eval评估框架集成脚本，支持多GPU并行评估
- **eval.sh**: 评估脚本的shell包装器，包含多个基准任务的评估命令
- **sample_a2d.py**: A2D模型的演示脚本，展示如何使用采样器进行文本生成

## 使用方法

### 1. 单个任务评估

```bash
# 评估GSM8K任务（5-shot）
accelerate launch --num_processes 1 dllm/pipelines/a2d/eval.py \
    --tasks gsm8k_cot \
    --model a2d \
    --apply_chat_template \
    --num_fewshot 0 \
    --model_args "pretrained=/share/wanghanzhen/MTP/model/dllm/Qwen3-0.6B-diffusion-bd3lm-v0.1,max_new_tokens=256,steps=256,block_size=32,cfg=0.0"
```

### 2. 使用eval.sh进行多个任务评估

```bash
# 评估所有任务（normal模式）
bash examples/a2d/bd3lm/eval.sh --num_gpu 1

# 仅评估编码任务（coder模式）
bash examples/a2d/bd3lm/eval.sh --model_type coder --num_gpu 1

# 自定义模型路径
bash examples/a2d/bd3lm/eval.sh --model_name_or_path /path/to/model --num_gpu 4
```

### 3. 模型采样示例

```bash
python -u examples/a2d/bd3lm/sample_a2d.py \
    --model_name_or_path "YOUR_MODEL_PATH" \
    --seed 42 \
    --visualize True
```

## 支持的任务

- **mmlu_generative**: MMLU generative任务
- **mmlu_pro**: MMLU Pro任务
- **hellaswag_gen**: HellaSwag generative任务
- **gsm8k_cot**: GSM8K chain-of-thought任务
- **bbh**: Big Bench Hard任务
- **minerva_math**: Minerva数学任务
- **humaneval_instruct**: HumanEval代码生成任务
- **mbpp_instruct**: MBPP代码生成任务

## 主要参数说明

### eval.py参数

- `--tasks`: 评估任务名称
- `--model`: 模型标识符（对于A2D应为"a2d"）
- `--apply_chat_template`: 是否应用chat template
- `--num_fewshot`: Few-shot学习的样本数
- `--model_args`: 模型参数字符串，包括：
  - `pretrained`: 模型路径
  - `max_new_tokens`: 最大生成token数
  - `steps`: 扩散步数
  - `block_size`: 块大小
  - `cfg`: 无分类器引导强度
  - `temperature`: 采样温度
  - `remasking`: 重新掩码策略

### eval.sh参数

- `--model_name_or_path`: 模型路径
- `--num_gpu`: 使用的GPU数量
- `--model_type`: 模型类型（normal或coder）

## 配置环境变量

eval.sh自动设置以下环境变量以确保稳定运行：

- `PYTHONPATH=.:$PYTHONPATH`: 包含当前目录到Python路径
- `HF_ALLOW_CODE_EVAL=1`: 允许代码评估
- `HF_DATASETS_TRUST_REMOTE_CODE=True`: 信任remote数据集代码
- `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`: 启用异步错误处理
- `NCCL_DEBUG=warn`: NCCL警告级别日志
- `TORCH_DISTRIBUTED_DEBUG=DETAIL`: 详细分布式调试日志

## 参考

该评估脚本基于以下现有实现改进：

- `dllm/pipelines/llada/eval.py`: LLaDA模型评估脚本
- `examples/a2d/bd3lm/sample.py`: BD3LM采样脚本

## 注意事项

1. 确保已安装所有必需的依赖库（transformers, datasets, lm-eval, accelerate等）
2. 使用GPU评估时，请确保有足够的显存
3. 对于大规模评估，建议使用accelerate多GPU并行
4. 评估结果会自动保存到日志目录

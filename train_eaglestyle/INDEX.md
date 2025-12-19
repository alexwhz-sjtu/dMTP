#!/usr/bin/env python3
"""
MTP Head Training Pipeline - 文件索引和快速导航

说明: 本脚本生成所有训练文件的导航指南
"""

import os
from pathlib import Path


def print_header():
    """打印头部信息"""
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                   MTP Head Training Pipeline                             ║
║              基于 pipeline.md 的扩散头完整训练实现                         ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)


def print_file_index():
    """打印文件索引"""
    current_dir = Path(__file__).parent if '__file__' in globals() else Path('.')
    
    print("\n" + "=" * 78)
    print("📁 文件索引")
    print("=" * 78)
    
    files = {
        "核心训练脚本": [
            ("data_collection.py", "Step 1: 从基础模型收集隐藏状态和 token"),
            ("mtp_dataset.py", "Step 2: 将收集的数据转换为训练样本"),
            ("train_mtp_head.py", "Step 3: 使用离散扩散训练 MTP 头"),
        ],
        "辅助工具": [
            ("test_pipeline.py", "验证数据加载和训练流程是否正常"),
            ("mtphead_trainer.py", "自定义 Trainer，实现离散扩散损失计算"),
            ("examples.py", "各种训练场景的实际命令示例"),
            ("config_examples.py", "预设的配置参数（快速测试、标准训练等）"),
            ("quick_start.sh", "交互式快速启动脚本"),
        ],
        "文档": [
            ("README.md", "项目概览和快速开始指南"),
            ("TRAINING_GUIDE.md", "详细的训练教程和常见问题解答"),
            ("QUICK_REFERENCE.md", "快速参考卡和命令速查"),
            ("COMPLETION_SUMMARY.md", "项目完成状态和使用说明"),
            ("pipeline.md", "原始的训练思路和架构设计文档"),
        ],
        "其他": [
            ("schedulers/", "Alpha 调度器（LinearAlphaScheduler, KappaAlphaScheduler）"),
            ("__init__.py", "Python 模块导出"),
        ],
    }
    
    for category, items in files.items():
        print(f"\n【{category}】")
        for filename, description in items:
            status = "✓" if (current_dir / filename).exists() else "✗"
            print(f"  {status} {filename:<30} # {description}")


def print_quick_start():
    """打印快速开始指南"""
    print("\n" + "=" * 78)
    print("🚀 快速开始")
    print("=" * 78)
    
    print("""
1️⃣  查看快速参考
    cat QUICK_REFERENCE.md

2️⃣  查看训练示例
    python examples.py --list           # 列出所有训练场景
    python examples.py --scenario quick_test   # 查看快速测试命令

3️⃣  运行快速启动脚本
    bash quick_start.sh

4️⃣  手动执行训练步骤

    # 步骤 1: 数据收集
    python data_collection.py \\
        --base_model_path Qwen/Qwen2-7B \\
        --input_data_path ./texts.txt \\
        --output_dir ./collected_data

    # 步骤 2: 验证流程
    python test_pipeline.py \\
        --data_dir ./collected_data/collected_data_final

    # 步骤 3: 训练模型
    python train_mtp_head.py \\
        --train_data_dir ./collected_data/collected_data_final \\
        --output_dir ./checkpoint \\
        --block_length 4 \\
        --per_device_train_batch_size 32 \\
        --num_train_epochs 5 \\
        --learning_rate 2e-4
    """)


def print_documentation():
    """打印文档导航"""
    print("\n" + "=" * 78)
    print("📚 文档导航")
    print("=" * 78)
    
    docs = {
        "QUICK_REFERENCE.md": "快速参考卡 - 常用命令速查",
        "README.md": "项目文档 - 详细功能介绍",
        "TRAINING_GUIDE.md": "训练指南 - 完整教程和问题解答",
        "COMPLETION_SUMMARY.md": "完成摘要 - 项目状态和使用说明",
        "pipeline.md": "原始文档 - 架构和算法说明",
    }
    
    print("\n推荐阅读顺序:")
    for i, (filename, description) in enumerate(docs.items(), 1):
        print(f"  {i}. {filename:<30} - {description}")


def print_use_cases():
    """打印使用场景"""
    print("\n" + "=" * 78)
    print("🎯 使用场景")
    print("=" * 78)
    
    scenarios = [
        ("快速验证", "quick_test", "python examples.py --scenario quick_test"),
        ("标准训练", "standard", "python examples.py --scenario standard"),
        ("大规模训练", "large_scale", "python examples.py --scenario large_scale"),
        ("模型微调", "finetune", "python examples.py --scenario finetune"),
        ("DeepSpeed 训练", "deepspeed", "python examples.py --scenario deepspeed"),
        ("超参研究", "experiment", "python examples.py --scenario experiment"),
    ]
    
    for name, scenario, cmd in scenarios:
        print(f"\n• {name}")
        print(f"  场景: {scenario}")
        print(f"  命令: {cmd}")


def print_key_features():
    """打印关键特性"""
    print("\n" + "=" * 78)
    print("✨ 关键特性")
    print("=" * 78)
    
    features = [
        "完整的离散扩散训练实现",
        "自动数据收集和预处理",
        "灵活的模型配置系统",
        "多 GPU 分布式训练支持",
        "详细的文档和示例",
        "测试脚本快速验证",
        "多种预设配置",
        "互动式快速启动",
    ]
    
    for feature in features:
        print(f"  ✓ {feature}")


def print_training_theory():
    """打印训练理论"""
    print("\n" + "=" * 78)
    print("🧠 训练原理")
    print("=" * 78)
    
    print("""
MTP（多令牌预测）头使用离散扩散原理训练：

1. 时间采样
   - 为每个样本随机采样时间步 t ∈ [ε, 1)
   - ε 通常设为 0.01 以避免退化

2. 掩码率计算
   - 通过调度器计算 α(t)
   - 掩码概率 p_mask = 1 - α(t)

3. 随机掩码
   - 以 p_mask 的概率独立掩码化每个 token
   - 掩码 token 替换为 [MASK] token ID

4. 前向传播
   - 输入: 条件隐藏状态 + 掩码序列
   - 输出: 预测的 logits

5. 损失计算
   - 仅在掩码位置计算交叉熵损失
   - 可选的损失权重（基于时间步）

6. 反向传播
   - 计算梯度并更新模型参数

关键参数：
- block_length (L): 一次预测的 token 数，通常 2-8
- scheduler_type: 调度器类型（linear 或 kappa）
- time_epsilon: 最小时间步，避免退化值
- loss_weight_type: 损失权重计算方式
    """)


def print_tips():
    """打印实用提示"""
    print("\n" + "=" * 78)
    print("💡 实用提示")
    print("=" * 78)
    
    tips = {
        "数据准备": [
            "输入文本文件：每行一个文档，无需特殊格式",
            "推荐最少 10K 文档用于良好的训练效果",
            "可处理任意大小的文本（自动分块）",
        ],
        "内存优化": [
            "OOM 时：减小 batch_size 或增加 gradient_accumulation_steps",
            "block_length 越大，显存需求越大",
            "使用 DeepSpeed Zero-2/3 处理大模型",
        ],
        "训练监控": [
            "loss 应该逐步下降，如果平坦检查学习率",
            "使用 logging_steps 参数调整日志频率",
            "查看 training_info.json 了解最终指标",
        ],
        "性能优化": [
            "多 GPU 训练：torch.distributed.launch --nproc_per_node N",
            "增加 num_workers 以加快数据加载",
            "使用 gradient_accumulation 改进精度而不增加 batch_size",
        ],
    }
    
    for category, items in tips.items():
        print(f"\n【{category}】")
        for tip in items:
            print(f"  • {tip}")


def print_next_steps():
    """打印下一步"""
    print("\n" + "=" * 78)
    print("📋 下一步")
    print("=" * 78)
    
    print("""
1. 准备输入数据
   └─ 创建包含训练文本的 texts.txt 文件

2. 查看快速参考
   └─ cat QUICK_REFERENCE.md

3. 选择适合的训练场景
   └─ python examples.py --list

4. 执行数据收集
   └─ python data_collection.py ...

5. 验证流程（可选）
   └─ python test_pipeline.py ...

6. 开始训练
   └─ python train_mtp_head.py ...

7. 监控训练进度
   └─ 查看 loss 和 checkpoint

8. 评估模型
   └─ 在下游任务上测试
    """)


def print_support():
    """打印帮助和支持"""
    print("\n" + "=" * 78)
    print("🤝 帮助和支持")
    print("=" * 78)
    
    support = {
        "遇到问题？": [
            "查看 TRAINING_GUIDE.md 的常见问题部分",
            "运行 python examples.py --tips",
            "检查日志和错误消息",
            "验证数据格式和路径",
        ],
        "学习资源": [
            "论文: Simple and Effective Masked Diffusion Language Models",
            "论文: Large Language Diffusion Models",
            "文档: HuggingFace Transformers",
        ],
        "快速命令": [
            "查看所有示例: python examples.py --list",
            "查看实用提示: python examples.py --tips",
            "测试流程: python test_pipeline.py --help",
            "查看配置: python config_examples.py --config standard",
        ],
    }
    
    for category, items in support.items():
        print(f"\n【{category}】")
        for item in items:
            print(f"  • {item}")


def main():
    """主函数"""
    print_header()
    print_file_index()
    print_quick_start()
    print_documentation()
    print_use_cases()
    print_key_features()
    print_training_theory()
    print_tips()
    print_next_steps()
    print_support()
    
    print("\n" + "=" * 78)
    print("✅ 准备就绪！开始训练吧！")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()

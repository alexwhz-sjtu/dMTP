"""
Step 3: MTP Head Training Script

Trains the diffusion-based MTP head following the pipeline.md training strategy:
1. Load pre-collected hidden states and tokens from base model
2. For each sample position l, train the MTP head to predict L tokens given h_l
3. Use discrete diffusion with masked tokens as input to the head
"""

import os
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any
import argparse
from dataclasses import dataclass, field, asdict
import logging

from transformers import (
    AutoConfig,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    HfArgumentParser,
    set_seed,
)
from transformers.trainer_callback import TrainerCallback

from mtp_dataset import create_dataloaders, MTPHeadDataset
from mtphead_trainer import MDLMTrainer
from schedulers.alpha import LinearAlphaScheduler
from schedulers.kappa import LinearKappaScheduler

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.modeling_mtp_head import DreamModel
from model.mtphead_config import MTPHeadConfig


logger = logging.getLogger(__name__)
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"

@dataclass
class ModelArguments:
    """Arguments for model configuration."""
    
    model_name_or_path: str = field(
        default="/share/wanghanzhen/MTP/dMTP/mtpmodel",
        metadata={"help": "Path to pretrained MTP head model or HuggingFace model ID"}
    )
    hidden_size: int = field(
        default=4096,
        metadata={"help": "Hidden size of MTP head (should match base model)"}
    )
    num_hidden_layers: int = field(
        default=1,
        metadata={"help": "Number of transformer layers in MTP head"}
    )
    vocab_size: int = field(
        default=151936,
        metadata={"help": "Vocabulary size"}
    )
    mask_token_id: int = field(
        default=151666,
        metadata={"help": "Token ID for [MASK]"}
    )
    pad_token_id: int = field(
        default=151643,
        metadata={"help": "Token ID for padding"}
    )


@dataclass
class DataArguments:
    """Arguments for data loading."""
    
    train_data_dir: str = field(
        default="/share/wanghanzhen/MTP/dMTP/train_eaglestyle/data_collection/training_data/prompt_000001",
        metadata={"help": "Directory containing collected training data (from data_collection.py)"}
    )
    block_length: int = field(
        default=4,
        metadata={"help": "Number of tokens to predict (L in pipeline.md)"}
    )
    max_samples: Optional[int] = field(
        default=1000,
        metadata={"help": "Maximum number of samples to use (for quick testing)"}
    )
    train_ratio: float = field(
        default=0.9,
        metadata={"help": "Ratio of data for training vs validation"}
    )
    test_mode: bool = field(
        default=False,
        metadata={"help": "Test mode: use only first hidden state + next block_length tokens to test code correctness"}
    )
    condition_len: int = field(
        default=1,
        metadata={"help": "Number of historical hidden states used as condition (sliding window)."}
    )


@dataclass
class TrainingExtraArguments:
    """Extra training arguments."""
    
    scheduler_type: str = field(
        default="linear",
        metadata={"help": "Type of alpha scheduler: 'linear' or 'kappa'"}
    )
    time_epsilon: float = field(
        default=0.01,
        metadata={"help": "Minimum timestep (avoid degenerate values near 0)"}
    )
    loss_weight_type: str = field(
        default="scheduler",
        metadata={"help": "Loss weight type: 'scheduler' or 'ones'"}
    )
    right_shift_logits: bool = field(
        default=False,
        metadata={"help": "Whether to right-shift logits (for causal LM training)"}
    )



class LoggingCallback(TrainerCallback):
    """Custom callback for logging training statistics."""
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0:
            # Safely get loss from log history
            loss_val = "N/A"
            if state.log_history and len(state.log_history) > 0:
                loss_val = state.log_history[-1].get('loss', 'N/A')
            logger.info(f"Step {state.global_step}: loss={loss_val}")


def setup_model(base_model_path:str, model_args: ModelArguments, device: str = "cuda"):
    """
    Load or initialize the MTP head model.
    
    Args:
        model_args: Model configuration arguments
        device: Device to load model on
        
    Returns:
        Initialized model
    """
    if model_args.model_name_or_path:
        
        # Import the model class
        logger.info("Initializing model from local...")
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        base_model_config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)

        # Step 2: Load full base model (we only need its state dict for lm_head & embed_tokens)
        print("Loading base model (to extract lm_head and embeddings)...")
        # Load the base model entirely on CPU to avoid occupying 
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            device_map={"": "cpu"},  # 强制放在 CPU
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )

        hidden_size = base_model_config.hidden_size
        vocab_size = base_model_config.vocab_size
        num_attention_heads = base_model_config.num_attention_heads
        num_key_value_heads = base_model_config.num_key_value_heads
        intermediate_size = base_model_config.intermediate_size

        mtp_head_config = MTPHeadConfig(
            hidden_size=hidden_size,
            num_hidden_layers=1,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            intermediate_size=intermediate_size,
            vocab_size=vocab_size,
        )
        
        model = DreamModel(mtp_head_config)
        # # Step 5: retrain lm_head
        # print("Copying lm_head and embed_tokens from base model...")
        # with torch.no_grad():
        #     # Copy language modeling head
        #     if hasattr(base_model, 'lm_head'):
        #         model.lm_head.weight.copy_(base_model.lm_head.weight)
        #     else:
        #         print("⚠️ Warning: base model has no 'lm_head', skipping.")

        #     # Freeze copied weights so they are not updated during MTP head training
        #     for p in model.lm_head.parameters():
        #         p.requires_grad = False


        # free base model asap so it won't remain on any device
 
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"✓ Model initialized with shared weights")
        print(f"  Model type: {mtp_head_config.model_type}")
        print(f"  Parameters (excluding shared parts): {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        print(f"  Vocab size: {vocab_size}")
        


            # logger.info("Using generic transformer model initialization")
            
            # # Fallback: create simple model config
            # from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
            # from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
            
            # config = Qwen2Config(
            #     hidden_size=model_args.hidden_size,
            #     num_hidden_layers=model_args.num_hidden_layers,
            #     vocab_size=model_args.vocab_size,
            #     num_attention_heads=max(1, model_arg.hidden_size // 64),
            #     intermediate_size=model_args.hidden_size * 4,
            # )
            # model = Qwen2ForCausalLM(confi

    # (via `training_args.fp16 = True`) to enable mixed precision safely.
    # This lets AMP keep some layers (LayerNorm, softmax, etc.) in FP32
    # while using FP16 where beneficial.
    # model = model.to(device)

    return model


def main():
    parser = HfArgumentParser((
        ModelArguments,
        DataArguments,
        TrainingArguments,
        TrainingExtraArguments,
    ))
    
    model_args, data_args, training_args, extra_args = parser.parse_args_into_dataclasses()

    # Ensure Trainer runs in mixed precision (fp16) when CUDA is available.
    # This improves performance and reduces GPU memory use. Users can still
    # override via CLI args (e.g., --fp16 False) if desired.
    # if torch.cuda.is_available():
    #     training_args.fp16 = True

    base_model_path = "/share/public/public_models/Qwen2.5-7B-Instruct"
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARNING,
    )
    
    logger.info(f"Model arguments: {asdict(model_args)}")
    logger.info(f"Data arguments: {asdict(data_args)}")
    logger.info(f"Training arguments: {asdict(training_args)}")
    logger.info(f"Extra arguments: {asdict(extra_args)}")
    
    # Set seed
    set_seed(training_args.seed)
    
    # Create output directory
    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # ========== Load Data ==========
    logger.info("Loading datasets...")
    if not data_args.train_data_dir:
        raise ValueError("train_data_dir must be specified")
    
    print(training_args.dataloader_num_workers)
    
    train_dataloader, dataset = create_dataloaders(
        data_dir=data_args.train_data_dir,
        block_length=data_args.block_length,
        batch_size=training_args.per_device_train_batch_size,
        num_workers=training_args.dataloader_num_workers,
        train_ratio=data_args.train_ratio,
        shuffle=True,
        mask_token_id=model_args.mask_token_id,
        pad_token_id=model_args.pad_token_id,
        test_mode=data_args.test_mode,
        condition_len=data_args.condition_len,
    )

    
    logger.info(f"Dataset info:")
    logger.info(f"  Total samples: {len(dataset)}")
    logger.info(f"  Block length: {data_args.block_length}")
    logger.info(f"  Hidden size: {dataset.hidden_size}")
    
    # ========== Setup Model ==========
    logger.info("Setting up model...")
    device = f"cuda:{training_args.local_rank}" if torch.cuda.is_available() else "cpu"
    model = setup_model(base_model_path, model_args, device=device)
    
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # ========== Setup Scheduler ==========
    logger.info(f"Setting up scheduler: {extra_args.scheduler_type}")
    if extra_args.scheduler_type == "linear":
        scheduler = LinearAlphaScheduler()
    elif extra_args.scheduler_type == "kappa":
        scheduler = LinearKappaScheduler()
    else:
        raise ValueError(f"Unknown scheduler type: {extra_args.scheduler_type}")
    
    # ========== Setup Trainer ==========
    logger.info("Setting up trainer...")

    trainer = MDLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataloader.dataset if hasattr(train_dataloader, 'dataset') else None,
        data_collator=train_dataloader.collate_fn,
        train_dataloader=train_dataloader,
        scheduler=scheduler,
        time_epsilon=extra_args.time_epsilon,
        loss_weight_type=extra_args.loss_weight_type,
        right_shift_logits=extra_args.right_shift_logits,
        callbacks=[LoggingCallback()],
    )
    
    # ========== Train ==========
    logger.info("Starting training...")
    train_result = trainer.train()
    
    logger.info(f"Training completed!")
    logger.info(f"Training result: {train_result}")
    
    # ========== Save Model ==========
    logger.info(f"Saving model to {training_args.output_dir}...")
    trainer.save_model(training_args.output_dir)
    
    # Save training info
    training_info = {
        "model_args": asdict(model_args),
        "data_args": asdict(data_args),
        "training_args": training_args.to_dict(),
        "extra_args": asdict(extra_args),
        "final_metrics": train_result.metrics,
    }
    
    info_path = Path(training_args.output_dir) / "training_info.json"
    with open(info_path, "w") as f:
        json.dump(training_info, f, indent=2)
    
    logger.info(f"Training info saved to {info_path}")
    
    return train_result


if __name__ == "__main__":
    main()

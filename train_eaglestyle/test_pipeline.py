"""
Quick test script to verify the training pipeline setup.

This script tests:
1. Dataset loading
2. Data shapes and formats
3. Model initialization
4. Training step forward pass

Usage:
    python test_pipeline.py --data_dir /path/to/collected/data/collected_data_final
"""

import torch
import argparse
from pathlib import Path
from typing import Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Try to create a minimal model
from model.mtphead_config import MTPHeadConfig
from model.modeling_mtp_head import DreamModel


def test_dataset_loading(data_dir: str, block_length: int = 4):
    """Test dataset loading and sample extraction."""
    print("=" * 50)
    print("Testing dataset loading...")
    print("=" * 50)
    
    try:
        from mtp_dataset import MTPHeadDataset, create_dataloaders
        
        # Test single dataset
        dataset = MTPHeadDataset(
            data_dir=data_dir,
            block_length=block_length,
        )
        
        print(f"✓ Dataset loaded successfully")
        print(f"  Dataset size: {len(dataset)}")
        print(f"  Sequence length: {dataset.seq_len}")
        print(f"  Hidden size: {dataset.hidden_size}")
        print(f"  Block length: {dataset.block_length}")
        
        # Test sample extraction
        sample = dataset[0]
        print(f"\n✓ Sample extracted")
        for key, value in sample.items():
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        
        # Test dataloader
        train_dl, val_dl, _ = create_dataloaders(
            data_dir=data_dir,
            block_length=block_length,
            batch_size=4,
            train_ratio=0.9,
        )
        
        print(f"\n✓ Dataloaders created")
        print(f"  Train samples: {len(train_dl.dataset)}")
        print(f"  Val samples: {len(val_dl.dataset)}")
        
        # Test batch
        batch = next(iter(train_dl))
        print(f"\n✓ Batch extracted")
        for key, value in batch.items():
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        
        return batch
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_model_initialization(
    base_model_path,
    hidden_size: int = 4096,
    vocab_size: int = 151936,
    block_length: int = 4
):
    """Test model initialization."""
    print("\n" + "=" * 50)
    print("Testing model initialization...")
    print("=" * 50)
    
    try:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        # load base model config 
        base_model_config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)


        # get base model's hidden_size
        # Step 2: Load full base model (we only need its state dict for lm_head & embed_tokens)
        print("Loading base model (to extract lm_head and embeddings)...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            device_map="cpu",  # 避免显存不足，只加载到 CPU
            torch_dtype="auto"
        )

        hidden_size = base_model_config.hidden_size
        vocab_size = base_model_config.vocab_size
        num_attention_heads = base_model_config.num_attention_heads
        num_key_value_heads = base_model_config.num_key_value_heads
        intermediate_size = base_model_config.intermediate_size

        print(f"hidden_size = {hidden_size}")
        
        mtp_head_config = MTPHeadConfig(
            hidden_size=hidden_size,
            num_hidden_layers=1,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            intermediate_size=intermediate_size,
            vocab_size=vocab_size,
        )
        
        model = DreamModel(mtp_head_config)
        # Step 5: Copy lm_head and embed_tokens from base model
        print("Copying lm_head and embed_tokens from base model...")
        with torch.no_grad():
            # Copy language modeling head
            if hasattr(base_model, 'lm_head'):
                model.lm_head.weight.copy_(base_model.lm_head.weight)
            else:
                print("⚠️ Warning: base model has no 'lm_head', skipping.")

            # Copy embedding layer
            if hasattr(base_model, 'get_input_embeddings'):
                base_embed = base_model.get_input_embeddings()
                model.embed_tokens.weight.copy_(base_embed.weight)
            else:
                print("⚠️ Warning: cannot get input embeddings from base model.")

        print(f"✓ Model initialized with shared weights")
        print(f"  Model type: {mtp_head_config.model_type}")
        print(f"  Parameters (excluding shared parts): {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        print(f"  Vocab size: {vocab_size}")
        
        return model
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_forward_pass(model, batch, device: str = "cuda"):
    """Test model forward pass with batch data."""
    print("\n" + "=" * 50)
    print("Testing forward pass...")
    print("=" * 50)
    
    try:
        if model is None or batch is None:
            print("✗ Skipped (missing model or batch)")
            return False
        
        model = model.to(device)
        
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        # Forward pass
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)
        labels = batch.get("labels", None)
        
        print(f"Input shapes:")
        print(f"  input_ids: {input_ids.shape}")
        if attention_mask is not None:
            print(f"  attention_mask: {attention_mask.shape}")
        if labels is not None:
            print(f"  labels: {labels.shape}")
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels if labels is not None else None,
            )
        
        print(f"\n✓ Forward pass successful")
        print(f"  Output logits shape: {outputs.logits.shape}")
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            print(f"  Loss: {outputs.loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step(model, batch, device: str = "cuda"):
    """Test a single training step."""
    print("\n" + "=" * 50)
    print("Testing training step...")
    print("=" * 50)
    
    try:
        if model is None or batch is None:
            print("✗ Skipped (missing model or batch)")
            return False
        
        model = model.to(device).train()
        
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Forward pass
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)
        labels = batch.get("labels", None)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels if labels is not None else None,
        )
        
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        print(f"✓ Training step successful")
        print(f"  Loss: {loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test MTP training pipeline")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to collected data directory")
    parser.add_argument("--block_length", type=int, default=4, help="Block length")
    parser.add_argument("--hidden_size", type=int, default=4096, help="Hidden size")
    parser.add_argument("--vocab_size", type=int, default=151936, help="Vocabulary size")
    parser.add_argument("--base_model_path", type=str, default="/share/public/public_models/Qwen2.5-7B-Instruct", help="Model path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    print(f"Device: {args.device}")
    print(f"Block length: {args.block_length}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Vocab size: {args.vocab_size}")
    
    # Test 1: Dataset loading
    batch = None
    if args.data_dir:
        batch = test_dataset_loading(args.data_dir, block_length=args.block_length)
    else:
        print("⚠ Skipping dataset test (--data_dir not provided)")
    
    # Test 2: Model initialization
    model = test_model_initialization(
        hidden_size=args.hidden_size,
        vocab_size=args.vocab_size,
        block_length=args.block_length,
        base_model_path=args.base_model_path,
    )
    
    # Test 3: Forward pass
    if batch is not None:
        test_forward_pass(model, batch, device=args.device)
    
    # Test 4: Training step
    if batch is not None:
        test_training_step(model, batch, device=args.device)
    
    print("\n" + "=" * 50)
    print("All tests completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()

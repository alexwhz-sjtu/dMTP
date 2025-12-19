"""
Step 1: Data Collection for MTP Head Training

This script collects hidden states and tokens from the base LLM's generated responses.

WORKFLOW:
    1. Load base model (e.g., Qwen2.5)
    2. For each input prompt/question:
       a. Generate response using greedy decoding
       b. Extract the response tokens (excluding the prompt)
       c. Get hidden states from the last layer for each response token
    3. Use sliding window to create training samples from each response
    4. Save as PyTorch tensors for training

STRIDE EXPLANATION:
    stride is the sliding window step size for creating training samples from responses.
    - sample_size: window length (e.g., 512 tokens)
    - stride: step size between consecutive windows (e.g., 256 tokens)
    - Example: for a response of 1024 tokens with sample_size=512, stride=256:
      Windows: [0:512], [256:768], [512:1024]
      This creates 3 samples with 50% overlap
    - Smaller stride → more samples, more overlap, more data
    - Larger stride → fewer samples, less overlap, less redundancy

Usage:
    python data_collection.py \\
        --base_model_path Qwen/Qwen2.5-7B-Instruct \\
        --input_data_path questions.txt \\
        --output_dir ./collected_data \\
        --max_new_tokens 512 \\
        --sample_size 512 \\
        --stride 256
"""

import json
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from collections import defaultdict


class DataCollector:
    """
    Collects hidden states and tokens from base model's generated responses.
    
    The collector:
    1. Takes prompts/questions as input
    2. Generates responses using the base model
    3. Extracts response tokens and their hidden states
    4. Creates training samples using sliding windows
    """
    
    def __init__(
        self,
        base_model_path: str,
        tokenizer_path: str = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize data collector.
        
        Args:
            base_model_path: Path to the base model
            tokenizer_path: Path to tokenizer (defaults to base_model_path)
            device: Device to run inference on
            dtype: Data type for model
        """
        self.device = device
        self.dtype = dtype
        
        print(f"Loading tokenizer from {tokenizer_path or base_model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path or base_model_path,
            use_fast=False,
            trust_remote_code=True
        )
        
        print(f"Loading base model from {base_model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()
        
    @torch.no_grad()
    def collect_from_text(
        self,
        text: str,
        max_length: int = 10240,
        max_new_tokens: int = 8192,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Collect hidden states and tokens from model's generated response.
        
        The input text is a prompt/question, and we collect the model's generated
        response (tokens and corresponding hidden states).
        
        Args:
            text: Input prompt/question text
            max_length: Maximum total sequence length (prompt + response)
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Dictionary containing:
                - input_ids: Generated response token IDs [response_len]
                - hidden_states: Hidden states from last layer for response [response_len, hidden_size]
                - tokens: Generated response tokens [response_len]
            Returns None if response is too short
        """
        try:
            # Tokenize the prompt/question
            encodings = self.tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            prompt_input_ids = encodings["input_ids"].to(self.device)
            prompt_length = prompt_input_ids.shape[1]
            
            # Generate response using greedy decoding
            # Note: output_hidden_states in generate() may not work in older transformers versions
            # So we generate first, then do a forward pass to get hidden states
            with torch.no_grad():
                generated_ids = self.model.generate(
                    prompt_input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Greedy decoding
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Extract the generated part (excluding the prompt)
            generated_token_ids = generated_ids[0, prompt_length:]  # [response_len]
            response_len = generated_token_ids.shape[0]
            
            if response_len < 2:
                return None  # Skip too short responses
            
            # Now get hidden states for the full sequence (prompt + response)
            # Do forward pass with full sequence to get hidden states
            with torch.no_grad():
                outputs = self.model(
                    input_ids=generated_ids,
                    output_hidden_states=True,
                    return_dict=True,
                )
            
            # Get hidden states from last layer: [batch=1, full_seq_len, hidden_size]
            all_hidden_states = outputs.hidden_states[-1][0, :, :]  # [full_seq_len, hidden_size]
            
            # Extract hidden states for the response part only
            response_hidden_states = all_hidden_states[prompt_length:, :]  # [response_len, hidden_size]
            
            # Verify shapes match
            assert response_hidden_states.shape[0] == response_len, \
                f"Hidden states shape {response_hidden_states.shape[0]} != response length {response_len}"
            
            # Decode to see the response (for logging)
            response_text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
            response_preview = response_text[:100].replace('\n', ' ')
            print(f"  ✓ Generated {response_len} tokens: '{response_preview}...'")
            
            return {
                "input_ids": generated_token_ids.cpu(),
                "hidden_states": response_hidden_states.cpu(),
                "tokens": generated_token_ids.cpu(),
            }
            
        except Exception as e:
            print(f"  ✗ Error collecting from text: {e}")
            return None
    
    @torch.no_grad()
    def collect_from_file(
        self,
        input_file: str,
        output_dir: str,
        max_samples: int = None,
        max_length: int = 8192,
        max_new_tokens: int = 10240,
    ) -> None:
        """
        Collect data from input file and save each prompt to a separate folder.
        
        Data format: each line is a JSON object with a 'prompt' field.
        For each prompt, a complete response is generated and saved to a dedicated subfolder.
        
        Args:
            input_file: Path to input text file (one JSON per line with 'prompt' field)
            output_dir: Root directory to save collected data
            max_samples: Maximum number of prompts to process
            max_length: Maximum total sequence length (prompt + response)
            max_new_tokens: Maximum number of tokens to generate per prompt
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        sample_count = 0
        
        print(f"Reading from {input_file}...")
        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line_idx, line in enumerate(tqdm(lines, desc="Collecting data")):
            line = line.strip()
            if not line:
                continue  # 跳过空行
            
            try:
                data = json.loads(line)
                text = data["prompt"]  # ✅ 提取 prompt 字段
            except json.JSONDecodeError as e:
                print(f"⚠️  Warning: Invalid JSON at line {line_idx + 1}: {e}")
                continue
            except KeyError:
                print(f"⚠️  Warning: Missing 'prompt' key at line {line_idx + 1}")
                continue

            if not text:
                continue
            
            sample_count += 1
            print(f"Processing prompt {sample_count})...")
            
            try:
                # Collect data from this prompt (generates response and collects hidden states)
                collected_data = self.collect_from_text(text, max_length=max_length, max_new_tokens=max_new_tokens)
                
                # Skip if response is too short
                if collected_data is None:
                    print(f"  ✗ Skipped: response too short")
                    continue
                
                seq_len = collected_data["input_ids"].shape[0]
                
                if seq_len < 2:
                    print(f"  ✗ Skipped: sequence length < 2")
                    continue
                
                # Create a dedicated folder for this prompt
                prompt_folder = Path(output_dir) / f"prompt_{sample_count:06d}"
                self._save_prompt_data(prompt_folder, collected_data)
                
                if max_samples and sample_count >= max_samples:
                    break
                    
            except Exception as e:
                print(f"  ✗ Error processing line {line_idx}: {e}")
                continue
        
        print(f"\nData collection complete!")
        print(f"  Total prompts processed: {sample_count}")
    
    def _save_prompt_data(self, prompt_folder: Path, data: Dict) -> None:
        """Save collected data for a single prompt to a dedicated folder."""
        prompt_folder.mkdir(parents=True, exist_ok=True)
        
        # Save tensors directly (no stacking needed since it's a single prompt)
        hidden_states = data["hidden_states"]  # [seq_len, hidden_size]
        input_ids = data["input_ids"]  # [seq_len]
        tokens = data["tokens"]  # [seq_len]
        
        torch.save(hidden_states, prompt_folder / "hidden_states.pt")
        torch.save(input_ids, prompt_folder / "input_ids.pt")
        torch.save(tokens, prompt_folder / "tokens.pt")
        
        seq_len = input_ids.shape[0]
        print(f"  ✓ Saved to {prompt_folder.name}: seq_len={seq_len}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect hidden states and tokens from base model for MTP head training"
    )
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to base model")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer")
    parser.add_argument("--input_data_path", type=str, required=True, help="Path to input text file (prompts/questions)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save collected data (each prompt in a subfolder)")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of prompts to process")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum total sequence length (prompt + response)")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum number of tokens to generate per prompt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    collector = DataCollector(
        base_model_path=args.base_model_path,
        tokenizer_path=args.tokenizer_path,
        device=args.device,
    )
    
    collector.collect_from_file(
        input_file=args.input_data_path,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()

"""
Step 2: Dataset loader for MTP Head Training

Loads pre-collected hidden states and tokens for training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Optional, List
import numpy as np


class MTPHeadDataset(Dataset):
    """
    Dataset for MTP head training.
    
    Data format:
    - hidden_states.pt: [num_samples, seq_len, hidden_size]
    - tokens.pt: [num_samples, seq_len]
    - (optional) input_ids.pt: [num_samples, seq_len]
    """
    
    def __init__(
        self,
        data_dir: str,
        block_length: int = 4,
        mask_token_id: int = 151666,
        pad_token_id: int = 151643,
        hidden_states_tensor: Optional[torch.Tensor] = None,
        tokens_tensor: Optional[torch.Tensor] = None,
        test_mode: bool = False,
        condition_len: int = 2,
    ):
        """
        Initialize MTP head dataset.
        
        Args:
            data_dir: Directory containing collected data files
            block_length: Number of tokens to predict (L in pipeline.md)
            mask_token_id: Token ID for [MASK]
            pad_token_id: Token ID for padding
            hidden_states_tensor: Optional pre-loaded hidden states tensor
            tokens_tensor: Optional pre-loaded tokens tensor
            test_mode: If True, use only first hidden state + next block_length tokens
        """
        self.data_dir = Path(data_dir)
        self.block_length = block_length
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.test_mode = test_mode
        self.condition_len = condition_len
        
        # Load data (either from provided tensors or from files)
        if hidden_states_tensor is not None and tokens_tensor is not None:
            self.hidden_states = hidden_states_tensor
            self.tokens = tokens_tensor
            self.input_ids = None
        else:
            self.hidden_states = torch.load(self.data_dir / "hidden_states.pt")
            self.tokens = torch.load(self.data_dir / "tokens.pt")
            # Optional
            try:
                self.input_ids = torch.load(self.data_dir / "input_ids.pt")
            except FileNotFoundError:
                self.input_ids = None
        
        print(f"Loaded hidden_states shape: {self.hidden_states.shape}")
        print(f"Loaded tokens shape: {self.tokens.shape}")
        
        # Handle both old format [num_samples, seq_len, hidden_size] and new format [seq_len, hidden_size]
        if self.hidden_states.dim() == 3:
            # Old format: [num_samples, seq_len, hidden_size]
            self.num_samples, self.seq_len, self.hidden_size = self.hidden_states.shape
            # Ensure tokens is also 2D [num_samples, seq_len]
            if self.tokens.dim() == 1:
                # This shouldn't happen in old format, but handle it just in case
                self.tokens = self.tokens.unsqueeze(0)
        elif self.hidden_states.dim() == 2:
            # New format: each prompt folder is ONE sample [seq_len, hidden_size]
            self.seq_len, self.hidden_size = self.hidden_states.shape
            self.num_samples = 1  # Single sample per folder
            # Expand to 3D: [1, seq_len, hidden_size] for uniform handling
            self.hidden_states = self.hidden_states.unsqueeze(0)
            # Expand tokens to 2D: [1, seq_len]
            if self.tokens.dim() == 1:
                self.tokens = self.tokens.unsqueeze(0)
        else:
            raise ValueError(f"Unexpected hidden_states shape: {self.hidden_states.shape}")
        
        print(f"Loaded dataset with {self.num_samples} samples, seq_len={self.seq_len}, hidden_size={self.hidden_size}")
        print(f"After processing - hidden_states shape: {self.hidden_states.shape}, tokens shape: {self.tokens.shape}")
        
        if self.test_mode:
            print(f"⚠️  TEST MODE: Using only first hidden state + next {block_length} tokens for overfitting")
        else:
            if self.seq_len <= self.block_length + (self.condition_len - 1):
                print(f"⚠️  Dataset seq_len={self.seq_len} is too short for block_length={self.block_length} and condition_len={self.condition_len}; no training positions will be available.")
    
    def __len__(self) -> int:
        """Return total number of training examples."""
        if self.test_mode:
            # Test mode: only one example (first hidden state -> next block_length tokens)
            return 1
        else:
            # For each sample, create training examples by sliding window, but skip the first (condition_len-1) positions
            per_sample_positions = max(0, self.seq_len - self.block_length - (self.condition_len - 1))
            return self.num_samples * per_sample_positions
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training example.
        
        Returns:
            Dictionary with:
                - in_hidden_states: Condition hidden state at position t [hidden_size]
                - input_ids: Masked token IDs [block_length]
                - labels: Target token IDs from position t+1 to t+L [block_length]
                - attention_mask: Attention mask [block_length]
        """
        if self.test_mode:
            # Test mode: use the first `condition_len` hidden states as condition and predict next block_length tokens
            sample_idx = 0
            position_idx = self.condition_len - 1
        else:
            # Flattened index across samples: each sample has per_sample_positions positions
            per_sample_positions = max(0, self.seq_len - self.block_length - (self.condition_len - 1))
            if per_sample_positions == 0:
                raise IndexError("No valid training positions available for given block_length and condition_len")
            sample_idx = idx // per_sample_positions
            pos_offset = idx % per_sample_positions
            # actual position index in the sequence must start at (condition_len - 1)
            position_idx = pos_offset + (self.condition_len - 1)
        
        # Condition: hidden states at positions (t - condition_len + 1) .. t
        # in_hs: [condition_len, hidden_size]
        start_cond = max(0, position_idx - (self.condition_len - 1))
        end_cond = position_idx + 1
        in_hs = self.hidden_states[sample_idx, start_cond:end_cond]  # [condition_len, hidden_size]
        
        # Target: tokens from position t+1 to t+block_length (i.e., t+1, t+2, ..., t+L)
        target_start = position_idx + 1  # Start from t+1
        target_end = position_idx + 1 + self.block_length
        
        # Ensure we don't exceed sequence length
        if target_end > self.seq_len:
            target_end = self.seq_len
            actual_block_length = target_end - target_start
        else:
            actual_block_length = self.block_length
        
        target_tokens = self.tokens[sample_idx, target_start:target_end]  # [actual_block_length]
        
        # Create input for MTP head: masked tokens
        masked_tokens = torch.full(
            (actual_block_length,),
            fill_value=self.mask_token_id,
            dtype=torch.long,
        )
         
        # Pad if necessary
        if actual_block_length < self.block_length:
            padding_length = self.block_length - actual_block_length
            masked_tokens = torch.cat([
                masked_tokens,
                torch.full((padding_length,), self.pad_token_id, dtype=torch.long)
            ])
            target_tokens = torch.cat([
                target_tokens,
                torch.full((padding_length,), -100, dtype=torch.long)  # -100 for padding in labels
            ])
        
        # TODO attention mask
        '''
        def build_hybrid_mask(seq_len, device):
            """
            seq_len: total length (e.g., 5)
            Returns: (seq_len, seq_len) mask, where mask[i, j] = 0 if i can attend to j, else -inf
            """
            mask = torch.zeros(seq_len, seq_len, device=device)
            # Condition (pos 0) can only attend to itself
            mask[0, 1:] = float('-inf')
            # Positions 1~end can attend to all (including condition and each other)
            # (already 0, so no change needed)
            return mask

        # 在 forward 中使用：
        seq_len = input_ids.shape[1]  # e.g., 5
        hybrid_mask_2d = build_hybrid_mask(seq_len, device=input_ids.device)  # (5, 5)
        # 扩展为 4D: (1, 1, L, L) -> 广播到 (B, H, L, L)
        hybrid_mask_4d = hybrid_mask_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)


        '''
        # Attention mask: True for all condition positions and actual tokens, False for padding
        cond_true = torch.ones(end_cond - start_cond, dtype=torch.bool)
        if actual_block_length < self.block_length:
            token_true = torch.ones(actual_block_length, dtype=torch.bool)
            token_false = torch.zeros(self.block_length - actual_block_length, dtype=torch.bool)
            attention_mask = torch.cat([cond_true, token_true, token_false])
        else:
            token_true = torch.ones(self.block_length, dtype=torch.bool)
            attention_mask = torch.cat([cond_true, token_true])
        
        return {
            "in_hidden_states": in_hs,  # [condition_len, hidden_size]
            "input_ids": masked_tokens,  # [block_length]
            "labels": target_tokens,  # [block_length]
            "attention_mask": attention_mask,  # [block_length]
        }


class MTPHeadCollator:
    """
    Custom collator for MTP head dataset.
    
    Handles batching of samples with different sequence lengths.
    """
    
    def __init__(self, pad_token_id: int = 151643):
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of samples.
        
        Args:
            batch: List of samples from dataset
            
        Returns:
            Batched tensors
        """
        # Support samples that may be either dicts or tuples/lists
        in_hs_list = []
        input_ids_list = []
        labels_list = []
        attention_mask_list = []

        for sample in batch:
            if isinstance(sample, dict):
                in_hs_list.append(sample.get("in_hidden_states"))
                input_ids_list.append(sample.get("input_ids"))
                labels_list.append(sample.get("labels"))
                attention_mask_list.append(sample.get("attention_mask"))
            elif isinstance(sample, (list, tuple)):
                # Try to unpack common ordering: (in_hidden_states, input_ids, labels, attention_mask)
                try:
                    in_hs_list.append(sample[0])
                    input_ids_list.append(sample[1])
                    labels_list.append(sample[2])
                    attention_mask_list.append(sample[3])
                except Exception:
                    raise KeyError("Batch sample is a tuple/list but does not follow expected order (in_hidden_states, input_ids, labels, attention_mask)")
            else:
                raise KeyError("Unsupported sample type in batch: expected dict or tuple/list")

        # Convert lists to tensors
        in_hidden_states = torch.stack(in_hs_list)
        input_ids = torch.stack(input_ids_list)
        labels = torch.stack(labels_list)
        attention_mask = torch.stack(attention_mask_list)
        
        return {
            "in_hidden_states": in_hidden_states,  # [batch_size, hidden_size]
            "input_ids": input_ids,  # [batch_size, block_length]
            "labels": labels,  # [batch_size, block_length]
            "attention_mask": attention_mask,  # [batch_size, block_length]
        }


def create_dataloaders(
    data_dir: str,
    block_length: int = 4,
    batch_size: int = 32,
    num_workers: int = 2,
    train_ratio: float = 0.9,
    shuffle: bool = True,
    mask_token_id: int = 151666,
    pad_token_id: int = 151643,
    per_subdir: bool = False,
    test_mode: bool = False,
    condition_len: int = 2,
) -> tuple:
    """
    Create train and validation dataloaders.
    
    Args:
        data_dir: Directory containing collected data
        block_length: Number of tokens to predict per sample
        batch_size: Batch size
        num_workers: Number of data loading workers
        train_ratio: Ratio of data for training (rest for validation)
        shuffle: Whether to shuffle training data
        mask_token_id: Token ID for [MASK]
        pad_token_id: Token ID for padding
        per_subdir: If True, load each subdirectory separately
        test_mode: If True, use only first hidden state + next block_length tokens (for debugging)
        
    Returns:
        Tuple of (train_dataloader, val_dataloader, dataset)
    """
    p = Path(data_dir)

    def _load_from_dir(p_dir: Path):
        hs_path = p_dir / "hidden_states.pt"
        tk_path = p_dir / "tokens.pt"
        if not (hs_path.exists() and tk_path.exists()):
            raise FileNotFoundError(f"missing hidden_states.pt or tokens.pt in {p_dir}")
        return torch.load(hs_path), torch.load(tk_path)

    # If user requests per-subdir loading, return a list of dataloaders per child dir
    if per_subdir:
        results = []
        # consider direct dir too if it contains tensors
        dirs_to_scan = []
        if (p / "hidden_states.pt").exists() and (p / "tokens.pt").exists():
            dirs_to_scan.append(p)
        # then child dirs
        dirs_to_scan.extend(sorted([c for c in p.iterdir() if c.is_dir()]))

        for d in dirs_to_scan:
            try:
                hs, tk = _load_from_dir(d)
            except FileNotFoundError:
                continue

            ds = MTPHeadDataset(
                data_dir=str(d),
                block_length=block_length,
                mask_token_id=mask_token_id,
                pad_token_id=pad_token_id,
                hidden_states_tensor=hs,
                tokens_tensor=tk,
                test_mode=test_mode,
                condition_len=condition_len,
            )

            total_size = len(ds)
            train_size = int(total_size * train_ratio)
            val_size = total_size - train_size

            train_dataset, val_dataset = torch.utils.data.random_split(
                ds,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )

            collator = MTPHeadCollator(pad_token_id=pad_token_id)

            train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                collate_fn=collator,
            )

            val_dataloader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collator,
            )

            results.append((str(d), train_dataloader, val_dataloader, ds))

        return results

    # default behavior: aggregate all child dirs or load direct tensors
    # If tensors are directly under data_dir, load them; otherwise concatenate child dirs
    if (p / "hidden_states.pt").exists() and (p / "tokens.pt").exists():
        hidden_states, tokens = _load_from_dir(p)
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
            tokens = tokens.unsqueeze(0)
    else:
        hs_list = []
        tk_list = []
        for child in sorted([c for c in p.iterdir() if c.is_dir()]):
            if (child / "hidden_states.pt").exists() and (child / "tokens.pt").exists():
                hs_child = torch.load(child / "hidden_states.pt")
                tk_child = torch.load(child / "tokens.pt")
                # Expand to 3D if needed [seq_len, hidden_size] -> [1, seq_len, hidden_size]
                if hs_child.dim() == 2:
                    hs_child = hs_child.unsqueeze(0)
                    tk_child = tk_child.unsqueeze(0)
                hs_list.append(hs_child)
                tk_list.append(tk_child)
        if not hs_list:
            raise FileNotFoundError(f"No hidden_states.pt / tokens.pt found in {data_dir} or its subdirs")
        hidden_states = torch.cat(hs_list, dim=0)
        tokens = torch.cat(tk_list, dim=0)

    dataset = MTPHeadDataset(
        data_dir=data_dir,
        block_length=block_length,
        mask_token_id=mask_token_id,
        pad_token_id=pad_token_id,
        hidden_states_tensor=hidden_states,
        tokens_tensor=tokens,
        test_mode=test_mode,
        condition_len=condition_len,
    )
    
    # Split dataset
    total_size = len(dataset)

    collator = MTPHeadCollator(pad_token_id=pad_token_id)
    
    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
    )

    
    return train_dataloader, dataset


if __name__ == "__main__":
    # Test dataset loading
    import sys
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
        
        dataset = MTPHeadDataset(data_dir)
        sample = dataset[0]
        
        print("Sample keys:", sample.keys())
        for k, v in sample.items():
            print(f"  {k}: {v.shape}")
        
        # Test dataloader
        train_dl, val_dl, _ = create_dataloaders(data_dir, batch_size=4)
        
        batch = next(iter(train_dl))
        print("\nBatch keys:", batch.keys())
        for k, v in batch.items():
            print(f"  {k}: {v.shape}")

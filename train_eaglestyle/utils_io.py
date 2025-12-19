"""I/O helpers: read model config, save/load collected data and JSON helpers.

Usage examples:
    from utils_io import get_hidden_size_from_model_dir, save_collected_data, load_collected_data

    hs = get_hidden_size_from_model_dir('/path/to/model')
    save_collected_data('out_dir', hidden_states_tensor, tokens_tensor, input_ids_tensor)
    hs, tokens, input_ids = load_collected_data('out_dir')
"""
import json
import os
from typing import Optional, Tuple

import torch

def get_hidden_size_from_model_dir(model_dir: str) -> Optional[int]:
    """Return hidden_size from model_dir/config.json.

    Tries to open config.json first; if not found, raises FileNotFoundError.
    """
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"config.json not found in {model_dir}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    # common keys: hidden_size, n_embd
    return cfg.get("hidden_size") or cfg.get("n_embd")


def save_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_collected_data(
    out_dir: str,
    hidden_states: torch.Tensor,
    tokens: torch.Tensor,
    input_ids: Optional[torch.Tensor] = None,
) -> None:
    """Save collected tensors to `out_dir` as torch files.

    Files: hidden_states.pt, tokens.pt, (optional) input_ids.pt
    """
    os.makedirs(out_dir, exist_ok=True)
    torch.save(hidden_states, os.path.join(out_dir, "hidden_states.pt"))
    torch.save(tokens, os.path.join(out_dir, "tokens.pt"))
    if input_ids is not None:
        torch.save(input_ids, os.path.join(out_dir, "input_ids.pt"))


def load_collected_data(data_dir: str) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Load saved collected tensors from `data_dir`.

    Returns (hidden_states, tokens, input_ids_or_None)
    """
    hidden_states = torch.load(os.path.join(data_dir, "hidden_states.pt"))
    tokens = torch.load(os.path.join(data_dir, "tokens.pt"))
    input_ids_path = os.path.join(data_dir, "input_ids.pt")
    input_ids = torch.load(input_ids_path) if os.path.exists(input_ids_path) else None
    return hidden_states, tokens, input_ids

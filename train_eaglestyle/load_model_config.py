"""CLI: read hidden_size from a model directory's config.json and optionally save it.

Usage:
    python load_model_config.py /share/public/public_models/Qwen2.5-7B-Instruct --out out.json
"""
import argparse
import json
from pathlib import Path

from utils_io import get_hidden_size_from_model_dir, save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_dir", type=str, help="Path to model directory containing config.json")
    p.add_argument("--out", type=str, default=None, help="Optional output JSON file to save hidden_size")
    args = p.parse_args()

    try:
        hs = get_hidden_size_from_model_dir(args.model_dir)
    except Exception as e:
        print(f"Error reading config: {e}")
        raise

    print(f"hidden_size: {hs}")
    if args.out:
        save_json(args.out, {"hidden_size": hs})
        print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()

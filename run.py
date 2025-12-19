# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# from model.mtp_model import MTPModel

# import torch
# model = MTPModel.from_pretrained(
#     # base_model_path="/share/public/public_models/Qwen3-8B",
#     base_model_path="/share/public/public_models/Qwen2.5-7B-Instruct",
#     ea_model_path="/share/wanghanzhen/.cache/huggingface/hub/models--Dream-org--Dream-v0-Instruct-7B/snapshots/05334cb9faaf763692dcf9d8737c642be2b2a6ae",
#     torch_dtype=torch.float16,
#     low_cpu_mem_usage=True,
#     device_map="auto",
#     total_token=-1
# )
# model.eval()

# prompt = "Give me a short introduction to large language model."
# messages = [
#     {"role": "user", "content": prompt}
# ]
# text = model.tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True,
#     enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
# )
# print(f"input:{text}")
# input_ids = model.tokenizer([text]).input_ids
# # input_ids=model.tokenizer([prompt]).input_ids
# input_ids = torch.as_tensor(input_ids).cuda()
# if mtp_mode:
#     output_ids=model.mtp_generate(input_ids,temperature=0.5,max_length=4096)
# else:
#     output_ids=model.naivegenerate(input_ids,temperature=0.5,max_length=4096)

# output=model.tokenizer.decode(output_ids[0])
# print(output)

import os
import sys
import argparse
import time
import torch

sys.path.append("/share/wanghanzhen/MTP")  # 替换为包含 model/ 的父目录

from model.mtp_model import MTPModel

def main():
    parser = argparse.ArgumentParser(description="Run MTP or naive generation with speed measurement.")
    parser.add_argument(
        "--mtp-mode", 
        action="store_true", 
        help="Enable MTP speculative decoding mode. If not set, use naive autoregressive generation."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Write an outline for a short 100-word blog post about why Christmas Cactus are a great buy.",
        help="Input prompt for generation."
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum length of generated sequence."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature."
    )
    args = parser.parse_args()

    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"

    # Load model
    print("Loading model...")
    model = MTPModel.from_pretrained(
        base_model_path="/share/public/public_models/Qwen2.5-7B-Instruct",
        # ea_model_path="/share/wanghanzhen/.cache/huggingface/hub/models--Dream-org--Dream-v0-Instruct-7B/snapshots/05334cb9faaf763692dcf9d8737c642be2b2a6ae",
        ea_model_path="/share/wanghanzhen/MTP/dMTP/train_eaglestyle/mtp_checkpoints_whead/checkpoint-6300",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        total_token=-1
    )
    model.eval()
    print("Model loaded.")

    # Prepare input
    # messages = [{"role": "user", "content": args.prompt}]
    # text = model.tokenizer.apply_chat_template(
    #     messages,
    #     tokenize=False,
    #     add_generation_prompt=True,
    #     enable_thinking=True
    # )

    # encodings = model.tokenizer(
    #     args.prompt,
    #     return_tensors="pt",
    # )
    print(f"\n[Input]\n{args.prompt}\n")

    input_ids = model.tokenizer([args.prompt]).input_ids
    input_ids = torch.as_tensor(input_ids).cuda()

    # Warm-up run (optional but recommended for timing)
    with torch.no_grad():
        if args.mtp_mode:
            _ = model.mtp_generate(input_ids, temperature=args.temperature, max_length=max(256, args.max_length))
        else:
            _ = model.naivegenerate(input_ids, temperature=args.temperature, max_length=max(256, args.max_length))

    torch.cuda.synchronize()
    start_time = time.time()

    # Actual generation
    with torch.no_grad():
        if args.mtp_mode:
            output_ids = model.mtp_generate(input_ids, temperature=args.temperature, max_length=args.max_length)
        else:
            output_ids = model.naivegenerate(input_ids, temperature=args.temperature, max_length=args.max_length)

    torch.cuda.synchronize()
    end_time = time.time()

    

    # Decode and print
    output = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    generated_text = output[len(args.prompt):].strip()

    # Speed metrics
    gen_tokens = len(model.tokenizer(generated_text).input_ids)
    gen_time = end_time - start_time
    speed = gen_tokens / gen_time if gen_time > 0 else 0

    print("=" * 60)
    print("[Generated Output]")
    print(generated_text)
    print("\n" + "=" * 60)
    print(f"Mode: {'MTP' if args.mtp_mode else 'Naive'}")
    print(f"Time: {gen_time:.2f} s | Tokens: {gen_tokens} | Speed: {speed:.2f} tok/s")

if __name__ == "__main__":
    main()
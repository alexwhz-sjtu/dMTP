import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 将项目根目录加入 sys.path
sys.path.append(parent_dir)
from model.dspec_model_test import DSPModel
from fastchat.model import get_conversation_template
from fastchat.llm_judge.common import load_questions
import torch
import argparse
from tqdm import tqdm
import time
import shortuuid
import json
import numpy as np

model_id = "Qwen3-8B"
if "Qwen3-14B" in model_id:
    base_model_path="/share/public/public_models/Qwen3-14B"
if "Qwen3-8B" in model_id:
    base_model_path="/share/public/public_models/Qwen3-8B"
if "Qwen3-32B" in model_id:
    base_model_path="/share/public/public_models/Qwen3-32B"
if "Qwen2.5-32B" in model_id:
    base_model_path="/share/public/public_models/Qwen2.5-32B-Instruct"


ea_model_path="/share/public/wanghanzhen/.cache/huggingface/hub/models--Dream-org--Dream-v0-Instruct-7B/snapshots/05334cb9faaf763692dcf9d8737c642be2b2a6ae"


if "Dream" in ea_model_path:
    ea_model_type = "Dream-7B"
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench-name",
        type=str,
        default="spec-bench",
        # default="mt-bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-file",
        type=str,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--total-token",
        type=int,
        default=60,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=8,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )

    args = parser.parse_args()

    
    model_id = model_id + "-" + ea_model_type + "-temperature-" + str(args.temperature)
    answer_file = f"/share/public/wanghanzhen/SpeculativeDecoding/d-Spec/model_answer/{args.bench_name}/{model_id}-max_length_{args.max_new_tokens}.jsonl"
    log_file = f"/logger/{args.bench_name}/{model_id}-max_length_{args.max_new_tokens}.log"
    print(f"Output to {answer_file}")

    model = DSPModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=ea_model_path,
        depth=args.depth,
        total_token=args.total_token,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.eval()

    # question_file = "/share/public/wanghanzhen/SpeculativeDecoding/d-Spec/dataset/LongWriter/evaluation/longbench_write_en.jsonl"
    question_file = "/share/public/wanghanzhen/SpeculativeDecoding/d-Spec/dataset/spec_bench/question.jsonl"
    question_begin = args.question_begin
    question_end = args.question_end
    bench_name = args.bench_name    

    # -----------------------------------------------
    #                  evaluation
    # -----------------------------------------------
    questions = load_questions(question_file, question_begin, question_end)
    # print(questions)

    ans_handles = []
    question = questions[0]
    # warmup
    for _ in range(3):
        torch.manual_seed(0)
        prompt = "Write an outline for a short 100-word blog post about why Christmas Cactus are a great buy."
        inputs = model.tokenizer([prompt], return_tensors="pt").to("cuda")
        input_ids = inputs.input_ids
 
        torch.cuda.synchronize()
        start_time = time.time()

        output_ids, new_token, step, accept_length_tree, _, _ , _ = model.dsp_generate(input_ids,
                            temperature=args.temperature,
                            max_new_tokens=512,
                            log=True)

        torch.cuda.synchronize()
        total_time = time.time() - start_time

    print('Warmup done')


    print(f"====== Evaluating on {bench_name} ======")

    total_time = 0.0
    total_length = 0

    for question in tqdm(questions):
        if bench_name == "spec-bench":
            range_end = len(question["turns"])
            qs_label = "turns"
        elif bench_name == "longwriter":
            qs_label = "prompt"
            range_end = 1
            
        accept_lengths_tree = []
        cur_accept_lengths_tree = []
        turns = []
        steps = []
        new_tokens = []
        wall_time = []
        choices = []
        output = None
        generate_length = 0
        torch.cuda.synchronize()
        start_time = time.time()

        for j in range(0, range_end):
            
            if bench_name == "spec-bench":
                prompt = question[qs_label][j]
            elif bench_name == "longwriter":
                prompt = question[qs_label]
                
            if "Qwen" in model_id:
                messages = [
                    {"role": "user", "content": prompt}
                ]
                text = model.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True if "think" in model_id else False, # Switches between thinking and non-thinking modes. Default is True.
                )

                if output is not None:
                    text = output + "\n" + text

                input_ids = model.tokenizer([text]).input_ids
                input_ids = torch.as_tensor(input_ids).cuda()

            # if "Llama" in model_id:
            #     conv
                
           
            output_ids, new_token, step, accept_length_tree, target_time, draft_time, tree_time  = model.dsp_generate(input_ids,
                                                temperature=args.temperature,
                                                max_new_tokens=args.max_new_tokens,
                                                log=True)
            output=model.tokenizer.decode(output_ids[0])
            

            accept_lengths_tree.extend(accept_length_tree)
            output_ids = output_ids[0][len(input_ids[0]):]
            output = model.tokenizer.decode(
                    output_ids,
                    spaces_between_special_tokens=False,
                )
            
            steps.append(int(step))
            generate_length += int(new_token)
            cur_accept_lengths_tree.extend(accept_length_tree)
            turns.append(output)
            print(f"length:{generate_length} | {target_time, draft_time, tree_time}")

        torch.cuda.synchronize()
        qs_time = time.time() - start_time
        total_time += qs_time
        total_length += generate_length

        choices.append({"index": 1, "turns": turns, "decoding_steps": steps, "new_tokens": generate_length, "total_time": qs_time, "large model time":target_time, "draft model time":draft_time, "accept_lengths": cur_accept_lengths_tree, "speed": round(generate_length / qs_time, 4)})
            
        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {}
            qs_ans_json = {
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "avg accept length": np.mean(cur_accept_lengths_tree),
            }
            for key in ["question_id", "category", "type"]:
                if key in question:
                    ans_json[key] = question[key]

            ans_json.update(qs_ans_json)
            fout.write(json.dumps(ans_json) + "\n")

    total_speed = round(total_length / total_time, 4)

    with open(os.path.expanduser(answer_file), "a") as fout:
            fout.write("\n MLA:" + np.mean(accept_lengths_tree) + "\n" + "speed" + str(total_speed) + "tokens/s\n")



    # prompt = "Give me a short introduction to large language model."
    # messages = [
    #     {"role": "user", "content": prompt}
    # ]
    # text = model.tokenizer.apply_chat_template(
    #     messages,
    #     tokenize=False,
    #     add_generation_prompt=True,
    #     enable_thinking=True if "think" in model_id else False, # Switches between thinking and non-thinking modes. Default is True.
    # )
    # print(f"input:{text}")
    # input_ids = model.tokenizer([text]).input_ids
    # # input_ids=model.tokenizer([prompt]).input_ids
    # input_ids = torch.as_tensor(input_ids).cuda()
    # output_ids=model.dsp_generate(input_ids,temperature=0.5,max_length=4096)
    # output=model.tokenizer.decode(output_ids[0])
    # print(output)
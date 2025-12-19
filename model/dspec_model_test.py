import copy
import json
import time

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import os
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig

from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .modeling_mixtral_kv import MixtralForCausalLM as KVMixtralForCausalLM
#from .modeling_qwen2_kv import LlamaForCausalLM as KVQwen2ForCausalLM
from .modeling_qwen2_kv import Qwen2ForCausalLM as KVQwen2ForCausalLM
from .modeling_qwen3_kv import Qwen3ForCausalLM as KVQwen3ForCausalLM
from .utils_test import *
from .kv_cache import initialize_past_key_values, deep_copy_past_key_values

from .diffu_drafter_test import DreamModel as Model1
from .diffu_drafter_test import DreamPreTrainedModel
from .d_config import DConfig
from Dream_utils.generation_utils import DreamGenerationConfig
from torch import cuda


class DSPModel(nn.Module):

    def __init__(
            self,
            use_eagle3,
            base_model,
            base_model_name_or_path,
            d_model_path,
            total_token,
            depth,
            top_k,
            threshold,
    ):

        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path, use_fast=False)
        self.use_eagle3 = use_eagle3
       
        self.total_tokens = total_token
        self.depth = depth
        self.top_k = top_k
        self.threshold = threshold
        self.d_config = DConfig.from_pretrained(d_model_path)
        self.d_generation_config = DreamGenerationConfig.from_model_config(self.d_config)
       
        self.d_model = Model1.from_pretrained(d_model_path, torch_dtype=self.base_model.dtype).to(self.base_model.device)
        

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    @classmethod
    def from_pretrained(
            cls,
            use_eagle3=True,
            base_model_path=None,
            ea_model_path=None,
            total_token=60,
            depth=7,
            top_k=10,
            threshold=1.0,
            **kwargs,
    ):
        # assert Type=="LLaMA" or "Mixtral"
        Type = AutoConfig.from_pretrained(base_model_path).architectures[0]

        if Type == 'LlamaForCausalLM':
            base_model = KVLlamaForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        elif Type == 'Qwen2ForCausalLM':
            base_model = KVQwen2ForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        elif Type == 'Qwen3ForCausalLM':
            base_model = KVQwen3ForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        else:
            base_model = KVMixtralForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )



        # initialize model
        model = cls(
            use_eagle3,
            base_model,
            base_model_path,
            ea_model_path,
            total_token,
            depth,
            top_k,
            threshold
        )

        return model

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            past_key_values=None,
            output_orig=False,
            position_ids=None,
    ):

        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
            hidden_states = outputs[0]

        if output_orig:
            return outputs, orig, hidden_states
        else:
            return outputs, hidden_states

    @torch.no_grad()
    # 大模型生成若干，和扩散模型后续比较
    def dsp_generate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=1024,
            max_length=32000,
            log=False,
            is_llama3=False,

    ):
        
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None



        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        accept_length_list = []

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        cur_length = input_len
        # prefill
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
            input_ids, self, past_key_values, logits_processor
        )
        new_token = 0
        max_length = max_length - self.total_tokens - 10

        total_target_time = 0.0
        total_draft_time = 0.0
        total_tree_time = 0.0

        for idx in range(max_length - 1024):
            draft_tokens = torch.tensor(draft_tokens, device = input_ids.device)
            draft_tokens = draft_tokens.unsqueeze(0)

            # ------------------------ 大模型数次forward --------------------------
            # 创建一个临时的、可扩展的 KV Cache（不要直接修改原 cache）
            if past_key_values is not None:
                temp_past_key_values = copy.deepcopy(past_key_values)  # 浅拷贝 key/value 张量，不共享

            else:
                temp_past_key_values = None


            input_ids_copy = input_ids.clone() 
            top_k = 8
            next_token_id = draft_tokens[0][0].unsqueeze(0).unsqueeze(0)
            print("------------------- one turn -------------------")
            for i in range(0, top_k):
                position_ids = torch.tensor([input_ids_copy.shape[1]], device=input_ids.device)
                if position_ids is not None and position_ids.dim() == 1:
                    position_ids = position_ids.unsqueeze(0)
                outputs, logits, hidden_state = self(
                    next_token_id,
                    output_orig=True,
                    past_key_values=temp_past_key_values,
                    position_ids=position_ids,
                )
                next_token_id = torch.argmax(logits[:,0], dim=-1)
                next_token = self.tokenizer.decode(next_token_id)
                print(f"position:{input_ids_copy.shape[1]+i} | token:{next_token}")
                next_token_id = next_token_id.unsqueeze(0)
                input_ids_copy = torch.cat((input_ids, next_token_id.to(input_ids.device)), dim=1)

                del temp_past_key_values   # 断开引用
                del input_ids_copy


                
            cuda.synchronize()
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            start_time.record()

            # Target model forward, get logits
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )

            end_time.record()
            end_time.synchronize()
            target_time = start_time.elapsed_time(end_time) / 1000  # seconds
            # retrieve_indices=tree_buffers["retrieve_indices"]
            # logits = logits[0, retrieve_indices]
            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]
            # verification
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor, self.tokenizer
            )
            # print(f"accept_length:{accept_length}")
            # Adjusting the input sequence, draft model forward
            input_ids, draft_tokens, retrieve_indices, tree_mask, \
                tree_position_ids, new_token, hidden_state, sample_token, info, draft_time, tree_time = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state_new,
                sample_p
            )

            accept_length_tree = input_ids.shape[1] - cur_length
            cur_length = accept_length_tree + cur_length
            accept_length_list.append(accept_length_tree)
            # print(f"\ncontext: {self.tokenizer.decode(input_ids[0])}\n")
            
            # print("Top 5 tokens and probabilities:")
            # for item in info["top_5"]:
            #     print(f"Token: '{item['text']}' (ID: {item['token_id']}), Probability: {item['probability']:.4f}")

            # print(f"Selected token (large model): '{info['selected_text']}'")
           

            # print(f"step:{idx} | target_time: {target_time:.4f} s, draft_time: {draft_time:.4f} s, tree_time: {tree_time:.4f} s")
            total_target_time += target_time
            total_draft_time += draft_time
            total_tree_time += tree_time

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
        if not log:
            return input_ids
        else:
            return input_ids, new_token, idx+1, accept_length_list, total_target_time, total_draft_time, total_tree_time 


    @torch.no_grad()
    def naivegenerate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=32000,
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.d_model.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]

        outputs = self.base_model(input_ids, past_key_values=past_key_values, use_cache=True)
        new_token = 0
        max_length = max_length - self.d_model.total_tokens - 10
        for idx in range(max_length):
            if logits_processor is not None:
                logits = outputs.logits[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)
            outputs = self.base_model(input_id, use_cache=True, past_key_values=past_key_values)
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            new_token += 1

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
        if not log:
            return input_ids
        else:
            return input_ids, new_token, idx


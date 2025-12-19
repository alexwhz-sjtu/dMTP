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
from .utils import *
from .kv_cache import initialize_past_key_values

from .modeling_mtp_head import DreamModel as Model1
from .modeling_mtp_head import DreamPreTrainedModel
from .mtphead_config import MTPHeadConfig
from torch import cuda
from .utils import DreamGenerationConfig


class MTPModel(nn.Module):

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
        self.d_config = MTPHeadConfig.from_pretrained(d_model_path)
        self.mtp_generation_config = DreamGenerationConfig.from_model_config(self.d_config)
       
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
            use_cache=True,
    ):

        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                use_cache=use_cache,
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
            hidden_states = outputs[0]

        if output_orig:
            return outputs, orig, hidden_states
        else:
            return outputs, hidden_states

    @torch.no_grad()
    # generate by the mtp&base model (d-Spec model)
    def mtp_generate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=1024,
            max_length=32000,
            mtp_length=4,
            log=False,
            is_llama3=False,
            

    ):
        
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None

        input_ids = input_ids.clone()

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

        input_len = input_ids.shape[1]  # prompt length
        new_token = 0
        input_id = None
        test_mode = True

        # sliding window queue of hidden states (each element shape: [B,1,hidden])
        cond_queue = []
        condition_len = 1

        for idx in range(max_length - 10):
            # If we don't yet have enough condition hidden states, use base model to generate next token
            if len(cond_queue) < condition_len:
                # feed either full prompt first, then single-token ids
                feed_ids = input_ids if new_token == 0 else input_id
                logits, hidden_state_new, outputs = get_feature(
                    self,
                    past_key_values,
                    feed_ids,
                    use_cache=True,
                    output_orig=True,
                )

                # sample / greedy for base model
                if logits_processor is not None:
                    step_logits = logits[:, -1]
                    step_logits = logits_processor(None, step_logits)
                    probs = torch.nn.functional.softmax(step_logits, dim=-1)
                    input_id = torch.multinomial(probs, 1)
                else:
                    input_id = logits[:, -1:].argmax(dim=-1)

                # append hidden state of the newly produced token
                last_h = hidden_state_new[:, -1:, :].clone()
                cond_queue.append(last_h)

                # append token to input_ids sequence
                input_ids = torch.cat([input_ids, input_id.to(input_ids.device)], dim=-1)

                new_token += 1

                # continue to next iteration until queue filled
                continue

            # Now cond_queue has at least condition_len items -> use mtp
            cond_tensor = torch.cat(cond_queue[-condition_len:], dim=1)  # [B, condition_len, hidden]

            # mtp predicts next mtp_length tokens
            next_n = mtp_sample(
                self.d_model,
                cond_tensor,
                input_id=None,
                max_length=mtp_length,
                generation_config=self.mtp_generation_config,
                tokenizer=self.tokenizer,
            )
            
            # Get base model hidden states for the tokens just appended (so queue holds base-model hidden states)
            # We feed next_n into base model (with use_cache) to obtain hidden states for those tokens
            # If next_n has multiple tokens, feed them all at once
            logits_n, hidden_state_new_n, outputs_n = get_feature(
                self,
                past_key_values,
                input_id,
                use_cache=True,
                output_orig=True,
            )

            # hidden_state_new_n shape: [B, T, hidden]
            # push each token's hidden into queue and maintain window length
            Tn = hidden_state_new_n.shape[1]
            for t_i in range(Tn):
                h_t = hidden_state_new_n[:, t_i:t_i+1, :].clone()
                cond_queue.append(h_t)
                # keep queue size bounded
                if len(cond_queue) > condition_len:
                    cond_queue.pop(0)

            if test_mode:
                # sample / greedy for base model
                if logits_processor is not None:
                    step_logits = logits_n[:, -1]
                    step_logits = logits_processor(None, step_logits)
                    probs = torch.nn.functional.softmax(step_logits, dim=-1)
                    input_id = torch.multinomial(probs, 1)
                else:
                    input_id = logits_n[:, -1:].argmax(dim=-1)

                # check original output
                input_ids = torch.cat(
                    [input_ids, input_id.to(input_ids.device)], dim=-1
                )

                new_token += 1
                
                print(f"\n ==================================== ")
                print(f"next n:{self.tokenizer.decode(next_n[0])}")
                print(f"origin:{self.tokenizer.decode(input_ids[0])}")
            else:
                # Update input. concat new-generated tokens to current sequence
                input_ids = torch.cat(
                    [input_ids, next_n.to(input_ids.device)], dim=-1
                )

                new_token += next_n.shape[1]

            # termination checks using latest appended tokens
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
            return input_ids, new_token, idx+1


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
        # self.d_model.reset_kv()

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
        max_length = max_length - 10
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


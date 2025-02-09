
## Builder_twolora.py
import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from . import *
from ..constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

@dataclass
class ModelVisonArguments:
    vision_tower: Optional[str] = field(default='openai/clip-vit-large-patch14-336')
    mm_vision_select_layer: Optional[int] = field(default=-2)
    pretrain_mm_mlp_adapter: Optional[str] = field(default='hugging_cache/llava-v1.5-7b/mm_projector.bin')
    mm_projector_type: Optional[str] = field(default='mlp2x_gelu')
    mm_vision_select_feature: Optional[str] = field(default="patch")

def load_pretrained_model(
    model_path,
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    device="cuda",
    use_lora=False,
    lora_rank=8,
    lora_alpha=32,
    lora_dropout=0.1,
    lora_target_modules=["down_proj", "up_proj"],
    connector_type: Optional[str] = None, 
    for_eval: Optional[bool] = None,
    adapter_path: Optional[str] = None,
    **kwargs
) -> LlavaLlamaForCausalLM:
    
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda": # 여기 
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else: # 여기 
        kwargs['torch_dtype'] = torch.float16

    ## Load LLaVA model(기본) 
    #1) LoRA 관련 키 제거
    for key in ["lora_r", "lora_alpha", "lora_dropout", "lora_target_modules", "use_lora", "inner_params"]:
        if key in kwargs:
            print(kwargs.pop(key))

    #2) 기존 모델 로딩
    model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    use_two_lora = True # two lora
    #3) LoRA 적용
    if use_lora:
        from peft import get_peft_model, LoraConfig, TaskType

        mix_lora = True
        if use_two_lora: # two lora
            lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM, # 
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=lora_target_modules
                )
            if not mix_lora: # peft        
                model = get_peft_model(model, lora_config)
                model.add_adapter(peft_config=lora_config, adapter_name = "visual")
                model.add_adapter(peft_config=lora_config, adapter_name = "textual")
                model.delete_adapter("default")

            else: # ★★ PeftMixedmodel 사용: Connector사용시, PeftMixedModel 사용 필수 ★★

                from peft import PeftMixedModel
                if for_eval: # test(load pretrained weights) | optional: adapter - init from scratch(구현 필요)
                    model = PeftMixedModel.from_pretrained(model, os.path.join(adapter_path, "visual"), "visual")
                    model.load_adapter(os.path.join(adapter_path, "textual"), adapter_name="textual")
                    
                    if connector_type:
                        model.load_adapter(os.path.join(adapter_path, "connector"), adapter_name="connector")


                else: # train(add new adapter)
                    model = PeftMixedModel(model, lora_config, adapter_name="visual")
                    model.add_adapter(peft_config=lora_config, adapter_name="textual")

                    if connector_type: # connector 설정
                        if connector_type == "ffn":
                            connector_config = LoraConfig(
                                task_type=TaskType.CAUSAL_LM,
                                r=lora_rank,
                                lora_alpha=16,
                                lora_dropout=0.1,
                                target_modules=lora_target_modules
                            )
                        elif connector_type == "attention":
                            connector_config = LoraConfig(
                                task_type=TaskType.CAUSAL_LM,
                                r=lora_rank,
                                lora_alpha=16,
                                lora_dropout=0.1,
                                target_modules=["q_proj", "k_proj"]
                            )

                        model.add_adapter(peft_config=connector_config, adapter_name="connector")
                        print("-> Connector 장착 완료")



        else: 
            model = get_peft_model(model, lora_config)

        print("-> L O R A 장 착 완 료")

    # initialize vision modeles(or Load ViT?)
    model_args = ModelVisonArguments()
    model.get_model().initialize_vision_modules(model_args)
    return model

## LLAVA -> !LORA! , FT 
import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
import torch.nn as nn
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

# ----------------------------------------
# Custom LoRA modules and Connector modules
# ----------------------------------------

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha):
        super(LoRALayer, self).__init__()
        self.rank = rank
        self.alpha = alpha
        # Low-rank matrices A and B
        self.A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.B = nn.Parameter(torch.randn(out_features, rank) * 0.01)
    
    def forward(self, x):
        delta = (x @ self.A.t()) @ self.B.t() * (self.alpha / self.rank)
        return delta

class OneAdapterConnector(nn.Module):
    """
    하나의 어댑터(LoRA, rank=16 등)를 사용하여 기존 선형 계층 출력과 LoRA 업데이트(Δ)를 합산한 후,
    MLP를 통해 후처리하여 compositional edit 지식을 생성하는 모듈.
    - if fusion=True 인 경우 MLP를 통해 후처리하여 compositional edit 지식을 생성하고,
    - if fusion=False 인 경우 단순히 base + Δ를 반환하는 모듈
    """
    def __init__(self, base_linear: nn.Linear, rank=16, alpha=32, mlp_hidden_dim=128):
        super(OneAdapterConnector, self).__init__()
        self.fusion = False # fowrad시, mlp를 사용할지 말지 선택하기위한 flag
        self.base_linear = base_linear
        in_features = base_linear.in_features
        out_features = base_linear.out_features

        self.lora = LoRALayer(in_features, out_features, rank=rank, alpha=alpha)
        self.mlp = nn.Sequential(
            nn.Linear(out_features, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, out_features)
        )
    def use_connector(self):
        self.fusion = True
    
    def set_default(self):
        self.fusion = False

    def forward(self, x):
        base_out = self.base_linear(x)
        delta = self.lora(x)

        if self.fusion:
            delta = self.mlp(delta)
        
        updated = base_out + delta
        return updated

class TwoAdapterConnector(nn.Module):
    """
    두 개의 어댑터(각각 LoRA, 예: rank=8)를 사용하여, 
    시각 편집용과 텍스트 편집용 업데이트를 분리한 후,
    fusion=True 인 경우 두 업데이트 값을 concat하여 MLP를 통해 융합하여 compositional edit 지식을 생성하고,
    fusion=False 인 경우 간단히 두 업데이트의 평균((visual+textual)/2)을 이용하여 반환하는 모듈.
    """
    def __init__(self, base_linear: nn.Linear, rank=8, alpha=32, mlp_hidden_dim=128):
        super(TwoAdapterConnector, self).__init__()
        self.mode = None

        self.base_linear = base_linear
        in_features = base_linear.in_features
        out_features = base_linear.out_features

        self.lora_visual = LoRALayer(in_features, out_features, rank=rank, alpha=alpha)
        self.lora_textual = LoRALayer(in_features, out_features, rank=rank, alpha=alpha)
        self.mlp = nn.Sequential(
            nn.Linear(2 * out_features, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, out_features)
        )
    
    def use_vis_adapter(self):
        self.mode = "visual"
    def use_text_adapter(self):
        self.mode = "textual"
    def use_connector(self):
        self.mode = "fusion"
    def set_default(self):
        self.mode = None

    def forward(self, x):
        mode = self.mode

        base_out = self.base_linear(x)
        if mode == "visual":
            delta = self.lora_visual(x)
        elif mode == "textual":
            delta = self.lora_textual(x)
        elif mode == "fusion":
            delta_v = self.lora_visual(x)
            delta_t = self.lora_textual(x)
            concat = torch.cat([delta_v, delta_t], dim=-1)
            delta = self.mlp(concat)

        else:
            raise ValueError("Unsupported mode. Choose 'visual', 'textual', or 'fusion'.")
        updated = base_out + delta
        return updated

# ----------------------------------------
# Custom module replacement function
# ----------------------------------------
def replace_target_modules_custom(model: torch.nn.Module, target_module_names: List[str],
                                  connector_type: str = "one",  # "one" or "two"
                                  rank: int = 16, alpha: int = 32, mlp_hidden_dim: int = 128): # size: 1024~2048 적당 예상
    """
    모델 내부의 모든 자식 모듈을 재귀적으로 순회하면서,
    이름에 target_module_names (예: ["down_proj", "up_proj"])가 포함되고 nn.Linear인 경우
    connector_type에 따라 OneAdapterConnector 혹은 TwoAdapterConnector로 교체.
    """
    for name, module in list(model.named_children()):
        replace_target_modules_custom(module, target_module_names, connector_type, rank, mlp_hidden_dim)
        if any(t in name for t in target_module_names) and isinstance(module, nn.Linear):
            if connector_type == "one":
                new_module = OneAdapterConnector(module, rank=rank, alpha=alpha, mlp_hidden_dim=mlp_hidden_dim)
            elif connector_type == "two":
                new_module = TwoAdapterConnector(module, rank=rank, alpha=alpha, mlp_hidden_dim=mlp_hidden_dim)
            else:
                raise ValueError("Unsupported connector_type. Use 'one' or 'two'.")
            setattr(model, name, new_module)
            #print(f"Replaced module '{name}','{module}' with {connector_type} adapter connector.")

# ----------------------------------------
# load_pretrained_model 수정 (builder.py 내)
# ----------------------------------------
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

    # !!!!!!!!!!! Load LLaVA model(기본) !!!!!!!!!!!!!!!
    # Remove LoRA-related keys from kwargs
    for key in ["lora_r", "lora_alpha", "lora_dropout", "lora_target_modules", "use_lora", "inner_params"]:
        if key in kwargs:
            print(f"Removing key: {kwargs.pop(key)}")

    ### 기존 모델 로딩
    model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    
    # !!LoRA 적용
    if use_lora:
        if connector_type is None: # LoRA만 사용 
            from peft import get_peft_model, LoraConfig, TaskType
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules
            )
            model = get_peft_model(model, lora_config)
            print("-> L O R A 장 착 완 료")
        else:
            # Custom connector 방식 적용: 여기서 connector_type는 "one" 또는 "two"
            replace_target_modules_custom(model, target_module_names=lora_target_modules,
                                          connector_type=connector_type,
                                          rank=lora_rank, alpha=lora_alpha, mlp_hidden_dim=128)
            print(f"-> Custom Connector ({connector_type} adapter) 적용 완료")

    # initialize vision modeles(or Load ViT?)
    model_args = ModelVisonArguments()
    model.get_model().initialize_vision_modules(model_args)
    return model
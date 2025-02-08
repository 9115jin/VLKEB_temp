import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

from . import *
from ..constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

# ----------------------------------------
# Model Vision Arguments
# ----------------------------------------
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
    단일 LoRA 어댑터를 사용하여, 기본 선형 계층 출력에 LoRA 업데이트를 더한 후,
    fusion 모드일 경우 MLP를 통해 후처리하여 compositional edit 지식을 생성하는 모듈.
    """
    def __init__(self, base_linear: nn.Linear, rank=16, alpha=32, mlp_hidden_dim=128):
        super(OneAdapterConnector, self).__init__()
        # fusion 모드 적용 여부
        self.fusion = False  
        self.base_linear = base_linear
        in_features = base_linear.in_features
        out_features = base_linear.out_features

        self.lora = LoRALayer(in_features, out_features, rank=rank, alpha=alpha)
        self.mlp = nn.Sequential(
            nn.Linear(out_features, mlp_hidden_dim),
            nn.ReLU(),  # 추후 GELU 등으로 변경 가능
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
    두 개의 LoRA 어댑터 (visual과 textual)를 사용하여, 
    모드에 따라 별도로 업데이트하거나, 두 업데이트를 융합하는 모듈.
    """
    def __init__(self, base_linear: nn.Linear, rank=8, alpha=32, mlp_hidden_dim=128):
        super(TwoAdapterConnector, self).__init__()
        self.mode = None  # "visual", "textual", "fusion" 중 하나로 설정됨

        self.base_linear = base_linear
        in_features = base_linear.in_features
        out_features = base_linear.out_features

        self.lora_visual = LoRALayer(in_features, out_features, rank=rank, alpha=alpha)
        self.lora_textual = LoRALayer(in_features, out_features, rank=rank, alpha=alpha)
        self.mlp = nn.Sequential(
            nn.Linear(2 * out_features, mlp_hidden_dim),
            nn.ReLU(),  # 필요시 GELU 등 다른 활성화 함수로 대체 가능
            nn.Linear(mlp_hidden_dim, out_features)
        )
    
    def use_vis_adapter(self):
        self.mode = "visual"
    
    def use_text_adapter(self):
        self.mode = "textual"
    
    def use_connector(self):
        self.mode = "fusion"
    
    def set_default(self):
        self.mode = "default"

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
        elif mode == "default":
            delta = torch.zeros_like(base_out)
        else:
            raise ValueError("Unsupported mode. Choose 'visual', 'textual', or 'fusion'.")
        updated = base_out + delta
        return updated

# ----------------------------------------
# Custom module replacement function
# ----------------------------------------
def replace_target_modules_custom(model: torch.nn.Module, target_module_names: List[str],
                                  connector_type: str = "one",  # "one" or "two"
                                  rank: int = 16, alpha: int = 32, mlp_hidden_dim: int = 128):
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
            #print(f"Replaced module '{name}' with {connector_type} adapter connector.")

# ----------------------------------------
# 모델 모드 전환 함수
# ----------------------------------------
def set_edit_mode(model: torch.nn.Module, mode: str):
    """
    모델 내부의 모든 커스텀 어댑터(OneAdapterConnector, TwoAdapterConnector)에 대해 편집 모드를 설정합니다.
    mode 인자는 "visual", "textual", "fusion", "default" 중 하나를 사용합니다.
    """
    for module in model.modules():
        if isinstance(module, TwoAdapterConnector):
            if mode == "visual":
                module.use_vis_adapter()
            elif mode == "textual":
                module.use_text_adapter()
            elif mode == "fusion":
                module.use_connector()
            elif mode == "default":
                module.set_default()
            else:
                raise ValueError("Unsupported mode. Use 'visual', 'textual', 'fusion', or 'default'.")
        elif isinstance(module, OneAdapterConnector):
            if mode == "fusion":
                module.use_connector()
            elif mode == "default":
                module.set_default()
            # OneAdapterConnector는 단일 업데이트만 지원하므로 visual/textual 구분은 없으며,
            # fusion 모드일 때만 MLP 후처리 적용합니다.

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
) -> 'LlavaLlamaForCausalLM':
    
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
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
    else:
        kwargs['torch_dtype'] = torch.float16

    # !!!!!!!!!!! Load LLaVA 모델(기본) !!!!!!!!!!!!!!!
    # LoRA 관련 키 제거
    for key in ["lora_r", "lora_alpha", "lora_dropout", "lora_target_modules", "use_lora", "inner_params"]:
        if key in kwargs:
            print(f"Removing key: {kwargs.pop(key)}")

    ### 기존 모델 로딩
    model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    
    # !!LoRA 적용
    if use_lora:
        if connector_type is None:
            # PEFT의 기본 LoRA 적용
            from peft import get_peft_model, LoraConfig, TaskType
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules
            )
            model = get_peft_model(model, lora_config)
            print("-> L O R A 적용 완료 (PEFT 기본 방식)")
        else:
            # Custom Connector 방식 적용 (connector_type는 "one" 또는 "two")
            replace_target_modules_custom(model, target_module_names=lora_target_modules,
                                          connector_type=connector_type,
                                          rank=lora_rank, alpha=lora_alpha, mlp_hidden_dim=128)
            print(f"-> Custom Connector ({connector_type} adapter) 적용 완료")

    # initialize vision modules (또는 ViT 로딩)
    model_args = ModelVisonArguments()
    model.get_model().initialize_vision_modules(model_args)
    return model














# import os
# import warnings
# import shutil

# from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
# import torch
# import torch.nn as nn
# from dataclasses import dataclass, field
# from typing import Dict, Optional, Sequence, List

# from . import *
# from ..constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

# @dataclass
# class ModelVisonArguments:
#     vision_tower: Optional[str] = field(default='openai/clip-vit-large-patch14-336')
#     mm_vision_select_layer: Optional[int] = field(default=-2)
#     pretrain_mm_mlp_adapter: Optional[str] = field(default='hugging_cache/llava-v1.5-7b/mm_projector.bin')
#     mm_projector_type: Optional[str] = field(default='mlp2x_gelu')
#     mm_vision_select_feature: Optional[str] = field(default="patch")

# # --- Custom Adapter Modules (Custom LoRA with optional MLP connector) ---
# # 여기서는 PEFT 내부의 기본 LoRA 모듈(LoraLayer)을 확장하여, 추가 MLP 후처리를 적용하는 버전을 작성합니다.
# # (아래 코드는 peft.tuners.lora.model.LoraLayer를 상속받아 CustomLoraLayer를 구현한 예시입니다.)

# import math
# import torch.nn.functional as F

# from peft.tuners.lora.model import LoraLayer as BaseLoraLayer

# class CustomLoraLayer(BaseLoraLayer):
#     """
#     Custom LoRA adapter that optionally passes the LoRA update through an MLP
#     before adding it to the base layer output.
    
#     - 기본 계산: delta = (x @ A^T) @ B^T * scaling
#     - 만약 self.use_connector가 True이면, delta = MLP(delta)
#     """
#     def __init__(self, in_features, out_features, r, lora_alpha, lora_dropout,
#                  init_lora_weights=True, use_rslora=False, use_dora=False,
#                  mlp_hidden_dim=256, use_connector=False):
#         super().__init__(in_features, out_features, r, lora_alpha, lora_dropout,
#                          init_lora_weights=init_lora_weights, use_rslora=use_rslora, use_dora=use_dora)
#         self.use_connector = use_connector
#         self.mlp = nn.Sequential(
#             nn.Linear(out_features, mlp_hidden_dim),
#             nn.ReLU(),
#             nn.Linear(mlp_hidden_dim, out_features)
#         )
    
#     def forward(self, x):
#         # 기존 LoRA update 계산 (PEFT 방식)
#         delta = (x @ self.lora_A.weight.T) @ self.lora_B.weight.T * self.scaling
#         if self.use_connector:
#             delta = self.mlp(delta)
#         return delta

# # Custom module replacement function for custom connector adapter.
# def replace_target_modules_custom(model: torch.nn.Module, target_module_names: List[str],
#                                   connector_type: str = "one",  # "one" (단일 adapter with MLP) 또는 "two" (두 adapter 후 fusion)
#                                   rank: int = 16, alpha: int = 32, mlp_hidden_dim: int = 128):
#     """
#     모델 내부의 모든 자식 모듈을 재귀적으로 순회하면서,
#     이름에 target_module_names (예: ["down_proj", "up_proj"])가 포함되고 nn.Linear인 경우,
#     connector_type에 따라 CustomLoraLayer 기반의 adapter로 교체합니다.
    
#     connector_type에 따라, custom adapter 내의 use_connector 플래그를 설정합니다.
#     예를 들어, "one" (단일 adapter)에서는 use_connector=True (즉, MLP 후처리 적용)
#     또는 "lora" 모드를 사용하고 싶다면 use_connector=False로 설정할 수 있습니다.
#     (여기서는 connector_type이 "one" 또는 "two"로 전달된다고 가정)
#     """
#     for name, module in list(model.named_children()):
#         replace_target_modules_custom(module, target_module_names, connector_type, rank, alpha, mlp_hidden_dim)
#         if any(t in name for t in target_module_names) and isinstance(module, nn.Linear):
#             # 기존 module은 base_linear
#             # 생성할 custom adapter는 PEFT의 base LoRA adapter 대신 CustomLoraLayer를 사용
#             # connector_type에 따라 use_connector 플래그를 설정합니다.
#             # (예: "one" 또는 "two"일 경우, 여기서는 둘 다 추가 MLP를 적용하는 것으로 가정합니다)
#             use_connector = True  # 혹은, 필요에 따라 connector_type 값에 따라 분기
#             new_module = OneAdapterConnector(module, rank=rank, alpha=alpha, mlp_hidden_dim=mlp_hidden_dim)
#             # 기존 OneAdapterConnector는 직접 정의한 버전인데, 여기서 내부의 self.lora를 CustomLoraLayer로 대체합니다.
#             new_module.lora = CustomLoraLayer(
#                 in_features=module.in_features,
#                 out_features=module.out_features,
#                 r=rank,
#                 lora_alpha=alpha,
#                 lora_dropout=0.1,  # 혹은 원래 값 적용
#                 mlp_hidden_dim=mlp_hidden_dim,
#                 use_connector=use_connector
#             )
#             setattr(model, name, new_module)
#     return model

# # --- load_pretrained_model 함수 ---
# def load_pretrained_model(
#     model_path,
#     load_8bit=False,
#     load_4bit=False,
#     device_map="auto",
#     device="cuda",
#     use_lora=False,
#     lora_rank=8,
#     lora_alpha=32,
#     lora_dropout=0.1,
#     lora_target_modules=["down_proj", "up_proj"],
#     # 새로운 옵션: connector_mode가 None이면 기본 PEFT를 사용, 
#     # connector_mode가 "one" 또는 "two" 등으로 지정되면 custom adapter 적용
#     connector_mode: Optional[str] = None,
#     **kwargs
# ) -> LlavaLlamaForCausalLM:
    
#     kwargs = {"device_map": device_map, **kwargs}

#     if device != "cuda":
#         kwargs['device_map'] = {"": device}

#     if load_8bit:
#         kwargs['load_in_8bit'] = True
#     elif load_4bit:
#         kwargs['load_in_4bit'] = True
#         from transformers import BitsAndBytesConfig
#         kwargs['quantization_config'] = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_compute_dtype=torch.float16,
#             bnb_4bit_use_double_quant=True,
#             bnb_4bit_quant_type='nf4'
#         )
#     else:
#         kwargs['torch_dtype'] = torch.float16

#     # LoRA 관련 키 제거
#     for key in ["lora_r", "lora_alpha", "lora_dropout", "lora_target_modules", "use_lora", "inner_params"]:
#         if key in kwargs:
#             print(f"Removing key: {kwargs.pop(key)}")

#     # 기존 모델 로딩
#     model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    
#     # LoRA 적용
#     if use_lora:
#         if connector_mode is None:
#             # 기본 PEFT LoRA 적용
#             from peft import get_peft_model, LoraConfig, TaskType
#             lora_config = LoraConfig(
#                 task_type=TaskType.CAUSAL_LM,
#                 r=lora_rank,
#                 lora_alpha=lora_alpha,
#                 lora_dropout=lora_dropout,
#                 target_modules=lora_target_modules
#             )
#             model = get_peft_model(model, lora_config)
#             print("-> L O R A (PEFT 기본 방식) 적용 완료")
#         else:
#             # connector_mode가 지정된 경우, custom adapter 적용
#             replace_target_modules_custom(model, target_module_names=lora_target_modules,
#                                           connector_type=connector_mode,
#                                           rank=lora_rank, alpha=lora_alpha, mlp_hidden_dim=128)
#             print(f"-> Custom Connector ({connector_mode} adapter) 적용 완료")
    
#     # Vision 모듈 초기화 (필요한 경우)
#     model_args = ModelVisonArguments()
#     model.get_model().initialize_vision_modules(model_args)
#     return model
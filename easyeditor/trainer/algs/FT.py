import copy
import logging
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from collections import deque
from tqdm import tqdm, trange

from . import local_nn
from .editable_model import EditableModel
from ..utils import _inner_params, _logits

LOG = logging.getLogger(__name__)


class FT(EditableModel):
    def __init__(self, model, config, model_constructor):
        super().__init__(model, config, model_constructor)

        if not str(self.config.device).startswith('cuda'):
            self.config.device = f'cuda:{self.config.device}'
        self.model = self.model.to(torch.float32)
        self.save_weight = None
        
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state_dict = super().state_dict(
            prefix=prefix, keep_vars=keep_vars
        )  # Get default state dict
        state_dict["model_config"] = self.model.config  # Include model config
        return state_dict

    def load_state_dict(self, state_dict, strict: bool = True):
        config = state_dict["model_config"]
        del state_dict["model_config"]
        if config != self.model.config:
            LOG.info("Loaded model config doesn't match current model config.")
            LOG.info(f"Loaded: {config}")
            LOG.info(f"Current: {self.model.config}")

        res = super().load_state_dict(state_dict, True)
        assert len(res.unexpected_keys) == 0, "Shouldn't have any unexpected keys"
        return res

    # Inference
    def forward(self, *inputs, **kwargs):
        if 'minigpt4' in self.config.model_name.lower() or 'blip' in self.config.model_name.lower() or 'llava' in self.config.model_name.lower():
            if self.config.use_lora and self.config.lora_connector_type in ["attention", "ffn"] : # LoRA의 경우, PeftModelForCasualLM -> LLavaLlamaCasualLM으로 래핑을 벗겨내야됨
                outputs = self.model.base_model(*inputs, **kwargs)
            else:
                outputs = self.model(*inputs, **kwargs) # FT, custom model
        else:
            raise not NotImplementedError("Model not supported")
        return outputs
    
    def outer_parameters(self):
        return None

    # Edit(Update Model)
    def edit(self, batch, condition=None, detach_history=False, return_factors=False, connector_mode=False, mode=None, peft=None):
        self.model.train()
        # if self.save_weight:
        #     self.model.load_state_dict(self.save_weight, strict=False)

        ## 업데이트하고자 하는 파라미터 명시: inner_params / LoRA... ##
        if not self.config.inner_params:  # inner_params가 비어 있는 경우
            if connector_mode: 
                    weights = {
                        n: p
                        for n, p in self.model.named_parameters()
                        if ("connector" in n) #MLP 파라미터만 업데이트
                    }
            else: # without Connector 
                if peft: # for peft 
                    if mode == "visual":
                        # visual 어댑터에 해당하는 파라미터만 선택 (default는 제외)
                        weights = {n: p for n, p in self.model.named_parameters() 
                                if "lora" in n and "visual" in n}
                    elif mode == "textual":
                        # textual 어댑터에 해당하는 파라미터만 선택
                        weights = {n: p for n, p in self.model.named_parameters() 
                                if "lora" in n and "textual" in n}
                    elif mode == "fusion":
                        print("구현 미정 - connector 작성 예정")
                    else: # "one" lora 
                        weights = {
                            n: p
                            for n, p in self.model.named_parameters()
                            if "lora" in n  # 기존 방식: LoRA 파라미터만 업데이트
                        }
                    
                else: # for custom code
                    if mode == "visual":
                        weights = {
                            n: p
                            for n, p in self.model.named_parameters()
                            if "lora_visual" in n  # 기존 방식: LoRA visual 파라미터만 업데이트
                        }
                    elif mode == "textual":
                        weights = {
                            n: p
                            for n, p in self.model.named_parameters()
                            if "lora_textual" in n  # 기존 방식: LoRA 파라미터만 업데이트
                        }
                    else: # "one" lora 
                        weights = {
                            n: p
                            for n, p in self.model.named_parameters()
                            if "lora" in n  # 기존 방식: LoRA 파라미터만 업데이트
                        }
        
        elif self.config.inner_params[0] in ['Qformer', 'mm_projector']:
            weights = {
                n: p
                for n, p in self.model.named_parameters()
                if n.find(self.config.inner_params[0]) != -1
            }
        else:
            names = set([n for n, p in self.model.named_parameters()])
            pset = set(self.config.inner_params)
            for p in pset:
                assert p in names, f"inner param {p} not in model"

            weights = {
                n: p
                for n, p in self.model.named_parameters()
                if n in pset
            }
        
        # Save old weights for future restoration
        # self.save_weight = {k: v.detach().clone() for k, v in weights.items()}
        ########

        ### Debug: Lora 있는지?, 학습가능한건? ###
        # print("==== Model Parameter Names ====")
        # for name, param in self.model.named_parameters():
        #     print(name)

        # # ### -------------------------------- ###

        if  connector_mode: # 코드실수: connector <-> adapter 다르게 설정함;;
            edit_lr = self.config.edit_lr/5
        else:
            edit_lr = self.config.edit_lr

        opt = torch.optim.AdamW(
                [v for _, v in weights.items()],
                lr=edit_lr
        )
                   
        # 업데이트 하고싶은 파라미터 불러오기
        for name, w in self.model.named_parameters():
            w.requires_grad = name in weights


        if 'minigpt4' in self.config.model_name.lower() or 'blip' in self.config.model_name.lower() or 'llava' in self.config.model_name.lower():
            pbar = trange(self.config.num_steps, ncols=120)
            for it in pbar: # 1개의 batch로 loss update 반복 'num_step'만큼, 'edit_lr': 1e-4 씩
                opt.zero_grad()

                ### For Edit with LoRA, !Unwrapping! is required ###
                if self.config.use_lora or self.config.lora_connector_type in ["attention", "ffn"]: 
                    outputs = self.model.model(batch) # PeftModelForCasualLM -> LlavaLlamaForCausalLM (LoRA: PeftModelForCasualLM) 
                
                elif self.config.lora_connector_type == "one":
                    if connector_mode:
                        for module in self.model.modules():
                            if hasattr(module, "use_connector"):
                                module.use_connector()

                    outputs = self.model(batch) #true면 lora+mlp, false면 lora

                elif self.config.lora_connector_type == "two":
                    for module in self.model.modules():
                        if hasattr(module, "use_vis_adapter"):
                            if mode == "visual":
                                module.use_vis_adapter()
                            elif mode == "textual" and hasattr(module, "use_text_adapter"):
                                module.use_text_adapter()
                            elif mode == "fusion" and hasattr(module, "use_connector"):
                                module.use_connector()

                    outputs = self.model(batch)

                else:
                    outputs = self.model(batch) # 입력 edit sample에 대한 출력(FT: LlavaLlamaForCausalLM)

                if not isinstance(outputs, torch.Tensor):
                    outputs = outputs.logits
                loss = self.edit_loss_fn(self.config, outputs, batch["labels"])["nll"] # torch.Size([1, 595, 32000]), torch.Size([1, 5])
                pbar.set_postfix({"loss": loss.item()})
                
                torch.autograd.set_detect_anomaly(True) # for debug
                loss.backward()

                opt.step()

                if connector_mode and it >= 2: # connector는 3번만 업데이트 # 5번으로 바꿀까 고민중임 attention만 ? 
                    break

        else:
            raise not NotImplementedError("Model not supported")

        edited_model = self.model

        return (
            FT(
                edited_model,
                self.config,
                self.model_constructor,
            ),
            {}
        )

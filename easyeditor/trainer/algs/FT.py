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

    def forward(self, *inputs, **kwargs):
        if 'minigpt4' in self.config.model_name.lower() or 'blip' in self.config.model_name.lower() or 'llava' in self.config.model_name.lower():
            if self.config.use_lora: # LoRA의 경우, PeftModelForCasualLM -> LLavaLlamaCasualLM으로 래핑을 벗겨내야됨
                outputs = self.model.base_model(*inputs, **kwargs)
                #internal_model = self.model.base_model.model  # LLaVA 모델
                #outputs = internal_model(*inputs, **kwargs) # LorA: PeftModelForCasualLM
            else:
                outputs = self.model(*inputs, **kwargs) # FT 
        elif "qwen-vl" in self.config.model_name.lower():
            outputs = _logits(self.model(inputs[0]['inputs'].to(self.model.device), **kwargs))
        elif "owl2" in self.model.config.name_or_path.lower():
            input_ids, image = inputs[0]['input_ids'].to(self.model.device), inputs[0]['image'].to(self.model.device, dtype=torch.float32)
            # from torch.cuda.amp import autocast
            # with autocast():
            outputs = _logits(self.model.train()(input_ids, 
                                        images=image))
        else:
            raise not NotImplementedError("Model not supported")
        return outputs
    
    def outer_parameters(self):
        return None

    def edit(self, batch, condition=None, detach_history=False, return_factors=False):
        if self.save_weight is not None:
            self.model.load_state_dict(self.save_weight, strict=False)

        self.model.train()

        # LoRA 또는 inner_params 처리
        if not self.config.inner_params:  # inner_params가 비어 있는 경우
            weights = {
                n: p
                for n, p in self.model.named_parameters()
                if "lora" in n  # LoRA 파라미터만 선택
            }
        elif self.config.inner_params[0] in ['Qformer', 'mm_projector', 'vision_model']:
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
        self.save_weight = {k: v.detach().clone() for k, v in weights.items()} # 학습가능 파라미터
        ########

        # ### Debug: Lora 있는지?, 학습가능한건? ###
        # print("==== Model Parameter Names ====")
        # for name, param in self.model.named_parameters():
        #     print(name)

        ### ### 

        opt = torch.optim.AdamW(
            [v for _, v in weights.items()],
            lr=self.config.edit_lr
        )
        for name, w in self.model.named_parameters():
            w.requires_grad = name in weights

        if 'minigpt4' in self.config.model_name.lower() or 'blip' in self.config.model_name.lower() or 'llava' in self.config.model_name.lower():
            pbar = trange(self.config.num_steps, ncols=120)
            for it in pbar:
                opt.zero_grad()
                if self.config.use_lora:
                    outputs = self.model.model(batch) # PeftModelForCasualLM -> LlavaLlamaForCausalLM (LoRA: PeftModelForCasualLM) 
                else:
                    outputs = self.model(batch) # 입력 edit sample에 대한 출력(FT: LlavaLlamaForCausalLM)
                if not isinstance(outputs, torch.Tensor):
                    outputs = outputs.logits
                loss = self.edit_loss_fn(self.config, outputs, batch["labels"])["nll"]
                pbar.set_postfix({"loss": loss.item()})
                loss.backward()

                opt.step()
        elif "qwen-vl" in self.config.model_name.lower():
            pbar = trange(self.config.num_steps, ncols=120)
            for it in pbar:
                opt.zero_grad()

                outputs = self.model(batch['inputs'].to(self.model.device))
                if not isinstance(outputs, torch.Tensor):
                    outputs = outputs.logits
                loss = self.edit_loss_fn(self.config, outputs, batch["labels"])["nll"]
                pbar.set_postfix({"loss": loss.item()})
                loss.backward()
                
                opt.step()
        elif "owl-2" in self.config.model_name.lower():
            pbar = trange(self.config.num_steps, ncols=120)
            for it in pbar:
                opt.zero_grad()

                input_ids, image = batch['input_ids'], batch['image']
                outputs = (self.model(input_ids.to(self.config.device), 
                                         images=image.to(self.config.device, dtype=torch.float32)))
                
                if not isinstance(outputs, torch.Tensor):
                    outputs = outputs.logits
                loss = self.edit_loss_fn(self.config, outputs, batch["labels"])["nll"]
                pbar.set_postfix({"loss": loss.item()})
                loss.backward()

                opt.step()
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

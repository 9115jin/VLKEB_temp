import logging
import re

import torch
import torch.nn as nn
import transformers
from transformers import GPT2Tokenizer, GPT2TokenizerFast, AutoModel, AutoModelForCausalLM, AutoTokenizer
# from transformers import LlavaPreTrainedModel

from .utils import scr

LOG = logging.getLogger(__name__)


class CastModule(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        in_cast: torch.dtype = torch.float32,
        out_cast: torch.dtype = None,
    ):
        super().__init__()

        self.underlying = module
        self.in_cast = in_cast
        self.out_cast = out_cast

    def cast(self, obj, dtype):
        if dtype is None:
            return obj

        if isinstance(obj, torch.Tensor):
            return obj.to(dtype)
        else:
            return obj

    def forward(self, *args, **kwargs):
        args = tuple(self.cast(a, self.in_cast) for a in args)
        kwargs = {k: self.cast(v, self.in_cast) for k, v in kwargs.items()}
        outputs = self.underlying(*args, **kwargs)
        if isinstance(outputs, torch.Tensor):
            outputs = self.cast(outputs, self.out_cast)
        elif isinstance(outputs, tuple):
            outputs = tuple(self.cast(o, self.out_cast) for o in outputs)
        else:
            raise RuntimeError(f"Not sure how to cast type {type(outputs)}")
        return outputs

    def extra_repr(self):
        return f"in_cast: {self.in_cast}\nout_cast: {self.out_cast}"


class BertClassifier(torch.nn.Module):
    def __init__(self, model_name, hidden_dim=768):
        super().__init__()
        self.model = transformers.BertModel.from_pretrained(model_name, cache_dir=scr())
        self.classifier = torch.nn.Linear(hidden_dim, 1)

    @property
    def config(self):
        return self.model.config

    def forward(self, *args, **kwargs):
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != "labels"}
        return self.classifier(self.model(*args, **filtered_kwargs)[1])

# 모델 불러오는 코드
def get_model(config):
    # 파라미터 가져오기 
    if config.model_class == "BertClassifier":
        model = BertClassifier(config.model_name)
    elif config.model_name == "blip2":
        from .blip2_models.blip2_opt import Blip2OPT
        
        model = Blip2OPT(
            vit_model="eva_clip_g",
            img_size=364,
            use_grad_checkpoint=True,
            vit_precision="fp32",
            freeze_vit=True,
            freeze_qformer=config.freeze_qformer,
            opt_model=config.name,
            state_dict_file=config.state_dict_file,
            qformer_name_or_path=config.qformer_name_or_path,
            qformer_checkpoint=config.qformer_checkpoint
        )
        # for name, param in model.named_parameters():
        #     print(f"{name}: {param.shape}")
    elif config.model_name == "minigpt4":
        from .blip2_models.mini_gpt4 import MiniGPT4

        model = MiniGPT4(
            vit_model="eva_clip_g",
            qformer_checkpoint=config.qformer_checkpoint,
            img_size=364,
            use_grad_checkpoint=True,
            vit_precision="fp32",
            freeze_vit=True,
            freeze_qformer=config.freeze_qformer,
            llama_model=config.name,
            state_dict_file=config.state_dict_file,
            qformer_name_or_path=config.qformer_name_or_path,
            pretrained_ckpt=config.pretrained_ckpt,
        )

        # for name, param in model.named_parameters():
        #     print(f"{name}: {param.shape}")
    elif config.model_name == "llava": ## LLAVA 모델은 여기서! ## 
        # 모델 로드
        from .llava.model.builder import load_pretrained_model
        if getattr(config, 'use_lora', False): # 기존 모델 불러옴(LLAVA)
            model = load_pretrained_model(
                model_path=config.name,
                device=config.device,
                use_lora=config.use_lora,
                lora_r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                lora_target_modules=config.lora_target_modules,
                inner_params=config.inner_params
            )
            
        else: # LoRA 적용된 LLAVA
            model = load_pretrained_model(model_path=config.name, device=config.device)
        
        # for name, param in model.named_parameters():
        #     print(f"{name}: {param.shape}")
    elif config.model_name == "qwen-vl":
        LOG.info(
            f"Loading model with name {config.model_name}"
        )
        model = AutoModelForCausalLM.from_pretrained(config.name, 
                                                     device_map=f"cuda:{config.device}", 
                                                     trust_remote_code=True)
    elif config.model_name == "owl-2":
        LOG.info(
            f"Loading model with name {config.model_name}"
        )
        from .mPLUG_Owl2.mplug_owl2.model.builder import load_pretrained_model
        from .mPLUG_Owl2.mplug_owl2.model.modeling_mplug_owl2 import replace_llama_modality_adaptive
        replace_llama_modality_adaptive()
        tokenizer , model, _, _ = load_pretrained_model(config.name, None, 'mplug_owl2', load_8bit=False, load_4bit=False, device=f"cuda:{config.device}")
        for param in model.parameters():
            param.requires_grad = True
    else:
        ModelClass = getattr(transformers, config.model_class)
        LOG.info(
            f"Loading model class {ModelClass} with name {config.model_name}"
        )
        model = ModelClass.from_pretrained(config.model_name, trust_remote_code=True, device_map='auto' if config.model_parallel else None)

    # if config.model.pt is not None:
    #     LOG.info(f"Loading model initialization from {config.model.pt}")
    #     state_dict = torch.load(config.model.pt, map_location="cpu")
    #
    #     try:
    #         model.load_state_dict(state_dict)
    #     except RuntimeError:
    #         LOG.info("Default load failed; stripping prefix and trying again.")
    #         state_dict = {re.sub("^model.", "", k): v for k, v in state_dict.items()}
    #
    #         model.load_state_dict(state_dict)
    #
    #     LOG.info("Loaded model initialization")

    #
    # After Model Loading, drop out setting
    if config.dropout is not None:
        n_reset = 0
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = config.dropout
                n_reset += 1

            if hasattr(m, "dropout"):  # Requires for BART, which uses F.dropout
                if isinstance(m.dropout, float):
                    m.dropout = config.dropout
                    n_reset += 1

            if hasattr( 
                m, "activation_dropout"
            ):  # Requires for BART, which uses F.dropout
                if isinstance(m.activation_dropout, float):
                    m.activation_dropout = config.dropout
                    n_reset += 1

        LOG.info(f"Set {n_reset} dropout modules to p={config.dropout}")

    param_names = [n for n, _ in model.named_parameters()]
    bad_inner_params = [p for p in config.inner_params if p not in param_names]
    if len(bad_inner_params) != 0:
        if config.inner_params[0] not in ['Qformer', 'mm_projector', 'vision_model']:
            raise ValueError(
                f"Params {bad_inner_params} do not exist in model of type {type(model)}."
            )

    # memory effieicnt: half
    if config.no_grad_layers is not None:
        if config.half:
            model.bfloat16()

        def upcast(mod):
            modlist = None
            for child in mod.children():
                if isinstance(child, nn.ModuleList):
                    assert modlist is None, f"Found multiple modlists for {mod}"
                    modlist = child
            if modlist is None:
                raise RuntimeError("Couldn't find a ModuleList child")

            LOG.info(
                f"Setting {len(modlist) - config.no_grad_layers} modules to full precision, with autocasting"
            )
            modlist[config.no_grad_layers :].to(torch.float32)
            modlist[config.no_grad_layers] = CastModule(modlist[config.no_grad_layers])
            modlist[-1] = CastModule(
                modlist[-1], in_cast=torch.float32, out_cast=torch.bfloat16
            )

        parents = []
        if hasattr(model, "transformer"):
            parents.append(model.transformer)
        if hasattr(model, "encoder"):
            parents.append(model.encoder)
        if hasattr(model, "decoder"):
            parents.append(model.decoder)
        if hasattr(model, "model"):
            parents.extend([model.model.encoder, model.model.decoder])

        for t in parents:
            t.no_grad_layers = config.no_grad_layers
            if config.half:
                upcast(t)

        if config.half:
            idxs = []
            for p in config.inner_params:
                for comp in p.split("."):
                    if comp.isdigit():
                        idxs.append(int(comp))
            max_idx, min_idx = str(max(idxs)), str(config.no_grad_layers)
            for pidx, p in enumerate(config.inner_params):
                comps = p.split(".")
                if max_idx in comps or min_idx in comps:
                    index = (
                        comps.index(max_idx)
                        if max_idx in comps
                        else comps.index(min_idx)
                    )
                    comps.insert(index + 1, "underlying")
                    new_p = ".".join(comps)
                    LOG.info(
                        f"Replacing config.inner_params[{pidx}] '{p}' -> '{new_p}'"
                    )
                    config.inner_params[pidx] = new_p

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('**************\n', f'total params: {pytorch_total_params}\n', '**************\n')
    return model


def get_tokenizer(config):
    tok_name = (
        config.tokenizer_name
        if config.tokenizer_name is not None
        else config.model.name
    )
    tokenizer =  getattr(transformers, config.tokenizer_class).from_pretrained(
        tok_name, cache_dir=scr()
    )
    if isinstance(tokenizer, GPT2Tokenizer) or isinstance(tokenizer, GPT2TokenizerFast):
        tokenizer.pad_token_id  = tokenizer.eos_token_id
        tokenizer.padding_side = 'left'
    return tokenizer


if __name__ == "__main__":
    m = BertClassifier("bert-base-uncased")
    m(torch.arange(5)[None, :])
    import pdb

    pdb.set_trace()

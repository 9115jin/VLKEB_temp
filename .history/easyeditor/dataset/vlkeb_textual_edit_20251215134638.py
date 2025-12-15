"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from collections import OrderedDict

from .processor.base_dataset import BaseDataset
from .processor.blip_processors import BlipImageEvalProcessor
from ..trainer.utils import dict_to
from PIL import Image
import random
import typing
import torch
import transformers
from tqdm import tqdm
from copy import deepcopy

class TextualDataset(BaseDataset):
    def __init__(self, data_dir: str, size:  typing.Optional[int] = None, config=None, no_image=False, hop=None, *args, **kwargs):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """

        # Load Tokenizer
        if (config is not None and hasattr(config, 'tokenizer_name')):
            tok_name = (
                config.tokenizer_name
                if config.tokenizer_name is not None
                else config.name
            )
            tokenizer = getattr(transformers, config.tokenizer_class).from_pretrained(
                tok_name, trust_remote_code=True
            )            
            if tokenizer.pad_token == None or tokenizer.pad_token == '':
                tokenizer.pad_token = tokenizer.eos_token  


        super().__init__(ann_paths=[data_dir]) # eval_multohop.json -> image 관련 설정?
             

        self.config = config
        self.tok = tokenizer
        self.max_length = 32

        self.prompt = "Question: {} Short answer: " # 어디에 쓰이지? 학습? Inference?

        data = []
        if size is not None:
            self.annotation = self.annotation[:size]
        if hop:
            self.hop = hop
            assert int(hop) in [1, 2, 3, 4], "hop should be 1, 2, 3, or 4"
            port_types = ['', '1-hop', '2-hop', '3-hop', '4-hop']
            port_type = port_types[int(hop)]
        
        for record in tqdm(self.annotation, ncols=120, desc='Loading Data'):

            if record['alt'] == "": # 편집할 내용 없으면 패스
                continue
            if hop and 'port_new' not in record.keys(): # Portability Data 아니면 패스(1-hop)
                continue


            # alt, pred가 리스트인 경우 문자열로 변환
            record['pred'] = " ".join(record['pred']) if isinstance(record['pred'], list) else record['pred']
            record['alt'] = " ".join(record['alt']) if isinstance(record['alt'], list) else record['alt']

            item = {
                'prompt': record['src'],
                'pred': record['pred'],
                'target': record['alt'],
                'rephrase_prompt': record['rephrase'],
                'cond': "{} >> {} || {}".format(  # 원본(pred) -> 편집(alt)  ||  질문(src)
                    record['pred'],
                    record['alt'],
                    record['src']
                )
            }
            
            # Text-Locality
            item['locality_prompt'] = record['loc']
            item['locality_ground_truth'] = record['loc_ans']

            if 'port_new' in record.keys():
                item['portability_prompt'] = []
                item['portability_ground_truth'] = []
                find_hop = False
                for ports in record['port_new']:
                    if ports['port_type'] == port_type:
                        find_hop = True
                        port_q = ports['Q&A']['Question']
                        port_a = ports['Q&A']['Answer']
                        item['portability_prompt'].append(port_q)
                        item['portability_ground_truth'].append(port_a)
                        break
                
                if not find_hop:
                    continue
            data.append(item)
            
        # if size is not None:
        #     data = data[:size]        
        self._data = data
        self.no_image = no_image

    def __getitem__(self, index):
        if self.no_image:
            return self._data[index]

        data = deepcopy(self._data[index])        
        # load image
        image_path = data['image']
        rephrase_image_path = data['image_rephrase']
        locality_image_path = data['multimodal_locality_image']
        
        image = Image.open(image_path).convert("RGB")
        rephrase_image = Image.open(rephrase_image_path).convert("RGB")
        locality_image = Image.open(locality_image_path).convert("RGB")
        
        if self.config.model_class == "Blip2OPT":
            image = self.vis_processor(image)
            rephrase_image = self.vis_processor(rephrase_image)
            locality_image = self.vis_processor(locality_image)
        elif self.config.model_class == "LLaVA":
            image = self.vis_processor(image, return_tensors='pt')['pixel_values'].to(dtype=torch.float16)
            rephrase_image = self.vis_processor(rephrase_image, return_tensors='pt')['pixel_values'].to(dtype=torch.float16)
            locality_image = self.vis_processor(locality_image, return_tensors='pt')['pixel_values'].to(dtype=torch.float16)
        else:
            raise NotImplementedError

        data['image'] = image
        data['image_rephrase'] = rephrase_image
        data['multimodal_locality_image'] = locality_image

        return data
    
    def __len__(self):
        return len(self._data)

    def collate_fn(self, batch):
        src = [b['prompt'] for b in batch]
        trg = [b['target'] for b in batch]
        cond = [b['cond'] for b in batch]
        rephrase = [b['rephrase_prompt'] for b in batch]

        loc_q = [b["locality_prompt"] for b in batch]
        loc_a = [b["locality_ground_truth"] for b in batch]

        # edit_inner - Reliability
        edit_inner = {}
        edit_inner['image'] = None
        edit_inner['text_input'] = [" ".join([s, t]) for s, t in zip(src, trg)]
        edit_inner['labels'] = trg
        edit_inner['prompts_len'] = [len(self.tok.encode(s, add_special_tokens=False)) for s in src]
        edit_inner['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        
        # edit_outer - Generality
        edit_outer = {}
        edit_outer['image'] = None
        edit_outer['text_input'] = [" ".join([r, t]) for r, t in zip(rephrase, trg)]
        edit_outer['labels'] = trg
        edit_outer['prompts_len'] = [len(self.tok.encode(r, add_special_tokens=False)) for r in rephrase]
        edit_outer['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        
        # loc
        loc = {}
        loc['image'] = None
        loc['text_input'] = [" ".join([q, a]) for q, a in zip(loc_q, loc_a)]
        loc['labels'] = loc_a
        loc['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in loc_q]
        loc['labels'] = self.tok(loc_a, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        
        # cond
        cond = self.tok(
            cond,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
            truncation=True,
        ).to(self.config.device)

        edit_ports = None
        # if 'portability_prompt' in batch[0].keys():
        #     edit_ports = []
        #     for port_q, port_a in zip(batch[0]['portability_prompt'], batch[0]['portability_ground_truth']):
        #         port = {}
        #         port['image'] = torch.stack(image, dim=0)
        #         port['text_input'] = [' '.join([port_q, port_a])]
        #         port['labels'] = [port_a]
        #         port['prompts_len'] = [len(self.tok.encode(port_q, add_special_tokens=False))]
        #         port['labels'] = self.tok([port_a], add_special_tokens=False, return_tensors="pt",)["input_ids"]
        #         edit_ports.append(port)

        
        batch = {
            "edit_inner": edit_inner, # Rel
            "edit_outer": edit_outer, # Gen
            "edit_outer_image": None,
            "loc": loc,# Loc
            "loc_image": None,
            'port': edit_ports,
            "cond": cond
        }
        return dict_to(batch, self.config.device)

class CompositionalDataset(BaseDataset):
    def __init__(self, data_dir: str, size:  typing.Optional[int] = None, config=None, no_image=False, hop=None, *args, **kwargs):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        # get tokenizer and vis_processor
        if config.model_class == "Blip2OPT":
            vis_processor = BlipImageEvalProcessor(image_size=364, mean=None, std=None)
        elif config.model_class == "LLaVA":
            vis_processor = transformers.CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        else:
            raise NotImplementedError("unknown model class")
        # Load Tokenizer
        if (config is not None and hasattr(config, 'tokenizer_name')):
            tok_name = (
                config.tokenizer_name
                if config.tokenizer_name is not None
                else config.name
            )
            tokenizer = getattr(transformers, config.tokenizer_class).from_pretrained(
                tok_name, trust_remote_code=True
            )            
            if tokenizer.pad_token == None or tokenizer.pad_token == '':
                tokenizer.pad_token = tokenizer.eos_token  
                
        vis_root = config.coco_image
        rephrase_root = config.rephrase_image
        super().__init__(vis_processor, vis_root, rephrase_root, [data_dir]) # eval_multohop.json -> image 관련 설정?

        self.config = config
        self.tok = tokenizer
        self.max_length = 32

        self.prompt = "Question: {} Short answer: " # 어디에 쓰이지? 학습? Inference?

        data = []
        if size is not None:
            self.annotation = self.annotation[:size]
        if hop:
            self.hop = hop
            assert int(hop) in [1, 2, 3, 4], "hop should be 1, 2, 3, or 4"
            port_types = ['', '1-hop', '2-hop', '3-hop', '4-hop']
            port_type = port_types[int(hop)]
        for record in tqdm(self.annotation, ncols=120, desc='Loading Data'):
            
            if record['alt'] == "": # 편집할 내용 없으면 패스
                continue
            if hop and 'port_new' not in record.keys(): # Portability Data 아니면 패스(1-hop)
                continue
            
            ## Visual Edit Data ##
            # 이미지 관련 경로
            item = {
                'prompt': record['src'],
                'pred': record['pred'],
                'target': record['alt'],
                'rephrase_prompt': record['rephrase'],
                'image': os.path.join(self.vis_root, record["image"]),
                'image_rephrase': os.path.join(self.rephrase_root, record["image_rephrase"]),
                'cond': "{} >> {} || {}".format(record['pred'], record['alt'], record['src']),
                'locality_prompt': record['loc'],
                'locality_ground_truth': record['loc_ans'],
                'multimodal_locality_image': os.path.join(self.vis_root, record['m_loc']),
                'multimodal_locality_prompt': record['m_loc_q'],
                'multimodal_locality_ground_truth': record['m_loc_a']
            }

            
            ##  Textual Edit Data ##
            textual_pred = " ".join(record["textual_edit"]['pred']) if isinstance(record["textual_edit"]['pred'], list) else record["textual_edit"]['pred']
            textual_alt = " ".join(record["textual_edit"]['alt']) if isinstance(record["textual_edit"]['alt'], list) else record["textual_edit"]['alt']

            if "textual_edit" in record:
                item["textual_edit"] = {
                    "prompt": record["textual_edit"]["src"],
                    "pred": textual_pred,
                    "target": textual_alt, 
                    "rephrase_prompt": record["textual_edit"]["rephrase"],
                    'cond': "{} >> {} || {}".format(  # 원본(pred) -> 편집(alt)  ||  질문(src)
                        textual_pred,
                        textual_alt,
                        record["textual_edit"]['src']
                        ),
                    'locality_prompt': record["textual_edit"]['loc'],
                    'locality_ground_truth': record["textual_edit"]['loc_ans']
                }
                
            # Compositional portabiliy
            if 'port_new' in record.keys():
                item['portability_prompt'] = []
                item['portability_ground_truth'] = []
                find_hop = False
                for ports in record['port_new']:
                    if ports['port_type'] == port_type:
                        find_hop = True
                        port_q = """ ports['Q&A']['Question'] """ #  "What country is the city in the image part of?"
                        port_a = textual_alt              #  e: "Billings,_Montana"(before textual edit) -> e`: "United States"(after tex edit)
                        item['portability_prompt'].append(port_q)
                        item['portability_ground_truth'].append(port_a)
                        break
                
                if not find_hop:
                    continue

            
            data.append(item)

        # if size is not None:
        #     data = data[:size]        
        self._data = data
        self.no_image = no_image
    # collate_fn(batch)를 만들기위해 batch size 만큼 호출됨
    def __getitem__(self, index):
        if self.no_image:
            return self._data[index]

        data = deepcopy(self._data[index])        
        # load image
        image_path = data['image']
        rephrase_image_path = data['image_rephrase']
        locality_image_path = data['multimodal_locality_image']
        
        image = Image.open(image_path).convert("RGB")
        rephrase_image = Image.open(rephrase_image_path).convert("RGB")
        locality_image = Image.open(locality_image_path).convert("RGB")
        
        if self.config.model_class == "Blip2OPT":
            image = self.vis_processor(image)
            rephrase_image = self.vis_processor(rephrase_image)
            locality_image = self.vis_processor(locality_image)
        elif self.config.model_class == "LLaVA":
            image = self.vis_processor(image, return_tensors='pt')['pixel_values'].to(dtype=torch.float16)
            rephrase_image = self.vis_processor(rephrase_image, return_tensors='pt')['pixel_values'].to(dtype=torch.float16)
            locality_image = self.vis_processor(locality_image, return_tensors='pt')['pixel_values'].to(dtype=torch.float16)
        else:
            raise NotImplementedError

        data['image'] = image
        data['image_rephrase'] = rephrase_image
        data['multimodal_locality_image'] = locality_image

        return data
    
    def __len__(self):
        return len(self._data)

    def collate_fn(self, batch):
        # -----------------------------------------------------
        # 1) Visual Edit 파트
        # -----------------------------------------------------
        # 예: prompt, target, cond, rephrase, image 등등 가져오기
        src = [b['prompt'] for b in batch]
        trg = [b['target'] for b in batch]
        rephrase = [b['rephrase_prompt'] for b in batch]

        image = [b['image'] for b in batch]
        image_rephrase = [b['image_rephrase'] for b in batch]

        loc_q = [b["locality_prompt"] for b in batch]
        loc_a = [b["locality_ground_truth"] for b in batch]

        m_loc_image = [b['multimodal_locality_image'] for b in batch]
        m_loc_q = [b['multimodal_locality_prompt'] for b in batch]
        m_loc_a = [b['multimodal_locality_ground_truth'] for b in batch]

        # edit_inner
        v_edit_inner = {}
        v_edit_inner['image'] = torch.stack(image, dim=0)
        v_edit_inner['text_input'] = [" ".join([s, t]) for s, t in zip(src, trg)]
        v_edit_inner['labels'] = trg
        v_edit_inner['prompts_len'] = [len(self.tok.encode(s, add_special_tokens=False)) for s in src]
        v_edit_inner['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt")["input_ids"]

        # edit_outer
        v_edit_outer = {}
        v_edit_outer['image'] = torch.stack(image, dim=0)
        v_edit_outer['text_input'] = [" ".join([r, t]) for r, t in zip(rephrase, trg)]
        v_edit_outer['labels'] = trg
        v_edit_outer['prompts_len'] = [len(self.tok.encode(r, add_special_tokens=False)) for r in rephrase]
        v_edit_outer['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt")["input_ids"]

        # edit_outer_image
        v_edit_outer_image = {}
        v_edit_outer_image['image'] = torch.stack(image_rephrase, dim=0)
        v_edit_outer_image['text_input'] = [" ".join([s, t]) for s, t in zip(src, trg)]
        v_edit_outer_image['labels'] = trg
        v_edit_outer_image['prompts_len'] = [len(self.tok.encode(s, add_special_tokens=False)) for s in src]
        v_edit_outer_image['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt")["input_ids"]

        # loc
        v_loc = {}
        v_loc['image'] = None
        v_loc['text_input'] = [" ".join([q, a]) for q, a in zip(loc_q, loc_a)]
        v_loc['labels'] = loc_a
        v_loc['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in loc_q]
        v_loc['labels'] = self.tok(loc_a, add_special_tokens=False, return_tensors="pt")["input_ids"]

        # loc_image
        v_loc_image = {}
        v_loc_image['image'] = torch.stack(m_loc_image, dim=0)
        v_loc_image['text_input'] = [" ".join([q, a]) for q, a in zip(m_loc_q, m_loc_a)]
        v_loc_image['labels'] = m_loc_a
        v_loc_image['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in m_loc_q]
        v_loc_image['labels'] = self.tok(m_loc_a, add_special_tokens=False, return_tensors="pt")["input_ids"]

        # -----------------------------------------------------
        # 2) Textual Edit 파트 (기존 textual_edit 부분)
        # -----------------------------------------------------
        t_src = [b['textual_edit']['prompt'] for b in batch]
        t_trg = [b['textual_edit']['target'] for b in batch]
        # loc_q / loc_a도 텍스트 기반으로 사용된다면 추가
        t_loc_q = [b.get("textual_edit", {}).get("locality_prompt", "") for b in batch]
        t_loc_a = [b.get("textual_edit", {}).get("locality_ground_truth", "") for b in batch]

        # edit_inner
        t_edit_inner = {}
        t_edit_inner['image'] = None
        t_edit_inner['text_input'] = [" ".join([s, t]) for s, t in zip(t_src, t_trg)]
        t_edit_inner['labels'] = t_trg
        t_edit_inner['prompts_len'] = [len(self.tok.encode(s, add_special_tokens=False)) for s in t_src]
        t_edit_inner['labels'] = self.tok(t_trg, add_special_tokens=False, return_tensors="pt")["input_ids"]

        # edit_outer
        # rephrase_prompt가 textual_edit 안에 있다고 가정
        t_rephrase = [b.get("textual_edit", {}).get("rephrase_prompt", "") for b in batch]
        t_edit_outer = {}
        t_edit_outer['image'] = None
        t_edit_outer['text_input'] = [" ".join([r, t]) for r, t in zip(t_rephrase, t_trg)] if t_rephrase else []
        t_edit_outer['labels'] = t_trg
        t_edit_outer['prompts_len'] = [len(self.tok.encode(r, add_special_tokens=False)) for r in t_rephrase] if t_rephrase else []
        if t_rephrase:
            t_edit_outer['labels'] = self.tok(t_trg, add_special_tokens=False, return_tensors="pt")["input_ids"]
        else:
            t_edit_outer['labels'] = None

        # loc
        t_loc = {}
        if t_loc_q and t_loc_a:
            t_loc['image'] = None
            t_loc['text_input'] = [" ".join([q, a]) for q, a in zip(t_loc_q, t_loc_a)]
            t_loc['labels'] = t_loc_a
            t_loc['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in t_loc_q]
            t_loc['labels'] = self.tok(t_loc_a, add_special_tokens=False, return_tensors="pt")["input_ids"]
        else:
            t_loc['image'] = None
            t_loc['text_input'] = []
            t_loc['labels'] = []
            t_loc['prompts_len'] = []

        # (Portability) 예시
        # 필요하면 textual_edit 쪽에도 추가 가능
        t_edit_ports = None
        # if 'portability_prompt' in b["textual_edit"].keys():
        #    ...


        # -----------------------------------------------------
        # 4) Compositional Portabiltiy (공통)
        # -----------------------------------------------------
        edit_ports = None
        if 'portability_prompt' in batch[0].keys():
            edit_ports = []
            for port_q, port_a in zip(batch[0]['portability_prompt'], batch[0]['portability_ground_truth']):
                port = {}
                # port['image'] = ???  # 필요하다면 활용
                port['image'] = torch.stack(image, dim=0)
                port['text_input'] = [' '.join([port_q, port_a])]
                port['labels'] = [port_a]
                port['prompts_len'] = [len(self.tok.encode(port_q, add_special_tokens=False))]
                port['labels'] = self.tok([port_a], add_special_tokens=False, return_tensors="pt")["input_ids"]
                edit_ports.append(port)


        # -----------------------------------------------------
        # 5) cond 토큰화 (공통)
        # -----------------------------------------------------
        cond = [b['cond'] for b in batch]
        cond = self.tok(
            cond,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
            truncation=True,
        ).to(self.config.device)

        # -----------------------------------------------------
        # 4) 최종 Batch 구성
        # -----------------------------------------------------
        # visual_edit와 textual_edit를 따로 구성
        visual_edit = {
            "edit_inner": v_edit_inner,
            "edit_outer": v_edit_outer,
            "edit_outer_image": v_edit_outer_image,
            "loc": v_loc,
            "loc_image": v_loc_image,
        }

        textual_edit = {
            "edit_inner": t_edit_inner,
            "edit_outer": t_edit_outer,
            "loc": t_loc,
            "port": t_edit_ports,  # 필요하면 정의
        }

        batch_dict = {
            "visual_edit": visual_edit,
            "textual_edit": textual_edit,
            "cond": cond,
            "port": edit_ports
        }

        # 모든 텐서를 self.config.device 로 이동(혹은 필요한 후처리)
        batch_dict = dict_to(batch_dict, self.config.device)
        return batch_dict

class CompositionalDataset_RAG(BaseDataset):
    def __init__(self, data_dir: str, size:  typing.Optional[int] = None, config=None, no_image=False, hop=None, *args, **kwargs):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        # get tokenizer and vis_processor
        if config.model_class == "Blip2OPT":
            vis_processor = BlipImageEvalProcessor(image_size=364, mean=None, std=None)
        elif config.model_class == "LLaVA":
            vis_processor = transformers.CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        else:
            raise NotImplementedError("unknown model class")
        # Load Tokenizer
        if (config is not None and hasattr(config, 'tokenizer_name')):
            tok_name = (
                config.tokenizer_name
                if config.tokenizer_name is not None
                else config.name
            )
            tokenizer = getattr(transformers, config.tokenizer_class).from_pretrained(
                tok_name, trust_remote_code=True
            )            
            if tokenizer.pad_token == None or tokenizer.pad_token == '':
                tokenizer.pad_token = tokenizer.eos_token  
                
        vis_root = config.coco_image
        rephrase_root = config.rephrase_image
        super().__init__(vis_processor, vis_root, rephrase_root, [data_dir]) # eval_multohop.json -> image 관련 설정?

        self.config = config
        self.tok = tokenizer
        self.max_length = 32

        self.prompt = "Question: {} Short answer: " # 어디에 쓰이지? 학습? Inference?

        data = []
        if size is not None:
            self.annotation = self.annotation[:size]
        if hop:
            self.hop = hop
            assert int(hop) in [1, 2, 3, 4], "hop should be 1, 2, 3, or 4"
            port_types = ['', '1-hop', '2-hop', '3-hop', '4-hop']
            port_type = port_types[int(hop)]
        for record in tqdm(self.annotation, ncols=120, desc='Loading Data'):
            
            if record['alt'] == "": # 편집할 내용 없으면 패스
                continue
            if hop and 'port_new' not in record.keys(): # Portability Data 아니면 패스(1-hop)
                continue
            
            ## Visual Edit Data ##
            # 이미지 관련 경로
            item = {
                'prompt': record['src'],
                'pred': record['pred'],
                'target': record['alt'],
                'rephrase_prompt': record['rephrase'],
                'image': os.path.join(self.vis_root, record["image"]),
                'image_rephrase': os.path.join(self.rephrase_root, record["image_rephrase"]),
                'cond': "{} >> {} || {}".format(record['pred'], record['alt'], record['src']),
                'locality_prompt': record['loc'],
                'locality_ground_truth': record['loc_ans'],
                'multimodal_locality_image': os.path.join(self.vis_root, record['m_loc']),
                'multimodal_locality_prompt': record['m_loc_q'],
                'multimodal_locality_ground_truth': record['m_loc_a']
            }

            
            ##  Textual Edit Data ##
            textual_pred = " ".join(record["textual_edit"]['pred']) if isinstance(record["textual_edit"]['pred'], list) else record["textual_edit"]['pred']
            textual_alt = " ".join(record["textual_edit"]['alt']) if isinstance(record["textual_edit"]['alt'], list) else record["textual_edit"]['alt']

            if "textual_edit" in record:
                item["textual_edit"] = {
                    "prompt": record["textual_edit"]["src"],
                    "pred": textual_pred,
                    "target": textual_alt, 
                    "rephrase_prompt": record["textual_edit"]["rephrase"],
                    'cond': "{} >> {} || {}".format(  # 원본(pred) -> 편집(alt)  ||  질문(src)
                        textual_pred,
                        textual_alt,
                        record["textual_edit"]['src']
                        ),
                    'locality_prompt': record["textual_edit"]['loc'],
                    'locality_ground_truth': record["textual_edit"]['loc_ans']
                }
                
            # Compositional portabiliy
            if 'port_new' in record.keys():
                item['portability_prompt'] = []
                item['portability_ground_truth'] = []
                find_hop = False
                for ports in record['port_new']:
                    if ports['port_type'] == port_type:
                        find_hop = True
                        port_q = ports['Q&A']['Question'] #  "What country is the city in the image part of?"
                        port_a = textual_alt              #  e: "Billings,_Montana"(before textual edit) -> e`: "United States"(after tex edit)
                        item['portability_prompt'].append(port_q)
                        item['portability_ground_truth'].append(port_a)
                        break
                
                if not find_hop:
                    continue

            
            data.append(item)

        # if size is not None:
        #     data = data[:size]        
        self._data = data
        self.no_image = no_image
    # collate_fn(batch)를 만들기위해 batch size 만큼 호출됨
    def __getitem__(self, index):
        if self.no_image:
            return self._data[index]

        data = deepcopy(self._data[index])     
        # load image
        image_path = data['image']
        rephrase_image_path = data['image_rephrase']
        locality_image_path = data['multimodal_locality_image']
        
        image = Image.open(image_path).convert("RGB")
        rephrase_image = Image.open(rephrase_image_path).convert("RGB")
        locality_image = Image.open(locality_image_path).convert("RGB")
        
        if self.config.model_class == "Blip2OPT":
            image = self.vis_processor(image)
            rephrase_image = self.vis_processor(rephrase_image)
            locality_image = self.vis_processor(locality_image)
        elif self.config.model_class == "LLaVA":
            image = self.vis_processor(image, return_tensors='pt')['pixel_values'].to(dtype=torch.float16)
            rephrase_image = self.vis_processor(rephrase_image, return_tensors='pt')['pixel_values'].to(dtype=torch.float16)
            locality_image = self.vis_processor(locality_image, return_tensors='pt')['pixel_values'].to(dtype=torch.float16)
        else:
            raise NotImplementedError

        data['image'] = image
        data['image_rephrase'] = rephrase_image
        data['multimodal_locality_image'] = locality_image

        return data
    
    def __len__(self):
        return len(self._data)

    def collate_fn(self, batch):
        # -----------------------------------------------------
        # 1) Visual Edit 파트
        # -----------------------------------------------------
        # 예: prompt, target, cond, rephrase, image 등등 가져오기
        src = [b['prompt'] for b in batch]
        trg = [b['target'] for b in batch]
        rephrase = [b['rephrase_prompt'] for b in batch]

        image = [b['image'] for b in batch]
        image_rephrase = [b['image_rephrase'] for b in batch]

        loc_q = [b["locality_prompt"] for b in batch]
        loc_a = [b["locality_ground_truth"] for b in batch]

        m_loc_image = [b['multimodal_locality_image'] for b in batch]
        m_loc_q = [b['multimodal_locality_prompt'] for b in batch]
        m_loc_a = [b['multimodal_locality_ground_truth'] for b in batch]

        # edit_inner
        v_edit_inner = {}
        v_edit_inner['image'] = torch.stack(image, dim=0)
        v_edit_inner['text_input'] = [" ".join([s, t]) for s, t in zip(src, trg)]
        v_edit_inner['labels'] = trg
        v_edit_inner['prompts_len'] = [len(self.tok.encode(s, add_special_tokens=False)) for s in src]
        v_edit_inner['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt")["input_ids"]

        # edit_outer
        v_edit_outer = {}
        v_edit_outer['image'] = torch.stack(image, dim=0)
        v_edit_outer['text_input'] = [" ".join([r, t]) for r, t in zip(rephrase, trg)]
        v_edit_outer['labels'] = trg
        v_edit_outer['prompts_len'] = [len(self.tok.encode(r, add_special_tokens=False)) for r in rephrase]
        v_edit_outer['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt")["input_ids"]

        # edit_outer_image
        v_edit_outer_image = {}
        v_edit_outer_image['image'] = torch.stack(image_rephrase, dim=0)
        v_edit_outer_image['text_input'] = [" ".join([s, t]) for s, t in zip(src, trg)]
        v_edit_outer_image['labels'] = trg
        v_edit_outer_image['prompts_len'] = [len(self.tok.encode(s, add_special_tokens=False)) for s in src]
        v_edit_outer_image['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt")["input_ids"]

        # loc
        v_loc = {}
        v_loc['image'] = None
        v_loc['text_input'] = [" ".join([q, a]) for q, a in zip(loc_q, loc_a)]
        v_loc['labels'] = loc_a
        v_loc['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in loc_q]
        v_loc['labels'] = self.tok(loc_a, add_special_tokens=False, return_tensors="pt")["input_ids"]

        # loc_image
        v_loc_image = {}
        v_loc_image['image'] = torch.stack(m_loc_image, dim=0)
        v_loc_image['text_input'] = [" ".join([q, a]) for q, a in zip(m_loc_q, m_loc_a)]
        v_loc_image['labels'] = m_loc_a
        v_loc_image['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in m_loc_q]
        v_loc_image['labels'] = self.tok(m_loc_a, add_special_tokens=False, return_tensors="pt")["input_ids"]

        # -----------------------------------------------------
        # 2) Textual Edit 파트 (기존 textual_edit 부분)
        # -----------------------------------------------------
        t_src = [b['textual_edit']['prompt'] for b in batch]
        t_trg = [b['textual_edit']['target'] for b in batch]
        # loc_q / loc_a도 텍스트 기반으로 사용된다면 추가
        t_loc_q = [b.get("textual_edit", {}).get("locality_prompt", "") for b in batch]
        t_loc_a = [b.get("textual_edit", {}).get("locality_ground_truth", "") for b in batch]

        # edit_inner
        t_edit_inner = {}
        t_edit_inner['image'] = None
        t_edit_inner['text_input'] = [" ".join([s, t]) for s, t in zip(t_src, t_trg)]
        t_edit_inner['labels'] = t_trg
        t_edit_inner['prompts_len'] = [len(self.tok.encode(s, add_special_tokens=False)) for s in t_src]
        t_edit_inner['labels'] = self.tok(t_trg, add_special_tokens=False, return_tensors="pt")["input_ids"]

        # edit_outer
        # rephrase_prompt가 textual_edit 안에 있다고 가정
        t_rephrase = [b.get("textual_edit", {}).get("rephrase_prompt", "") for b in batch]
        t_edit_outer = {}
        t_edit_outer['image'] = None
        t_edit_outer['text_input'] = [" ".join([r, t]) for r, t in zip(t_rephrase, t_trg)] if t_rephrase else []
        t_edit_outer['labels'] = t_trg
        t_edit_outer['prompts_len'] = [len(self.tok.encode(r, add_special_tokens=False)) for r in t_rephrase] if t_rephrase else []
        if t_rephrase:
            t_edit_outer['labels'] = self.tok(t_trg, add_special_tokens=False, return_tensors="pt")["input_ids"]
        else:
            t_edit_outer['labels'] = None

        # loc
        t_loc = {}
        if t_loc_q and t_loc_a:
            t_loc['image'] = None
            t_loc['text_input'] = [" ".join([q, a]) for q, a in zip(t_loc_q, t_loc_a)]
            t_loc['labels'] = t_loc_a
            t_loc['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in t_loc_q]
            t_loc['labels'] = self.tok(t_loc_a, add_special_tokens=False, return_tensors="pt")["input_ids"]
        else:
            t_loc['image'] = None
            t_loc['text_input'] = []
            t_loc['labels'] = []
            t_loc['prompts_len'] = []

        # (Portability) 예시
        # 필요하면 textual_edit 쪽에도 추가 가능
        t_edit_ports = None
        # if 'portability_prompt' in b["textual_edit"].keys():
        #    ...


        # -----------------------------------------------------
        # 4) Compositional Portabiltiy (공통)
        # -----------------------------------------------------
        prompt_template = (
        "Visual Editing Knowledge: {visual_info}\n"
        "Text Editing Knowledge: {textual_info}\n"
        "--------------------------------\n"
        "Question: {question}\n"
        "Answer: "
         )
        if 'portability_prompt' in batch[0].keys():
            edit_ports = []
            for port_q, port_a in zip(batch[0]['portability_prompt'], batch[0]['portability_ground_truth']):
                port = {}
                # visual 정보: 예시로 batch[0]의 'prompt'와 'target' 사용
                visual_src = batch[0]['prompt']
                visual_trg = batch[0]['target']
                
                # textual 정보: textual_edit 부분에서 예시 가져오기 (존재한다면)
                t_src = batch[0].get('textual_edit', {}).get('prompt', '')
                t_trg = batch[0].get('textual_edit', {}).get('target', '')
                
                visual_info = f"{visual_src} -> {visual_trg}"
                textual_info = f"{t_src} -> {t_trg}"
                question = port_q  # portability 질문
                
                # 영어 프롬프트 생성
                prompt = prompt_template.format(
                    visual_info=visual_info,
                    textual_info=textual_info,
                    question=question
                )

                port['text_input'] = [" ".join([prompt, port_a])]  # 모델은 이 prompt를 기반으로 답변 생성
                port['labels'] = self.tok([port_a], add_special_tokens=False, return_tensors="pt")["input_ids"]
                port['prompts_len'] = [len(self.tok.encode(prompt, add_special_tokens=False))]
                # 예시로 이미지 정보도 필요하다면 할당
                port['image'] = torch.stack(image, dim=0)
                
                edit_ports.append(port)



        # if 'portability_prompt' in batch[0].keys():
        #     edit_ports = []
        #     for port_q, port_a in zip(batch[0]['portability_prompt'], batch[0]['portability_ground_truth']):
        #         port = {}
        #         port['image'] = torch.stack(image, dim=0)
        #         port['text_input'] = [' '.join([port_q, port_a])]
        #         port['labels'] = [port_a]
        #         port['prompts_len'] = [len(self.tok.encode(port_q, add_special_tokens=False))]
        #         port['labels'] = self.tok([port_a], add_special_tokens=False, return_tensors="pt")["input_ids"]
        #         edit_ports.append(port)


        # -----------------------------------------------------
        # 5) cond 토큰화 (공통)
        # -----------------------------------------------------
        cond = [b['cond'] for b in batch]
        cond = self.tok(
            cond,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
            truncation=True,
        ).to(self.config.device)

        # -----------------------------------------------------
        # 4) 최종 Batch 구성
        # -----------------------------------------------------
        # visual_edit와 textual_edit를 따로 구성
        visual_edit = {
            "edit_inner": v_edit_inner,
            "edit_outer": v_edit_outer,
            "edit_outer_image": v_edit_outer_image,
            "loc": v_loc,
            "loc_image": v_loc_image,
        }

        textual_edit = {
            "edit_inner": t_edit_inner,
            "edit_outer": t_edit_outer,
            "loc": t_loc,
            "port": t_edit_ports,  # 필요하면 정의
        }

        batch_dict = {
            "visual_edit": visual_edit,
            "textual_edit": textual_edit,
            "cond": cond,
            "port": edit_ports
        }

        # 모든 텐서를 self.config.device 로 이동(혹은 필요한 후처리)
        batch_dict = dict_to(batch_dict, self.config.device)
        return batch_dict

class CompositionalDataset_RAG_Simple(BaseDataset):
    def __init__(self, data_dir: str, size:  typing.Optional[int] = None, config=None, no_image=False, hop=None, *args, **kwargs):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        # get tokenizer and vis_processor
        if config.model_class == "Blip2OPT":
            vis_processor = BlipImageEvalProcessor(image_size=364, mean=None, std=None)
        elif config.model_class == "LLaVA":
            vis_processor = transformers.CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        else:
            raise NotImplementedError("unknown model class")
        # Load Tokenizer
        if (config is not None and hasattr(config, 'tokenizer_name')):
            tok_name = (
                config.tokenizer_name
                if config.tokenizer_name is not None
                else config.name
            )
            tokenizer = getattr(transformers, config.tokenizer_class).from_pretrained(
                tok_name, trust_remote_code=True
            )            
            if tokenizer.pad_token == None or tokenizer.pad_token == '':
                tokenizer.pad_token = tokenizer.eos_token  
                
        vis_root = config.coco_image
        rephrase_root = config.rephrase_image
        super().__init__(vis_processor, vis_root, rephrase_root, [data_dir]) # eval_multohop.json -> image 관련 설정?

        self.config = config
        self.tok = tokenizer
        self.max_length = 32

        self.prompt = "Question: {} Short answer: " # 어디에 쓰이지? 학습? Inference?

        data = []
        if size is not None:
            self.annotation = self.annotation[:size]
        if hop:
            self.hop = hop
            assert int(hop) in [1, 2, 3, 4], "hop should be 1, 2, 3, or 4"
            port_types = ['', '1-hop', '2-hop', '3-hop', '4-hop']
            port_type = port_types[int(hop)]
        for record in tqdm(self.annotation, ncols=120, desc='Loading Data'):
            
            if record['alt'] == "": # 편집할 내용 없으면 패스
                continue
            if hop and 'port_new' not in record.keys(): # Portability Data 아니면 패스(1-hop)
                continue
            
            ## Visual Edit Data ##
            # 이미지 관련 경로
            item = {
                'prompt': record['src'],
                'pred': record['pred'],
                'target': record['alt'],
                'rephrase_prompt': record['rephrase'],
                'image': os.path.join(self.vis_root, record["image"]),
                'image_rephrase': os.path.join(self.rephrase_root, record["image_rephrase"]),
                'cond': "{} >> {} || {}".format(record['pred'], record['alt'], record['src']),
                'locality_prompt': record['loc'],
                'locality_ground_truth': record['loc_ans'],
                'multimodal_locality_image': os.path.join(self.vis_root, record['m_loc']),
                'multimodal_locality_prompt': record['m_loc_q'],
                'multimodal_locality_ground_truth': record['m_loc_a']
            }

            
            ##  Textual Edit Data ##
            textual_pred = " ".join(record["textual_edit"]['pred']) if isinstance(record["textual_edit"]['pred'], list) else record["textual_edit"]['pred']
            textual_alt = " ".join(record["textual_edit"]['alt']) if isinstance(record["textual_edit"]['alt'], list) else record["textual_edit"]['alt']

            if "textual_edit" in record:
                item["textual_edit"] = {
                    "prompt": record["textual_edit"]["src"],
                    "pred": textual_pred,
                    "target": textual_alt, 
                    "rephrase_prompt": record["textual_edit"]["rephrase"],
                    'cond': "{} >> {} || {}".format(  # 원본(pred) -> 편집(alt)  ||  질문(src)
                        textual_pred,
                        textual_alt,
                        record["textual_edit"]['src']
                        ),
                    'locality_prompt': record["textual_edit"]['loc'],
                    'locality_ground_truth': record["textual_edit"]['loc_ans']
                }
                
            # Compositional portabiliy
            if 'port_new' in record.keys():
                item['portability_prompt'] = []
                item['portability_ground_truth'] = []
                find_hop = False
                for ports in record['port_new']:
                    if ports['port_type'] == port_type:
                        find_hop = True
                        port_q = ports['Q&A']['Question'] #  "What country is the city in the image part of?"
                        port_a = textual_alt              #  e: "Billings,_Montana"(before textual edit) -> e`: "United States"(after tex edit)
                        item['portability_prompt'].append(port_q)
                        item['portability_ground_truth'].append(port_a)
                        break
                
                if not find_hop:
                    continue

            
            data.append(item)

        # if size is not None:
        #     data = data[:size]        
        self._data = data
        self.no_image = no_image
    # collate_fn(batch)를 만들기위해 batch size 만큼 호출됨
    def __getitem__(self, index):
        if self.no_image:
            return self._data[index]

        data = deepcopy(self._data[index])     
        # load image
        image_path = data['image']
        rephrase_image_path = data['image_rephrase']
        locality_image_path = data['multimodal_locality_image']
        
        image = Image.open(image_path).convert("RGB")
        rephrase_image = Image.open(rephrase_image_path).convert("RGB")
        locality_image = Image.open(locality_image_path).convert("RGB")
        
        if self.config.model_class == "Blip2OPT":
            image = self.vis_processor(image)
            rephrase_image = self.vis_processor(rephrase_image)
            locality_image = self.vis_processor(locality_image)
        elif self.config.model_class == "LLaVA":
            image = self.vis_processor(image, return_tensors='pt')['pixel_values'].to(dtype=torch.float16)
            rephrase_image = self.vis_processor(rephrase_image, return_tensors='pt')['pixel_values'].to(dtype=torch.float16)
            locality_image = self.vis_processor(locality_image, return_tensors='pt')['pixel_values'].to(dtype=torch.float16)
        else:
            raise NotImplementedError

        data['image'] = image
        data['image_rephrase'] = rephrase_image
        data['multimodal_locality_image'] = locality_image

        return data
    
    def __len__(self):
        return len(self._data)

    def collate_fn(self, batch):
        # -----------------------------------------------------
        # 1) Visual Edit 파트
        # -----------------------------------------------------
        # 예: prompt, target, cond, rephrase, image 등등 가져오기
        src = [b['prompt'] for b in batch]
        trg = [b['target'] for b in batch]
        rephrase = [b['rephrase_prompt'] for b in batch]

        image = [b['image'] for b in batch]
        image_rephrase = [b['image_rephrase'] for b in batch]

        loc_q = [b["locality_prompt"] for b in batch]
        loc_a = [b["locality_ground_truth"] for b in batch]

        m_loc_image = [b['multimodal_locality_image'] for b in batch]
        m_loc_q = [b['multimodal_locality_prompt'] for b in batch]
        m_loc_a = [b['multimodal_locality_ground_truth'] for b in batch]

        # edit_inner
        v_edit_inner = {}
        v_edit_inner['image'] = torch.stack(image, dim=0)
        v_edit_inner['text_input'] = [" ".join([s, t]) for s, t in zip(src, trg)]
        v_edit_inner['labels'] = trg
        v_edit_inner['prompts_len'] = [len(self.tok.encode(s, add_special_tokens=False)) for s in src]
        v_edit_inner['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt")["input_ids"]

        # edit_outer
        v_edit_outer = {}
        v_edit_outer['image'] = torch.stack(image, dim=0)
        v_edit_outer['text_input'] = [" ".join([r, t]) for r, t in zip(rephrase, trg)]
        v_edit_outer['labels'] = trg
        v_edit_outer['prompts_len'] = [len(self.tok.encode(r, add_special_tokens=False)) for r in rephrase]
        v_edit_outer['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt")["input_ids"]

        # edit_outer_image
        v_edit_outer_image = {}
        v_edit_outer_image['image'] = torch.stack(image_rephrase, dim=0)
        v_edit_outer_image['text_input'] = [" ".join([s, t]) for s, t in zip(src, trg)]
        v_edit_outer_image['labels'] = trg
        v_edit_outer_image['prompts_len'] = [len(self.tok.encode(s, add_special_tokens=False)) for s in src]
        v_edit_outer_image['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt")["input_ids"]

        # loc
        v_loc = {}
        v_loc['image'] = None
        v_loc['text_input'] = [" ".join([q, a]) for q, a in zip(loc_q, loc_a)]
        v_loc['labels'] = loc_a
        v_loc['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in loc_q]
        v_loc['labels'] = self.tok(loc_a, add_special_tokens=False, return_tensors="pt")["input_ids"]

        # loc_image
        v_loc_image = {}
        v_loc_image['image'] = torch.stack(m_loc_image, dim=0)
        v_loc_image['text_input'] = [" ".join([q, a]) for q, a in zip(m_loc_q, m_loc_a)]
        v_loc_image['labels'] = m_loc_a
        v_loc_image['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in m_loc_q]
        v_loc_image['labels'] = self.tok(m_loc_a, add_special_tokens=False, return_tensors="pt")["input_ids"]

        # -----------------------------------------------------
        # 2) Textual Edit 파트 (기존 textual_edit 부분)
        # -----------------------------------------------------
        t_src = [b['textual_edit']['prompt'] for b in batch]
        t_trg = [b['textual_edit']['target'] for b in batch]
        # loc_q / loc_a도 텍스트 기반으로 사용된다면 추가
        t_loc_q = [b.get("textual_edit", {}).get("locality_prompt", "") for b in batch]
        t_loc_a = [b.get("textual_edit", {}).get("locality_ground_truth", "") for b in batch]

        # edit_inner
        t_edit_inner = {}
        t_edit_inner['image'] = None
        t_edit_inner['text_input'] = [" ".join([s, t]) for s, t in zip(t_src, t_trg)]
        t_edit_inner['labels'] = t_trg
        t_edit_inner['prompts_len'] = [len(self.tok.encode(s, add_special_tokens=False)) for s in t_src]
        t_edit_inner['labels'] = self.tok(t_trg, add_special_tokens=False, return_tensors="pt")["input_ids"]

        # edit_outer
        # rephrase_prompt가 textual_edit 안에 있다고 가정
        t_rephrase = [b.get("textual_edit", {}).get("rephrase_prompt", "") for b in batch]
        t_edit_outer = {}
        t_edit_outer['image'] = None
        t_edit_outer['text_input'] = [" ".join([r, t]) for r, t in zip(t_rephrase, t_trg)] if t_rephrase else []
        t_edit_outer['labels'] = t_trg
        t_edit_outer['prompts_len'] = [len(self.tok.encode(r, add_special_tokens=False)) for r in t_rephrase] if t_rephrase else []
        if t_rephrase:
            t_edit_outer['labels'] = self.tok(t_trg, add_special_tokens=False, return_tensors="pt")["input_ids"]
        else:
            t_edit_outer['labels'] = None

        # loc
        t_loc = {}
        if t_loc_q and t_loc_a:
            t_loc['image'] = None
            t_loc['text_input'] = [" ".join([q, a]) for q, a in zip(t_loc_q, t_loc_a)]
            t_loc['labels'] = t_loc_a
            t_loc['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in t_loc_q]
            t_loc['labels'] = self.tok(t_loc_a, add_special_tokens=False, return_tensors="pt")["input_ids"]
        else:
            t_loc['image'] = None
            t_loc['text_input'] = []
            t_loc['labels'] = []
            t_loc['prompts_len'] = []

        # (Portability) 예시
        # 필요하면 textual_edit 쪽에도 추가 가능
        t_edit_ports = None
        # if 'portability_prompt' in b["textual_edit"].keys():
        #    ...


        # -----------------------------------------------------
        # 4) Compositional Portabiltiy (공통)
        # -----------------------------------------------------
        prompt_template = (
        "{visual_info} {textual_info} {question} "
         )
        if 'portability_prompt' in batch[0].keys():
            edit_ports = []
            for port_q, port_a in zip(batch[0]['portability_prompt'], batch[0]['portability_ground_truth']):
                port = {}
                # visual 정보: 예시로 batch[0]의 'prompt'와 'target' 사용
                visual_src = batch[0]['prompt']
                visual_trg = batch[0]['target']
                
                # textual 정보: textual_edit 부분에서 예시 가져오기 (존재한다면)
                t_src = batch[0].get('textual_edit', {}).get('prompt', '')
                t_trg = batch[0].get('textual_edit', {}).get('target', '')
                
                visual_info = f"{visual_src} {visual_trg}"
                textual_info = f"{t_src} {t_trg}"
                question = port_q  # portability 질문
                
                # 영어 프롬프트 생성
                prompt = prompt_template.format(
                    visual_info=visual_info,
                    textual_info=textual_info,
                    question=question
                )

                port['text_input'] = [" ".join([prompt, port_a])]  # 모델은 이 prompt를 기반으로 답변 생성
                port['labels'] = self.tok([port_a], add_special_tokens=False, return_tensors="pt")["input_ids"]
                port['prompts_len'] = [len(self.tok.encode(prompt, add_special_tokens=False))]
                # 예시로 이미지 정보도 필요하다면 할당
                port['image'] = torch.stack(image, dim=0)
                
                edit_ports.append(port)



        # if 'portability_prompt' in batch[0].keys():
        #     edit_ports = []
        #     for port_q, port_a in zip(batch[0]['portability_prompt'], batch[0]['portability_ground_truth']):
        #         port = {}
        #         port['image'] = torch.stack(image, dim=0)
        #         port['text_input'] = [' '.join([port_q, port_a])]
        #         port['labels'] = [port_a]
        #         port['prompts_len'] = [len(self.tok.encode(port_q, add_special_tokens=False))]
        #         port['labels'] = self.tok([port_a], add_special_tokens=False, return_tensors="pt")["input_ids"]
        #         edit_ports.append(port)


        # -----------------------------------------------------
        # 5) cond 토큰화 (공통)
        # -----------------------------------------------------
        cond = [b['cond'] for b in batch]
        cond = self.tok(
            cond,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
            truncation=True,
        ).to(self.config.device)

        # -----------------------------------------------------
        # 4) 최종 Batch 구성
        # -----------------------------------------------------
        # visual_edit와 textual_edit를 따로 구성
        visual_edit = {
            "edit_inner": v_edit_inner,
            "edit_outer": v_edit_outer,
            "edit_outer_image": v_edit_outer_image,
            "loc": v_loc,
            "loc_image": v_loc_image,
        }

        textual_edit = {
            "edit_inner": t_edit_inner,
            "edit_outer": t_edit_outer,
            "loc": t_loc,
            "port": t_edit_ports,  # 필요하면 정의
        }

        batch_dict = {
            "visual_edit": visual_edit,
            "textual_edit": textual_edit,
            "cond": cond,
            "port": edit_ports
        }

        # 모든 텐서를 self.config.device 로 이동(혹은 필요한 후처리)
        batch_dict = dict_to(batch_dict, self.config.device)
        return batch_dict
    
class CompositionalDataset_RAG_70(BaseDataset):
    def __init__(self, data_dir: str, size:  typing.Optional[int] = None, config=None, no_image=False, hop=None, *args, **kwargs):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        # get tokenizer and vis_processor
        if config.model_class == "Blip2OPT":
            vis_processor = BlipImageEvalProcessor(image_size=364, mean=None, std=None)
        elif config.model_class == "LLaVA":
            vis_processor = transformers.CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        else:
            raise NotImplementedError("unknown model class")
        # Load Tokenizer
        if (config is not None and hasattr(config, 'tokenizer_name')):
            tok_name = (
                config.tokenizer_name
                if config.tokenizer_name is not None
                else config.name
            )
            tokenizer = getattr(transformers, config.tokenizer_class).from_pretrained(
                tok_name, trust_remote_code=True
            )            
            if tokenizer.pad_token == None or tokenizer.pad_token == '':
                tokenizer.pad_token = tokenizer.eos_token  
                
        vis_root = config.coco_image
        rephrase_root = config.rephrase_image
        super().__init__(vis_processor, vis_root, rephrase_root, [data_dir]) # eval_multohop.json -> image 관련 설정?

        self.config = config
        self.tok = tokenizer
        self.max_length = 32

        self.prompt = "Question: {} Short answer: " # 어디에 쓰이지? 학습? Inference?

        data = []
        if size is not None:
            self.annotation = self.annotation[:size]
        if hop:
            self.hop = hop
            assert int(hop) in [1, 2, 3, 4], "hop should be 1, 2, 3, or 4"
            port_types = ['', '1-hop', '2-hop', '3-hop', '4-hop']
            port_type = port_types[int(hop)]
        for record in tqdm(self.annotation, ncols=120, desc='Loading Data'):
            
            if record['alt'] == "": # 편집할 내용 없으면 패스
                continue
            if hop and 'port_new' not in record.keys(): # Portability Data 아니면 패스(1-hop)
                continue
            
            ## Visual Edit Data ##
            # 이미지 관련 경로
            item = {
                'prompt': record['src'],
                'pred': record['pred'],
                'target': record['alt'],
                'rephrase_prompt': record['rephrase'],
                'image': os.path.join(self.vis_root, record["image"]),
                'image_rephrase': os.path.join(self.rephrase_root, record["image_rephrase"]),
                'cond': "{} >> {} || {}".format(record['pred'], record['alt'], record['src']),
                'locality_prompt': record['loc'],
                'locality_ground_truth': record['loc_ans'],
                'multimodal_locality_image': os.path.join(self.vis_root, record['m_loc']),
                'multimodal_locality_prompt': record['m_loc_q'],
                'multimodal_locality_ground_truth': record['m_loc_a']
            }

            
            ##  Textual Edit Data ##
            textual_pred = " ".join(record["textual_edit"]['pred']) if isinstance(record["textual_edit"]['pred'], list) else record["textual_edit"]['pred']
            textual_alt = " ".join(record["textual_edit"]['alt']) if isinstance(record["textual_edit"]['alt'], list) else record["textual_edit"]['alt']

            if "textual_edit" in record:
                item["textual_edit"] = {
                    "prompt": record["textual_edit"]["src"],
                    "pred": textual_pred,
                    "target": textual_alt, 
                    "rephrase_prompt": record["textual_edit"]["rephrase"],
                    'cond': "{} >> {} || {}".format(  # 원본(pred) -> 편집(alt)  ||  질문(src)
                        textual_pred,
                        textual_alt,
                        record["textual_edit"]['src']
                        ),
                    'locality_prompt': record["textual_edit"]['loc'],
                    'locality_ground_truth': record["textual_edit"]['loc_ans']
                }
                
            # Compositional portabiliy
            if 'port_new' in record.keys():
                item['portability_prompt'] = []
                item['portability_ground_truth'] = []
                find_hop = False
                for ports in record['port_new']:
                    if ports['port_type'] == port_type:
                        find_hop = True
                        port_q = ports['Q&A']['Question'] #  "What country is the city in the image part of?"
                        port_a = textual_alt              #  e: "Billings,_Montana"(before textual edit) -> e`: "United States"(after tex edit)
                        item['portability_prompt'].append(port_q)
                        item['portability_ground_truth'].append(port_a)
                        break
                
                if not find_hop:
                    continue

            
            data.append(item)

        # if size is not None:
        #     data = data[:size]        
        self._data = data
        self.no_image = no_image
    # collate_fn(batch)를 만들기위해 batch size 만큼 호출됨
    def __getitem__(self, index):
        if self.no_image:
            return self._data[index]

        data = deepcopy(self._data[index])     
        # load image
        image_path = data['image']
        rephrase_image_path = data['image_rephrase']
        locality_image_path = data['multimodal_locality_image']
        
        image = Image.open(image_path).convert("RGB")
        rephrase_image = Image.open(rephrase_image_path).convert("RGB")
        locality_image = Image.open(locality_image_path).convert("RGB")
        
        if self.config.model_class == "Blip2OPT":
            image = self.vis_processor(image)
            rephrase_image = self.vis_processor(rephrase_image)
            locality_image = self.vis_processor(locality_image)
        elif self.config.model_class == "LLaVA":
            image = self.vis_processor(image, return_tensors='pt')['pixel_values'].to(dtype=torch.float16)
            rephrase_image = self.vis_processor(rephrase_image, return_tensors='pt')['pixel_values'].to(dtype=torch.float16)
            locality_image = self.vis_processor(locality_image, return_tensors='pt')['pixel_values'].to(dtype=torch.float16)
        else:
            raise NotImplementedError

        data['image'] = image
        data['image_rephrase'] = rephrase_image
        data['multimodal_locality_image'] = locality_image

        return data
    
    def __len__(self):
        return len(self._data)

    def collate_fn(self, batch):
        # -----------------------------------------------------
        # 1) Visual Edit 파트
        # -----------------------------------------------------
        # 예: prompt, target, cond, rephrase, image 등등 가져오기
        src = [b['prompt'] for b in batch]
        trg = [b['target'] for b in batch]
        rephrase = [b['rephrase_prompt'] for b in batch]

        image = [b['image'] for b in batch]
        image_rephrase = [b['image_rephrase'] for b in batch]

        loc_q = [b["locality_prompt"] for b in batch]
        loc_a = [b["locality_ground_truth"] for b in batch]

        m_loc_image = [b['multimodal_locality_image'] for b in batch]
        m_loc_q = [b['multimodal_locality_prompt'] for b in batch]
        m_loc_a = [b['multimodal_locality_ground_truth'] for b in batch]

        # edit_inner
        v_edit_inner = {}
        v_edit_inner['image'] = torch.stack(image, dim=0)
        v_edit_inner['text_input'] = [" ".join([s, t]) for s, t in zip(src, trg)]
        v_edit_inner['labels'] = trg
        v_edit_inner['prompts_len'] = [len(self.tok.encode(s, add_special_tokens=False)) for s in src]
        v_edit_inner['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt")["input_ids"]

        # edit_outer
        v_edit_outer = {}
        v_edit_outer['image'] = torch.stack(image, dim=0)
        v_edit_outer['text_input'] = [" ".join([r, t]) for r, t in zip(rephrase, trg)]
        v_edit_outer['labels'] = trg
        v_edit_outer['prompts_len'] = [len(self.tok.encode(r, add_special_tokens=False)) for r in rephrase]
        v_edit_outer['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt")["input_ids"]

        # edit_outer_image
        v_edit_outer_image = {}
        v_edit_outer_image['image'] = torch.stack(image_rephrase, dim=0)
        v_edit_outer_image['text_input'] = [" ".join([s, t]) for s, t in zip(src, trg)]
        v_edit_outer_image['labels'] = trg
        v_edit_outer_image['prompts_len'] = [len(self.tok.encode(s, add_special_tokens=False)) for s in src]
        v_edit_outer_image['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt")["input_ids"]

        # loc
        v_loc = {}
        v_loc['image'] = None
        v_loc['text_input'] = [" ".join([q, a]) for q, a in zip(loc_q, loc_a)]
        v_loc['labels'] = loc_a
        v_loc['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in loc_q]
        v_loc['labels'] = self.tok(loc_a, add_special_tokens=False, return_tensors="pt")["input_ids"]

        # loc_image
        v_loc_image = {}
        v_loc_image['image'] = torch.stack(m_loc_image, dim=0)
        v_loc_image['text_input'] = [" ".join([q, a]) for q, a in zip(m_loc_q, m_loc_a)]
        v_loc_image['labels'] = m_loc_a
        v_loc_image['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in m_loc_q]
        v_loc_image['labels'] = self.tok(m_loc_a, add_special_tokens=False, return_tensors="pt")["input_ids"]

        # -----------------------------------------------------
        # 2) Textual Edit 파트 (기존 textual_edit 부분)
        # -----------------------------------------------------
        t_src = [b['textual_edit']['prompt'] for b in batch]
        t_trg = [b['textual_edit']['target'] for b in batch]
        # loc_q / loc_a도 텍스트 기반으로 사용된다면 추가
        t_loc_q = [b.get("textual_edit", {}).get("locality_prompt", "") for b in batch]
        t_loc_a = [b.get("textual_edit", {}).get("locality_ground_truth", "") for b in batch]

        # edit_inner
        t_edit_inner = {}
        t_edit_inner['image'] = None
        t_edit_inner['text_input'] = [" ".join([s, t]) for s, t in zip(t_src, t_trg)]
        t_edit_inner['labels'] = t_trg
        t_edit_inner['prompts_len'] = [len(self.tok.encode(s, add_special_tokens=False)) for s in t_src]
        t_edit_inner['labels'] = self.tok(t_trg, add_special_tokens=False, return_tensors="pt")["input_ids"]

        # edit_outer
        # rephrase_prompt가 textual_edit 안에 있다고 가정
        t_rephrase = [b.get("textual_edit", {}).get("rephrase_prompt", "") for b in batch]
        t_edit_outer = {}
        t_edit_outer['image'] = None
        t_edit_outer['text_input'] = [" ".join([r, t]) for r, t in zip(t_rephrase, t_trg)] if t_rephrase else []
        t_edit_outer['labels'] = t_trg
        t_edit_outer['prompts_len'] = [len(self.tok.encode(r, add_special_tokens=False)) for r in t_rephrase] if t_rephrase else []
        if t_rephrase:
            t_edit_outer['labels'] = self.tok(t_trg, add_special_tokens=False, return_tensors="pt")["input_ids"]
        else:
            t_edit_outer['labels'] = None

        # loc
        t_loc = {}
        if t_loc_q and t_loc_a:
            t_loc['image'] = None
            t_loc['text_input'] = [" ".join([q, a]) for q, a in zip(t_loc_q, t_loc_a)]
            t_loc['labels'] = t_loc_a
            t_loc['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in t_loc_q]
            t_loc['labels'] = self.tok(t_loc_a, add_special_tokens=False, return_tensors="pt")["input_ids"]
        else:
            t_loc['image'] = None
            t_loc['text_input'] = []
            t_loc['labels'] = []
            t_loc['prompts_len'] = []

        # (Portability) 예시
        # 필요하면 textual_edit 쪽에도 추가 가능
        t_edit_ports = None
        # if 'portability_prompt' in b["textual_edit"].keys():
        #    ...


        # -----------------------------------------------------
        # 4) Compositional Portabiltiy (공통)
        # -----------------------------------------------------
        prompt_template = (
        "Visual Editing Knowledge: {visual_info}\n"
        "Text Editing Knowledge: {textual_info}\n"
        "--------------------------------\n"
        "Question: {question}\n"
        "Answer: "
         )
        if 'portability_prompt' in batch[0].keys():
            edit_ports = []
            for port_q, port_a in zip(batch[0]['portability_prompt'], batch[0]['portability_ground_truth']):
                port = {}
                
                ## 70% 정확도 검색 가정 ##
                # visual edit data
                if random.random() < 0.84:
                    visual_src = batch[0]['prompt']
                    visual_trg = batch[0]['target']
                else:
                    rand_idx = random.choice(range(len(self._data)))
                    random_item = self._data[rand_idx]
                    visual_src = random_item['prompt']
                    visual_trg = random_item['target']

                # textual edit data
                if random.random() < 0.84:
                    t_src = batch[0].get('textual_edit', {}).get('prompt', '')
                    t_trg = batch[0].get('textual_edit', {}).get('target', '')
                else:
                    rand_idx = random.choice(range(len(self._data)))
                    random_item = self._data[rand_idx]
                    t_src = random_item.get('textual_edit', {}).get('prompt', '')
                    t_trg = random_item.get('textual_edit', {}).get('target', '')
                
                # 검색된 Prompt로 format 작성
                visual_info = f"{visual_src} -> {visual_trg}"
                textual_info = f"{t_src} -> {t_trg}"
                question = port_q  # portability 질문
                
                # 영어 프롬프트 생성
                prompt = prompt_template.format(
                    visual_info=visual_info,
                    textual_info=textual_info,
                    question=question
                )

                port['text_input'] = [" ".join([prompt, port_a])]  # 모델은 이 prompt를 기반으로 답변 생성
                port['labels'] = self.tok([port_a], add_special_tokens=False, return_tensors="pt")["input_ids"]
                port['prompts_len'] = [len(self.tok.encode(prompt, add_special_tokens=False))]
                # 예시로 이미지 정보도 필요하다면 할당
                port['image'] = torch.stack(image, dim=0)
                
                edit_ports.append(port)



        # if 'portability_prompt' in batch[0].keys():
        #     edit_ports = []
        #     for port_q, port_a in zip(batch[0]['portability_prompt'], batch[0]['portability_ground_truth']):
        #         port = {}
        #         port['image'] = torch.stack(image, dim=0)
        #         port['text_input'] = [' '.join([port_q, port_a])]
        #         port['labels'] = [port_a]
        #         port['prompts_len'] = [len(self.tok.encode(port_q, add_special_tokens=False))]
        #         port['labels'] = self.tok([port_a], add_special_tokens=False, return_tensors="pt")["input_ids"]
        #         edit_ports.append(port)


        # -----------------------------------------------------
        # 5) cond 토큰화 (공통)
        # -----------------------------------------------------
        cond = [b['cond'] for b in batch]
        cond = self.tok(
            cond,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
            truncation=True,
        ).to(self.config.device)

        # -----------------------------------------------------
        # 4) 최종 Batch 구성
        # -----------------------------------------------------
        # visual_edit와 textual_edit를 따로 구성
        visual_edit = {
            "edit_inner": v_edit_inner,
            "edit_outer": v_edit_outer,
            "edit_outer_image": v_edit_outer_image,
            "loc": v_loc,
            "loc_image": v_loc_image,
        }

        textual_edit = {
            "edit_inner": t_edit_inner,
            "edit_outer": t_edit_outer,
            "loc": t_loc,
            "port": t_edit_ports,  # 필요하면 정의
        }

        batch_dict = {
            "visual_edit": visual_edit,
            "textual_edit": textual_edit,
            "cond": cond,
            "port": edit_ports
        }

        # 모든 텐서를 self.config.device 로 이동(혹은 필요한 후처리)
        batch_dict = dict_to(batch_dict, self.config.device)
        return batch_dict

class CompositionalDataset_RAG_50(BaseDataset):
    def __init__(self, data_dir: str, size:  typing.Optional[int] = None, config=None, no_image=False, hop=None, *args, **kwargs):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        # get tokenizer and vis_processor
        if config.model_class == "Blip2OPT":
            vis_processor = BlipImageEvalProcessor(image_size=364, mean=None, std=None)
        elif config.model_class == "LLaVA":
            vis_processor = transformers.CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        else:
            raise NotImplementedError("unknown model class")
        # Load Tokenizer
        if (config is not None and hasattr(config, 'tokenizer_name')):
            tok_name = (
                config.tokenizer_name
                if config.tokenizer_name is not None
                else config.name
            )
            tokenizer = getattr(transformers, config.tokenizer_class).from_pretrained(
                tok_name, trust_remote_code=True
            )            
            if tokenizer.pad_token == None or tokenizer.pad_token == '':
                tokenizer.pad_token = tokenizer.eos_token  
                
        vis_root = config.coco_image
        rephrase_root = config.rephrase_image
        super().__init__(vis_processor, vis_root, rephrase_root, [data_dir]) # eval_multohop.json -> image 관련 설정?

        self.config = config
        self.tok = tokenizer
        self.max_length = 32

        self.prompt = "Question: {} Short answer: " # 어디에 쓰이지? 학습? Inference?

        data = []
        if size is not None:
            self.annotation = self.annotation[:size]
        if hop:
            self.hop = hop
            assert int(hop) in [1, 2, 3, 4], "hop should be 1, 2, 3, or 4"
            port_types = ['', '1-hop', '2-hop', '3-hop', '4-hop']
            port_type = port_types[int(hop)]
        for record in tqdm(self.annotation, ncols=120, desc='Loading Data'):
            
            if record['alt'] == "": # 편집할 내용 없으면 패스
                continue
            if hop and 'port_new' not in record.keys(): # Portability Data 아니면 패스(1-hop)
                continue
            
            ## Visual Edit Data ##
            # 이미지 관련 경로
            item = {
                'prompt': record['src'],
                'pred': record['pred'],
                'target': record['alt'],
                'rephrase_prompt': record['rephrase'],
                'image': os.path.join(self.vis_root, record["image"]),
                'image_rephrase': os.path.join(self.rephrase_root, record["image_rephrase"]),
                'cond': "{} >> {} || {}".format(record['pred'], record['alt'], record['src']),
                'locality_prompt': record['loc'],
                'locality_ground_truth': record['loc_ans'],
                'multimodal_locality_image': os.path.join(self.vis_root, record['m_loc']),
                'multimodal_locality_prompt': record['m_loc_q'],
                'multimodal_locality_ground_truth': record['m_loc_a']
            }

            
            ##  Textual Edit Data ##
            textual_pred = " ".join(record["textual_edit"]['pred']) if isinstance(record["textual_edit"]['pred'], list) else record["textual_edit"]['pred']
            textual_alt = " ".join(record["textual_edit"]['alt']) if isinstance(record["textual_edit"]['alt'], list) else record["textual_edit"]['alt']

            if "textual_edit" in record:
                item["textual_edit"] = {
                    "prompt": record["textual_edit"]["src"],
                    "pred": textual_pred,
                    "target": textual_alt, 
                    "rephrase_prompt": record["textual_edit"]["rephrase"],
                    'cond': "{} >> {} || {}".format(  # 원본(pred) -> 편집(alt)  ||  질문(src)
                        textual_pred,
                        textual_alt,
                        record["textual_edit"]['src']
                        ),
                    'locality_prompt': record["textual_edit"]['loc'],
                    'locality_ground_truth': record["textual_edit"]['loc_ans']
                }
                
            # Compositional portabiliy
            if 'port_new' in record.keys():
                item['portability_prompt'] = []
                item['portability_ground_truth'] = []
                find_hop = False
                for ports in record['port_new']:
                    if ports['port_type'] == port_type:
                        find_hop = True
                        port_q = ports['Q&A']['Question'] #  "What country is the city in the image part of?"
                        port_a = textual_alt              #  e: "Billings,_Montana"(before textual edit) -> e`: "United States"(after tex edit)
                        item['portability_prompt'].append(port_q)
                        item['portability_ground_truth'].append(port_a)
                        break
                
                if not find_hop:
                    continue

            
            data.append(item)

        # if size is not None:
        #     data = data[:size]        
        self._data = data
        self.no_image = no_image
    # collate_fn(batch)를 만들기위해 batch size 만큼 호출됨
    def __getitem__(self, index):
        if self.no_image:
            return self._data[index]

        data = deepcopy(self._data[index])     
        # load image
        image_path = data['image']
        rephrase_image_path = data['image_rephrase']
        locality_image_path = data['multimodal_locality_image']
        
        image = Image.open(image_path).convert("RGB")
        rephrase_image = Image.open(rephrase_image_path).convert("RGB")
        locality_image = Image.open(locality_image_path).convert("RGB")
        
        if self.config.model_class == "Blip2OPT":
            image = self.vis_processor(image)
            rephrase_image = self.vis_processor(rephrase_image)
            locality_image = self.vis_processor(locality_image)
        elif self.config.model_class == "LLaVA":
            image = self.vis_processor(image, return_tensors='pt')['pixel_values'].to(dtype=torch.float16)
            rephrase_image = self.vis_processor(rephrase_image, return_tensors='pt')['pixel_values'].to(dtype=torch.float16)
            locality_image = self.vis_processor(locality_image, return_tensors='pt')['pixel_values'].to(dtype=torch.float16)
        else:
            raise NotImplementedError

        data['image'] = image
        data['image_rephrase'] = rephrase_image
        data['multimodal_locality_image'] = locality_image

        return data
    
    def __len__(self):
        return len(self._data)

    def collate_fn(self, batch):
        # -----------------------------------------------------
        # 1) Visual Edit 파트
        # -----------------------------------------------------
        # 예: prompt, target, cond, rephrase, image 등등 가져오기
        src = [b['prompt'] for b in batch]
        trg = [b['target'] for b in batch]
        rephrase = [b['rephrase_prompt'] for b in batch]

        image = [b['image'] for b in batch]
        image_rephrase = [b['image_rephrase'] for b in batch]

        loc_q = [b["locality_prompt"] for b in batch]
        loc_a = [b["locality_ground_truth"] for b in batch]

        m_loc_image = [b['multimodal_locality_image'] for b in batch]
        m_loc_q = [b['multimodal_locality_prompt'] for b in batch]
        m_loc_a = [b['multimodal_locality_ground_truth'] for b in batch]

        # edit_inner
        v_edit_inner = {}
        v_edit_inner['image'] = torch.stack(image, dim=0)
        v_edit_inner['text_input'] = [" ".join([s, t]) for s, t in zip(src, trg)]
        v_edit_inner['labels'] = trg
        v_edit_inner['prompts_len'] = [len(self.tok.encode(s, add_special_tokens=False)) for s in src]
        v_edit_inner['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt")["input_ids"]

        # edit_outer
        v_edit_outer = {}
        v_edit_outer['image'] = torch.stack(image, dim=0)
        v_edit_outer['text_input'] = [" ".join([r, t]) for r, t in zip(rephrase, trg)]
        v_edit_outer['labels'] = trg
        v_edit_outer['prompts_len'] = [len(self.tok.encode(r, add_special_tokens=False)) for r in rephrase]
        v_edit_outer['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt")["input_ids"]

        # edit_outer_image
        v_edit_outer_image = {}
        v_edit_outer_image['image'] = torch.stack(image_rephrase, dim=0)
        v_edit_outer_image['text_input'] = [" ".join([s, t]) for s, t in zip(src, trg)]
        v_edit_outer_image['labels'] = trg
        v_edit_outer_image['prompts_len'] = [len(self.tok.encode(s, add_special_tokens=False)) for s in src]
        v_edit_outer_image['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt")["input_ids"]

        # loc
        v_loc = {}
        v_loc['image'] = None
        v_loc['text_input'] = [" ".join([q, a]) for q, a in zip(loc_q, loc_a)]
        v_loc['labels'] = loc_a
        v_loc['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in loc_q]
        v_loc['labels'] = self.tok(loc_a, add_special_tokens=False, return_tensors="pt")["input_ids"]

        # loc_image
        v_loc_image = {}
        v_loc_image['image'] = torch.stack(m_loc_image, dim=0)
        v_loc_image['text_input'] = [" ".join([q, a]) for q, a in zip(m_loc_q, m_loc_a)]
        v_loc_image['labels'] = m_loc_a
        v_loc_image['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in m_loc_q]
        v_loc_image['labels'] = self.tok(m_loc_a, add_special_tokens=False, return_tensors="pt")["input_ids"]

        # -----------------------------------------------------
        # 2) Textual Edit 파트 (기존 textual_edit 부분)
        # -----------------------------------------------------
        t_src = [b['textual_edit']['prompt'] for b in batch]
        t_trg = [b['textual_edit']['target'] for b in batch]
        # loc_q / loc_a도 텍스트 기반으로 사용된다면 추가
        t_loc_q = [b.get("textual_edit", {}).get("locality_prompt", "") for b in batch]
        t_loc_a = [b.get("textual_edit", {}).get("locality_ground_truth", "") for b in batch]

        # edit_inner
        t_edit_inner = {}
        t_edit_inner['image'] = None
        t_edit_inner['text_input'] = [" ".join([s, t]) for s, t in zip(t_src, t_trg)]
        t_edit_inner['labels'] = t_trg
        t_edit_inner['prompts_len'] = [len(self.tok.encode(s, add_special_tokens=False)) for s in t_src]
        t_edit_inner['labels'] = self.tok(t_trg, add_special_tokens=False, return_tensors="pt")["input_ids"]

        # edit_outer
        # rephrase_prompt가 textual_edit 안에 있다고 가정
        t_rephrase = [b.get("textual_edit", {}).get("rephrase_prompt", "") for b in batch]
        t_edit_outer = {}
        t_edit_outer['image'] = None
        t_edit_outer['text_input'] = [" ".join([r, t]) for r, t in zip(t_rephrase, t_trg)] if t_rephrase else []
        t_edit_outer['labels'] = t_trg
        t_edit_outer['prompts_len'] = [len(self.tok.encode(r, add_special_tokens=False)) for r in t_rephrase] if t_rephrase else []
        if t_rephrase:
            t_edit_outer['labels'] = self.tok(t_trg, add_special_tokens=False, return_tensors="pt")["input_ids"]
        else:
            t_edit_outer['labels'] = None

        # loc
        t_loc = {}
        if t_loc_q and t_loc_a:
            t_loc['image'] = None
            t_loc['text_input'] = [" ".join([q, a]) for q, a in zip(t_loc_q, t_loc_a)]
            t_loc['labels'] = t_loc_a
            t_loc['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in t_loc_q]
            t_loc['labels'] = self.tok(t_loc_a, add_special_tokens=False, return_tensors="pt")["input_ids"]
        else:
            t_loc['image'] = None
            t_loc['text_input'] = []
            t_loc['labels'] = []
            t_loc['prompts_len'] = []

        # (Portability) 예시
        # 필요하면 textual_edit 쪽에도 추가 가능
        t_edit_ports = None
        # if 'portability_prompt' in b["textual_edit"].keys():
        #    ...


        # -----------------------------------------------------
        # 4) Compositional Portabiltiy (공통)
        # -----------------------------------------------------
        prompt_template = (
        "Visual Editing Knowledge: {visual_info}\n"
        "Text Editing Knowledge: {textual_info}\n"
        "--------------------------------\n"
        "Question: {question}\n"
        "Answer: "
         )
        if 'portability_prompt' in batch[0].keys():
            edit_ports = []
            for port_q, port_a in zip(batch[0]['portability_prompt'], batch[0]['portability_ground_truth']):
                port = {}
                
                ## 70% 정확도 검색 가정 ##
                # visual edit data
                if random.random() < 0.70:
                    visual_src = batch[0]['prompt']
                    visual_trg = batch[0]['target']
                else:
                    rand_idx = random.choice(range(len(self._data)))
                    random_item = self._data[rand_idx]
                    visual_src = random_item['prompt']
                    visual_trg = random_item['target']

                # textual edit data
                if random.random() < 0.70:
                    t_src = batch[0].get('textual_edit', {}).get('prompt', '')
                    t_trg = batch[0].get('textual_edit', {}).get('target', '')
                else:
                    rand_idx = random.choice(range(len(self._data)))
                    random_item = self._data[rand_idx]
                    t_src = random_item.get('textual_edit', {}).get('prompt', '')
                    t_trg = random_item.get('textual_edit', {}).get('target', '')
                
                # 검색된 Prompt로 format 작성
                visual_info = f"{visual_src} -> {visual_trg}"
                textual_info = f"{t_src} -> {t_trg}"
                question = port_q  # portability 질문
                
                # 영어 프롬프트 생성
                prompt = prompt_template.format(
                    visual_info=visual_info,
                    textual_info=textual_info,
                    question=question
                )

                port['text_input'] = [" ".join([prompt, port_a])]  # 모델은 이 prompt를 기반으로 답변 생성
                port['labels'] = self.tok([port_a], add_special_tokens=False, return_tensors="pt")["input_ids"]
                port['prompts_len'] = [len(self.tok.encode(prompt, add_special_tokens=False))]
                # 예시로 이미지 정보도 필요하다면 할당
                port['image'] = torch.stack(image, dim=0)
                
                edit_ports.append(port)



        # if 'portability_prompt' in batch[0].keys():
        #     edit_ports = []
        #     for port_q, port_a in zip(batch[0]['portability_prompt'], batch[0]['portability_ground_truth']):
        #         port = {}
        #         port['image'] = torch.stack(image, dim=0)
        #         port['text_input'] = [' '.join([port_q, port_a])]
        #         port['labels'] = [port_a]
        #         port['prompts_len'] = [len(self.tok.encode(port_q, add_special_tokens=False))]
        #         port['labels'] = self.tok([port_a], add_special_tokens=False, return_tensors="pt")["input_ids"]
        #         edit_ports.append(port)


        # -----------------------------------------------------
        # 5) cond 토큰화 (공통)
        # -----------------------------------------------------
        cond = [b['cond'] for b in batch]
        cond = self.tok(
            cond,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
            truncation=True,
        ).to(self.config.device)

        # -----------------------------------------------------
        # 4) 최종 Batch 구성
        # -----------------------------------------------------
        # visual_edit와 textual_edit를 따로 구성
        visual_edit = {
            "edit_inner": v_edit_inner,
            "edit_outer": v_edit_outer,
            "edit_outer_image": v_edit_outer_image,
            "loc": v_loc,
            "loc_image": v_loc_image,
        }

        textual_edit = {
            "edit_inner": t_edit_inner,
            "edit_outer": t_edit_outer,
            "loc": t_loc,
            "port": t_edit_ports,  # 필요하면 정의
        }

        batch_dict = {
            "visual_edit": visual_edit,
            "textual_edit": textual_edit,
            "cond": cond,
            "port": edit_ports
        }

        # 모든 텐서를 self.config.device 로 이동(혹은 필요한 후처리)
        batch_dict = dict_to(batch_dict, self.config.device)
        return batch_dict

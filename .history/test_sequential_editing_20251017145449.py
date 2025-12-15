import os
import torch
import types
from statistics import mean

from easyeditor import BaseEditor, MultimodalTrainer, MultimodalEditor
from easyeditor import CaptionDataset, VQADataset, TextualDataset, CompositionalDataset, CompositionalDataset_RAG, CompositionalDataset_RAG_Simple, CompositionalDataset_RAG_70, CompositionalDataset_RAG_50
from easyeditor import MENDMultimodalTrainingHparams, SERACMultimodalTrainingHparams, IKEMultimodalHyperParams, MENDMultimodalHparams \
    , SERACMultimodalHparams, FTMultimodalHparams
from easyeditor import encode_ike_facts_multimodal
from sentence_transformers import SentenceTransformer
import sys
from datetime import datetime



####################### MiniGPT4 ##########################
##### VLKEB  Setting ######
def test_MiniGPT4_FT():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/minigpt4.yaml')
    eval_ds = CaptionDataset(eval_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial(log=True, gap_num=gap_num)

def test_MiniGPT4_FT_VIS():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/minigpt4_qformer.yaml')
    eval_ds = CaptionDataset(eval_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial(log=True, gap_num=gap_num)

def test_MiniGPT4_MEND():
    hparams = MENDMultimodalHparams.from_hparams('hparams/MEND/minigpt4.yaml')
    eval_ds = CaptionDataset(eval_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial(log=True, gap_num=gap_num)

def test_MiniGPT4_SERAC():
    hparams = SERACMultimodalHparams.from_hparams('hparams/SERAC/minigpt4.yaml')
    eval_ds = CaptionDataset(eval_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial(log=True, gap_num=gap_num)

##### Compositoin: CCKE Setting ######
## Baselines

# FT
def test_MiniGPT4_FT_composition():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/minigpt4.yaml')
    eval_ds = CompositionalDataset('datasets/eval_compositional_edit_new.json', config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )

    trainer.test_sequencial_compositional_ft(log=True, test_num=200 , gap_num=gap_num)



    trainer.test_sequencial_compositional_ft(log=True, test_num=200 , gap_num=gap_num)

def test_MiniGPT4_FT_composition_0():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/minigpt4_0.yaml')
    eval_ds = CompositionalDataset('datasets/eval_compositional_edit_new.json', config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )

    trainer.test_sequencial_compositional_ft(log=True, test_num=500 , gap_num=gap_num)

def test_MiniGPT4_FT_composition_1():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/minigpt4_1.yaml')
    eval_ds = CompositionalDataset('datasets/eval_compositional_edit_new.json', config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )

    trainer.test_sequencial_compositional_ft(log=True, test_num=500 , gap_num=gap_num)

def test_MiniGPT4_FT_composition_2():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/minigpt4_2.yaml')
    eval_ds = CompositionalDataset('datasets/eval_compositional_edit_new.json', config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )

    trainer.test_sequencial_compositional_ft(log=True, test_num=500 , gap_num=gap_num)

def test_MiniGPT4_FT_composition_3():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/minigpt4_3.yaml')
    eval_ds = CompositionalDataset('datasets/eval_compositional_edit_new.json', config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )

    trainer.test_sequencial_compositional_ft(log=True, test_num=500 , gap_num=gap_num)



# LoRA(rank 16)
def test_MiniGPT4_CompositionalEdit_one_lora():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/minigpt4_compositional_edit_r16.yaml')
    eval_ds = CompositionalDataset('datasets/eval_compositional_edit_new.json', config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_compositional(log=True, test_num=500 ,gap_num=gap_num)

def test_MiniGPT4_CompositionalEdit_one_lora_4():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/minigpt4_compositional_edit_r16_4.yaml')
    eval_ds = CompositionalDataset('datasets/eval_compositional_edit_new.json', config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_compositional(log=True, test_num=500 ,gap_num=gap_num)

def test_MiniGPT4_CompositionalEdit_one_lora_5():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/minigpt4_compositional_edit_r16_5.yaml')
    eval_ds = CompositionalDataset('datasets/eval_compositional_edit_new.json', config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_compositional(log=True, test_num=500 ,gap_num=gap_num)

def test_MiniGPT4_CompositionalEdit_one_lora_6():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/minigpt4_compositional_edit_r16_6.yaml')
    eval_ds = CompositionalDataset('datasets/eval_compositional_edit_new.json', config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_compositional(log=True, test_num=500 ,gap_num=gap_num)

def test_MiniGPT4_CompositionalEdit_one_lora_7():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/minigpt4_compositional_edit_r16_7.yaml')
    eval_ds = CompositionalDataset('datasets/eval_compositional_edit_new.json', config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_compositional(log=True, test_num=500 ,gap_num=gap_num)


# OURS
# train connector train@50
def test_MiniGPT4_CompositionalEdit_Connector_attention_rag_50():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/minigpt4_compositional_edit_connector_lora_attention_rag_50.yaml')
    eval_ds = CompositionalDataset_RAG_50('datasets/train_compositional_edit.json', config=hparams, hop=hop) 
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_compositional_connector_attention_rag_50(log=True, test_num=500 ,gap_num=gap_num) 

def test_MiniGPT4_CompositionalEdit_Connector_attention_rag_70():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/minigpt4_compositional_edit_connector_lora_attention_rag_70.yaml')
    eval_ds = CompositionalDataset_RAG_50('datasets/train_compositional_edit.json', config=hparams, hop=hop) 
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_compositional_connector_attention_rag_70(log=True, test_num=500 ,gap_num=gap_num) 

def test_MiniGPT4_CompositionalEdit_Connector_attention_rag_50_eval():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/blip2_compositional_edit_connector_lora_attention_rag_50_eval.yaml')
    eval_ds = CompositionalDataset_RAG('datasets/eval_compositional_edit_new.json', config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_compositional_connector_eval(log=True, test_num=200 ,gap_num=gap_num) # 600개부터 터짐
 
####################### BLIP2 ##########################
#region: BLIP2
def test_Blip2OPT_FT():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/blip2.yaml')
    eval_ds = CaptionDataset(eval_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial(log=True, gap_num=gap_num)

def test_Blip2OPT_FT_VIS():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/blip2_qformer.yaml')
    eval_ds = CaptionDataset(eval_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial(log=True, gap_num=gap_num)

def test_Blip2OPT_MEND():
    hparams = MENDMultimodalHparams.from_hparams('hparams/MEND/blip2.yaml')
    eval_ds = CaptionDataset(eval_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial(log=True, gap_num=gap_num)

def test_Blip2OPT_SERAC():
    hparams = SERACMultimodalHparams.from_hparams('hparams/SERAC/blip2.yaml')
    eval_ds = CaptionDataset(eval_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial(log=True, gap_num=gap_num)

# Two LorA + Connector(Self-Attention) + RAG(비율 조정: 50~70% 정확도)
# def test_LLaVA_CompositionalEdit_Connector_attention_rag_70():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava_compositional_edit_connector_lora_attention_rag_70.yaml')
    eval_ds = CompositionalDataset_RAG_70('datasets/train_compositional_edit.json', config=hparams, hop=hop) # prompt feeding 변경 필요 
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_compositional_connector_attention_rag_70(log=True, test_num=500 ,gap_num=gap_num) # 600개부터 터짐

#endregion

####################### LLAVA ##########################
#### FT ####
def test_LLaVA_FT():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava.yaml')
    eval_ds = CaptionDataset(eval_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial(log=True, gap_num=gap_num)

def test_LLaVA_FT_VIS():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava_mmproj.yaml')
    eval_ds = CaptionDataset(eval_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial(log=True, gap_num=gap_num)

#### FT Composition #######
def test_LLaVA_FT_Composition():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava.yaml')
    eval_ds = CompositionalDataset('datasets/eval_compositional_edit_new.json', config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    
    trainer.test_sequencial_compositional_ft(log=True, test_num=200 , gap_num=gap_num)

def test_LLaVA_FT_VIS_Composition():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava_mmproj.yaml')
    eval_ds = CompositionalDataset('datasets/eval_compositional_edit.json', config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_compositional_ft_vis(log=True, test_num=5 , gap_num=gap_num)
    

#region: Unused Code(VLKEB)
####################### VLKEB #######################
def test_LLaVA_MEND():
    hparams = MENDMultimodalTrainingHparams.from_hparams('hparams/MEND/llava.yaml')
    eval_ds = CaptionDataset(eval_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial(log=True, gap_num=gap_num)

def test_LLaVA_SERAC():
    hparams = SERACMultimodalTrainingHparams.from_hparams('hparams/SERAC/llava.yaml')
    eval_ds = CaptionDataset(eval_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial(log=True, gap_num=gap_num)

def test_LLaVA_LORA():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava_lora_slowly.yaml')
    eval_ds = CaptionDataset(eval_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial(log=True, gap_num=gap_num)

def test_LLaVA_VisEdit():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava_vis_edit.yaml')
    eval_ds = CaptionDataset(eval_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial(log=True, gap_num=gap_num)

# def test_LLaVA_TextualEdit():
#     hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava_textual_edit.yaml')
#     eval_ds = TextualDataset('datasets/train_textual_edit.json', config=hparams, hop=hop, no_image=True)
#     trainer = MultimodalTrainer(
#         config=hparams,
#         train_set=eval_ds,
#         val_set=eval_ds
#     )
#     trainer.test_sequencial_textual(log=True, gap_num=gap_num)
#endregion

##### Compositoin: CCKE Setting ######
### --- Baselines --- ### 
# FT 
def test_LLaVA_FT_Composition_0():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava_0.yaml')
    eval_ds = CompositionalDataset('datasets/eval_compositional_edit_new.json', config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    
    trainer.test_sequencial_compositional_ft(log=True, test_num=500 , gap_num=gap_num)

def test_LLaVA_FT_Composition_1():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava_1.yaml')
    eval_ds = CompositionalDataset('datasets/eval_compositional_edit_new.json', config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    
    trainer.test_sequencial_compositional_ft(log=True, test_num=500 , gap_num=gap_num)

def test_LLaVA_FT_Composition_2():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava_2.yaml')
    eval_ds = CompositionalDataset('datasets/eval_compositional_edit_new.json', config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    
    trainer.test_sequencial_compositional_ft(log=True, test_num=500 , gap_num=gap_num)

def test_LLaVA_FT_Composition_3():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava_3.yaml')
    eval_ds = CompositionalDataset('datasets/eval_compositional_edit_new.json', config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    
    trainer.test_sequencial_compositional_ft(log=True, test_num=500 , gap_num=gap_num)


# LoRA(rank:16)
def test_LLaVA_CompositionalEdit_one_lora():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava_compositional_edit_r16.yaml')
    eval_ds = CompositionalDataset('datasets/eval_compositional_edit_new.json', config=hparams, hop=hop) # updated.json -> new.json
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_compositional(log=True, test_num=500 ,gap_num=gap_num) # 600개부터 터짐

def test_LLaVA_CompositionalEdit_one_lora_gpu4():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava_compositional_edit_r16_gpu4.yaml')
    eval_ds = CompositionalDataset('datasets/eval_compositional_edit_new.json', config=hparams, hop=hop) # updated.json -> new.json
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_compositional(log=True, test_num=500 ,gap_num=gap_num) # 600개부터 터짐
    ## Visual Edit + Textual Edit --> 각 sample을 pair로 묶어서 데이터 구성시키면 될듯. 

def test_LLaVA_CompositionalEdit_one_lora_gpu5():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava_compositional_edit_r16_gpu5.yaml')
    eval_ds = CompositionalDataset('datasets/eval_compositional_edit_new.json', config=hparams, hop=hop) # updated.json -> new.json
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_compositional(log=True, test_num=500 ,gap_num=gap_num) # 600개부터 터짐
    ## Visual Edit + Textual Edit --> 각 sample을 pair로 묶어서 데이터 구성시키면 될듯. 

def test_LLaVA_CompositionalEdit_one_lora_gpu6():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava_compositional_edit_r16_gpu6.yaml')
    eval_ds = CompositionalDataset('datasets/eval_compositional_edit_new.json', config=hparams, hop=hop) # updated.json -> new.json
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_compositional(log=True, test_num=500 ,gap_num=gap_num) # 600개부터 터짐
    ## Visual Edit + Textual Edit --> 각 sample을 pair로 묶어서 데이터 구성시키면 될듯. 

def test_LLaVA_CompositionalEdit_one_lora_gpu7():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava_compositional_edit_r16_gpu7.yaml')
    eval_ds = CompositionalDataset('datasets/eval_compositional_edit_new.json', config=hparams, hop=hop) # updated.json -> new.json
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_compositional(log=True, test_num=500 ,gap_num=gap_num) # 600개부터 터짐
    ## Visual Edit + Textual Edit --> 각 sample을 pair로 묶어서 데이터 구성시키면 될듯. 



# ohers: LoRA(dual lora)
def test_LLaVA_CompositionalEdit_two_lora():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava_compositional_edit.yaml')
    #eval_ds = CompositionalDataset('datasets/train_compositional_edit.json', config=hparams, hop=hop)
    eval_ds = CompositionalDataset('datasets/eval_compositional_edit_new.json', config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_compositional_two(log=True, test_num=200 ,gap_num=gap_num) # 500 -> 200



## Train Connectors
# def test_LLaVA_CompositionalEdit_Connector():
#     hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava_compositional_edit_connector.yaml')
#     eval_ds = CompositionalDataset('datasets/train_compositional_edit.json', config=hparams, hop=hop)
#     trainer = MultimodalTrainer(
#         config=hparams,
#         train_set=eval_ds,
#         val_set=eval_ds
#     )
#     trainer.test_sequencial_compositional_connector(log=True, test_num=500 ,gap_num=gap_num) # 600개부터 터짐
#     ## Visual Edit + Textual Edit --> 각 sample을 pair로 묶어서 데이터 구성시키면 될듯. 

# def test_LLaVA_CompositionalEdit_Connector_Two():
#     hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava_compositional_edit_connector_two.yaml')
#     eval_ds = CompositionalDataset('datasets/train_compositional_edit.json', config=hparams, hop=hop)
#     trainer = MultimodalTrainer(
#         config=hparams,
#         train_set=eval_ds,
#         val_set=eval_ds
#     )
#     trainer.test_sequencial_compositional_connector_two(log=True, test_num=200 ,gap_num=gap_num) # 600개부터 터짐
#     ## Visual Edit + Textual Edit --> 각 sample을 pair로 묶어서 데이터 구성시키면 될듯. 

## LLaVA 1.5V 7B
#region: LLaVA 1.5V 7B - Connector(Attention) 
# Two LorA + Connector(Self-Attention)
def test_LLaVA_CompositionalEdit_Connector_attention():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava_compositional_edit_connector_lora_attention.yaml')
    eval_ds = CompositionalDataset('datasets/train_compositional_edit.json', config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_compositional_connector_attention(log=True, test_num=500 ,gap_num=gap_num) # 600개부터 터짐
    ## Visual Edit + Textual Edit --> 각 sample을 pair로 묶어서 데이터 구성시키면 될듯. 
def test_LLaVA_CompositionalEdit_Connector_attention_eval():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava_compositional_edit_connector_lora_attention_eval.yaml')
    eval_ds = CompositionalDataset('datasets/eval_compositional_edit.json', config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_compositional_connector_eval(log=True, test_num=200 ,gap_num=gap_num) # 600개부터 터짐
 
# Two LorA + Connector(Self-Attention) + RAG
def test_LLaVA_CompositionalEdit_Connector_attention_rag():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava_compositional_edit_connector_lora_attention_rag.yaml')
    eval_ds = CompositionalDataset_RAG('datasets/train_compositional_edit.json', config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_compositional_connector_attention_rag(log=True, test_num=500 ,gap_num=gap_num) # 600개부터 터짐
def test_LLaVA_CompositionalEdit_Connector_attention_rag_eval():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava_compositional_edit_connector_lora_attention_rag_eval.yaml')
    eval_ds = CompositionalDataset_RAG('datasets/eval_compositional_edit.json', config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_compositional_connector_eval(log=True, test_num=200 ,gap_num=gap_num) # 600개부터 터짐
 
# Two LorA + Connector(Self-Attention) + RAG(비율 조정: 50~70% 정확도)
# -- train@70%, 50%
def test_LLaVA_CompositionalEdit_Connector_attention_rag_70():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava_compositional_edit_connector_lora_attention_rag_70.yaml')
    eval_ds = CompositionalDataset_RAG_70('datasets/train_compositional_edit.json', config=hparams, hop=hop) # prompt feeding 변경 필요 
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_compositional_connector_attention_rag_70(log=True, test_num=500 ,gap_num=gap_num) # 600개부터 터짐
def test_LLaVA_CompositionalEdit_Connector_attention_rag_50():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava_compositional_edit_connector_lora_attention_rag_50.yaml')
    eval_ds = CompositionalDataset_RAG_50('datasets/train_compositional_edit.json', config=hparams, hop=hop) # prompt feeding 변경 필요 
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_compositional_connector_attention_rag_50(log=True, test_num=500 ,gap_num=gap_num) # 600개부터 터짐

# -- test@100 --
def test_LLaVA_CompositionalEdit_Connector_attention_rag_100_eval():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava_compositional_edit_connector_lora_attention_rag_100_eval.yaml')
    eval_ds = CompositionalDataset_RAG('datasets/eval_compositional_edit_new.json', config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_compositional_connector_eval(log=True, test_num=200 ,gap_num=gap_num) # 600개부터 터짐
def test_LLaVA_CompositionalEdit_Connector_attention_rag_70_eval():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava_compositional_edit_connector_lora_attention_rag_70_eval.yaml')
    eval_ds = CompositionalDataset_RAG('datasets/eval_compositional_edit_new.json', config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_compositional_connector_eval(log=True, test_num=200 ,gap_num=gap_num) # 600개부터 터짐
def test_LLaVA_CompositionalEdit_Connector_attention_rag_50_eval():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava_compositional_edit_connector_lora_attention_rag_50_eval.yaml')
    eval_ds = CompositionalDataset_RAG('datasets/eval_compositional_edit_new.json', config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_compositional_connector_eval(log=True, test_num=200 ,gap_num=gap_num) # 600개부터 터짐

# -- test@50% --
def test_LLaVA_CompositionalEdit_Connector_attention_rag_100_eval_50():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava_compositional_edit_connector_lora_attention_rag_eval.yaml')
    eval_ds = CompositionalDataset_RAG_50('datasets/eval_compositional_edit_new.json', config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_compositional_connector_eval_50(log=True, test_num=200 ,gap_num=gap_num) # 600개부터 터짐
def test_LLaVA_CompositionalEdit_Connector_attention_rag_70_eval_50():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava_compositional_edit_connector_lora_attention_rag_70_eval_70.yaml')
    eval_ds = CompositionalDataset_RAG_50('datasets/eval_compositional_edit_new.json', config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_compositional_connector_eval_50(log=True, test_num=200 ,gap_num=gap_num) # 600개부터 터짐
def test_LLaVA_CompositionalEdit_Connector_attention_rag_50_eval_50():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava_compositional_edit_connector_lora_attention_rag_50_eval_50.yaml')
    eval_ds = CompositionalDataset_RAG_50('datasets/eval_compositional_edit_new.json', config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_compositional_connector_eval_50(log=True, test_num=200 ,gap_num=gap_num) # 600개부터 터짐
 
# Visualization(activation)
def test_LLaVA_CompositionalEdit_Connector_attention_rag_50_eval_50_vis():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava_compositional_edit_connector_lora_attention_rag_50_eval_50_vis.yaml')
    eval_ds = CompositionalDataset_RAG_50('datasets/eval_compositional_edit.json', config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_compositional_connector_eval_vis(log=True, test_num=10 ,gap_num=10) # 600개부터 터짐


#endregion



if __name__ == "__main__":
    function_name = sys.argv[1]
    hop = 1
    os.makedirs('results/results_sequencial', exist_ok=True)

    eval_json_path = 'datasets/eval_multihop.json'
    if function_name not in globals() or not callable(globals()[function_name]):
        print(f"Error: Function '{function_name}' does not exist.")
        sys.exit(1)
    for gap_num in [0 , 10, 20, 50, 100]:
        globals()[function_name]()
    
    

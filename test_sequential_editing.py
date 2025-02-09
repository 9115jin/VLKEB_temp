import os
import torch
import types
from statistics import mean

from easyeditor import BaseEditor, MultimodalTrainer, MultimodalEditor
from easyeditor import CaptionDataset, VQADataset, TextualDataset, CompositionalDataset, CompositionalDataset_RAG, CompositionalDataset_RAG_Simple
from easyeditor import MENDMultimodalTrainingHparams, SERACMultimodalTrainingHparams, IKEMultimodalHyperParams, MENDMultimodalHparams \
    , SERACMultimodalHparams, FTMultimodalHparams
from easyeditor import encode_ike_facts_multimodal
from sentence_transformers import SentenceTransformer
import sys
from datetime import datetime



####################### MiniGPT4 ##########################

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


####################### BLIP2 ##########################
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

####################### LLAVA ##########################

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

def test_LLaVA_TextualEdit():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava_textual_edit.yaml')
    eval_ds = TextualDataset('datasets/train_textual_edit.json', config=hparams, hop=hop, no_image=True)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_textual(log=True, gap_num=gap_num)

def test_LLaVA_CompositionalEdit():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava_compositional_edit.yaml')
    eval_ds = CompositionalDataset('datasets/train_compositional_edit.json', config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_compositional(log=True, test_num=500 ,gap_num=gap_num) # 600개부터 터짐
    ## Visual Edit + Textual Edit --> 각 sample을 pair로 묶어서 데이터 구성시키면 될듯. 

def test_LLaVA_CompositionalEdit_one_lora():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava_compositional_edit_r16.yaml')
    eval_ds = CompositionalDataset('datasets/train_compositional_edit.json', config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_compositional(log=True, test_num=500 ,gap_num=gap_num) # 600개부터 터짐
    ## Visual Edit + Textual Edit --> 각 sample을 pair로 묶어서 데이터 구성시키면 될듯. 

def test_LLaVA_CompositionalEdit_Connector():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava_compositional_edit_connector.yaml')
    eval_ds = CompositionalDataset('datasets/train_compositional_edit.json', config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_compositional_connector(log=True, test_num=500 ,gap_num=gap_num) # 600개부터 터짐
    ## Visual Edit + Textual Edit --> 각 sample을 pair로 묶어서 데이터 구성시키면 될듯. 

def test_LLaVA_CompositionalEdit_Connector_Two():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava_compositional_edit_connector_two.yaml')
    eval_ds = CompositionalDataset('datasets/train_compositional_edit.json', config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_compositional_connector_two(log=True, test_num=200 ,gap_num=gap_num) # 600개부터 터짐
    ## Visual Edit + Textual Edit --> 각 sample을 pair로 묶어서 데이터 구성시키면 될듯. 

def test_LLaVA_CompositionalEdit_Connector_Two_slow():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava_compositional_edit_connector_two_slow.yaml')
    eval_ds = CompositionalDataset('datasets/train_compositional_edit.json', config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_compositional_connector_two(log=True, test_num=200 ,gap_num=gap_num) # 600개부터 터짐
    ## Visual Edit + Textual Edit --> 각 sample을 pair로 묶어서 데이터 구성시키면 될듯. 

# 최종 버젼(Two LoRA, PEFT 사용)
def test_LLaVA_CompositionalEdit_two_lora():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava_compositional_edit.yaml')
    eval_ds = CompositionalDataset('datasets/train_compositional_edit.json', config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_compositional_two(log=True, test_num=500 ,gap_num=gap_num) 

def test_LLaVA_VisEdit_two_lora():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava_vis_edit.yaml')
    eval_ds = CaptionDataset('datasets/train_compositional_edit.json', config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial(log=True, gap_num=gap_num, test_num=500)

def test_LLaVA_TextualEdit_two_lora():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava_textual_edit.yaml')
    eval_ds = TextualDataset('datasets/train_textual_edit.json', config=hparams, hop=hop, no_image=True)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_textual(log=True, gap_num=gap_num, test_num=500)

# LLAVA(BASE) + RAG(retrieval 성능 분석용)
def test_LLaVA_RAG():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava_rag.yaml')
    eval_ds = CompositionalDataset_RAG_Simple('datasets/train_compositional_edit.json', config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_base_with_rag(log=True, gap_num=0, test_num=500)

# RAG + Two LoRA
def test_LLaVA_RAG_With_Two_Lora():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava_compositional_edit_rag_with_twolora.yaml')
    eval_ds = CompositionalDataset_RAG('datasets/train_compositional_edit.json', config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )

    trainer.test_sequencial_rag_with_two_lora(log=True, gap_num=gap_num, test_num=500)

# RAG + Two LoRA + dare_linear
def test_LLaVA_RAG_With_Two_Lora_Connector():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava_compositional_edit_rag_with_twolora_connector.yaml')
    eval_ds = CompositionalDataset_RAG('datasets/train_compositional_edit.json', config=hparams, hop=hop)
    #eval_ds = CompositionalDataset_RAG_Simple('datasets/train_compositional_edit.json', config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )

    trainer.test_sequencial_rag_with_two_lora_connector(log=True, gap_num=gap_num, test_num=500)

# RAG + Two LoRA + dare_linear
def test_LLaVA_RAG_With_Two_Lora_wo_prompt():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava_compositional_edit_rag_with_twolora_wo_prompt.yaml')
    eval_ds = CompositionalDataset('datasets/train_compositional_edit.json', config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )

    trainer.test_sequencial_rag_with_two_lora_connector(log=True, gap_num=gap_num, test_num=500)

## Connector(FFN)
# Two LorA + Connector(FFN)
def test_LLaVA_CompositionalEdit_Connector_ffn():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava_compositional_edit_connector_lora_ffn.yaml')
    eval_ds = CompositionalDataset('datasets/train_compositional_edit.json', config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_compositional_connector_ffn(log=True, test_num=500 ,gap_num=gap_num) # 600개부터 터짐
    ## Visual Edit + Textual Edit --> 각 sample을 pair로 묶어서 데이터 구성시키면 될듯. 
def test_LLaVA_CompositionalEdit_Connector_ffn_eval():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava_compositional_edit_connector_lora_ffn_eval.yaml')
    eval_ds = CompositionalDataset('datasets/eval_compositional_edit.json', config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_compositional_connector_ffn_eval(log=True, test_num=200 ,gap_num=gap_num) # 600개부터 터짐
    ## Visual Edit + Textual Edit --> 각 sample을 pair로 묶어서 데이터 구성시키면 될듯. 

# Two LorA + Connector(FFN)
def test_LLaVA_CompositionalEdit_Connector_ffn_rag():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava_compositional_edit_connector_lora_ffn_rag.yaml')
    eval_ds = CompositionalDataset_RAG('datasets/train_compositional_edit.json', config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_compositional_connector_ffn_rag(log=True, test_num=500 ,gap_num=gap_num) # 600개부터 터짐
    ## Visual Edit + Textual Edit --> 각 sample을 pair로 묶어서 데이터 구성시키면 될듯. 
def test_LLaVA_CompositionalEdit_Connector_ffn_rag_eval():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava_compositional_edit_connector_lora_ffn_rag_eval.yaml')
    eval_ds = CompositionalDataset_RAG('datasets/eval_compositional_edit.json', config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_compositional_connector_eval(log=True, test_num=2 ,gap_num=gap_num) # 600개부터 터짐
    ## Visual Edit + Textual Edit --> 각 sample을 pair로 묶어서 데이터 구성시키면 될듯. 


## Connector(Attention) 
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
    trainer.test_sequencial_compositional_connector_attention_eval(log=True, test_num=200 ,gap_num=gap_num) # 600개부터 터짐
 
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
    eval_ds = CompositionalDataset('datasets/eval_compositional_edit.json', config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_compositional_connector_attention_eval(log=True, test_num=200 ,gap_num=gap_num) # 600개부터 터짐
 


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
    
    

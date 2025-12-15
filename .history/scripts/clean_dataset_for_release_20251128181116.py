#!/usr/bin/env python3
"""
CCKEB 데이터셋 정제 스크립트 (공개용)

제거할 필드:
- port_new[].triple1, triple2, tex_Q&A (메타데이터, 코드 미사용)
- textual_edit.image, textual_edit.image_rephrase (항상 None)
- textual_edit.port_new[].triple1, triple2, triple2_edit (메타데이터)
- 2-hop, 3-hop, 4-hop (실험에서 hop=1 고정)

유지할 필드:
- src, rephrase, pred, alt, image, image_rephrase
- loc, loc_ans, m_loc, m_loc_q, m_loc_a
- src_q, rephrase_q, m_loc_q_q (OURS용 Query Decomposition, eval에만 있음)
- port_new[].port_type, Q&A (1-hop만)
- textual_edit (image/image_rephrase 제외)
"""

import json
import os
import argparse
from copy import deepcopy

def clean_port_new(port_new_list, keep_only_1hop=True):
    """port_new 리스트에서 불필요한 필드 제거
    
    제거: triple1, triple2, tex_Q&A
    유지: port_type, Q&A
    """
    if not port_new_list:
        return port_new_list
    
    cleaned = []
    for port in port_new_list:
        # 1-hop만 유지 (실험에서 hop=1로 고정됨)
        if keep_only_1hop and port.get('port_type') != '1-hop':
            continue
        
        cleaned_port = {
            'port_type': port.get('port_type'),
            'Q&A': port.get('Q&A')  # Question, Answer, Query(OURS용)
        }
        cleaned.append(cleaned_port)
    return cleaned

def clean_textual_edit(textual_edit, keep_only_1hop=True):
    """textual_edit에서 불필요한 필드 제거
    
    제거: image, image_rephrase (항상 None)
    """
    if not textual_edit:
        return textual_edit
    
    cleaned = {
        'src': textual_edit.get('src'),
        'rephrase': textual_edit.get('rephrase'),
        'pred': textual_edit.get('pred'),
        'alt': textual_edit.get('alt'),
        'loc': textual_edit.get('loc'),
        'loc_ans': textual_edit.get('loc_ans'),
    }
    
    # port_new 정제
    if 'port_new' in textual_edit and textual_edit['port_new']:
        cleaned['port_new'] = clean_port_new(textual_edit['port_new'], keep_only_1hop)
    
    return cleaned

def clean_sample(sample, keep_only_1hop=True):
    """단일 샘플 정제"""
    cleaned = {
        # 기본 필드 (Visual Edit)
        'src': sample.get('src'),
        'rephrase': sample.get('rephrase'),
        'pred': sample.get('pred'),
        'alt': sample.get('alt'),
        'image': sample.get('image'),
        'image_rephrase': sample.get('image_rephrase'),
        
        # Locality
        'loc': sample.get('loc'),
        'loc_ans': sample.get('loc_ans'),
        
        # Multimodal Locality
        'm_loc': sample.get('m_loc'),
        'm_loc_q': sample.get('m_loc_q'),
        'm_loc_a': sample.get('m_loc_a'),
    }
    
    # OURS용 Query Decomposition (있으면 유지 - eval에만 있음)
    if 'src_q' in sample:
        cleaned['src_q'] = sample.get('src_q')
    if 'rephrase_q' in sample:
        cleaned['rephrase_q'] = sample.get('rephrase_q')
    if 'm_loc_q_q' in sample:
        cleaned['m_loc_q_q'] = sample.get('m_loc_q_q')
    
    # Portability 정제
    if 'port_new' in sample and sample['port_new']:
        cleaned['port_new'] = clean_port_new(sample['port_new'], keep_only_1hop)
    
    # Textual Edit 정제
    if 'textual_edit' in sample and sample['textual_edit']:
        cleaned['textual_edit'] = clean_textual_edit(sample['textual_edit'], keep_only_1hop)
    
    return cleaned

def process_dataset(input_path, output_path, keep_only_1hop=True):
    """데이터셋 정제 및 저장"""
    # 데이터 로드
    print(f"Loading: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Original samples: {len(data)}")
    
    # 원본 크기 계산
    original_size = os.path.getsize(input_path) / (1024 * 1024)
    print(f"Original size: {original_size:.2f} MB")
    
    # 정제
    cleaned_data = [clean_sample(sample, keep_only_1hop) for sample in data]
    
    # 저장
    print(f"\nSaving: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
    
    # 결과 크기 계산
    output_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Cleaned size: {output_size:.2f} MB")
    print(f"Size reduction: {(1 - output_size/original_size) * 100:.1f}%")
    
    # 검증
    print(f"\n=== Verification ===")
    print(f"Cleaned samples: {len(cleaned_data)}")
    print(f"Top-level keys (sample 0): {list(cleaned_data[0].keys())}")
    
    if 'port_new' in cleaned_data[0] and cleaned_data[0]['port_new']:
        print(f"port_new keys: {list(cleaned_data[0]['port_new'][0].keys())}")
    
    if 'textual_edit' in cleaned_data[0]:
        print(f"textual_edit keys: {list(cleaned_data[0]['textual_edit'].keys())}")
        if 'port_new' in cleaned_data[0]['textual_edit'] and cleaned_data[0]['textual_edit']['port_new']:
            print(f"textual_edit.port_new keys: {list(cleaned_data[0]['textual_edit']['port_new'][0].keys())}")
    
    print("\n✅ Dataset cleaned successfully!")
    return cleaned_data

def main():
    parser = argparse.ArgumentParser(description='CCKEB 데이터셋 정제')
    parser.add_argument('--mode', choices=['eval', 'train', 'all'], default='all',
                        help='정제할 데이터셋 (eval/train/all)')
    args = parser.parse_args()
    
    datasets = []
    
    if args.mode in ['eval', 'all']:
        datasets.append({
            'input': '/home/9115jin/temp/datasets/eval_comp_0411.json',
            'output': '/home/9115jin/temp/datasets/CCKEB_eval.json',
            'name': 'CCKEB_eval'
        })
    
    if args.mode in ['train', 'all']:
        datasets.append({
            'input': '/home/9115jin/project/multimodaledit/VLKEB/datasets/train_compositional_edit.json',
            'output': '/home/9115jin/temp/datasets/CCKEB_train.json',
            'name': 'CCKEB_train'
        })
    
    for ds in datasets:
        print(f"\n{'='*50}")
        print(f"Processing: {ds['name']}")
        print(f"{'='*50}")
        process_dataset(ds['input'], ds['output'])

if __name__ == '__main__':
    main()

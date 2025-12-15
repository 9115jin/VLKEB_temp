import json
import argparse
import os

def update_dataset(input_path, output_path=None):
    print(f"Loading: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    updated_count = 0
    for item in data:
        # Update port_type and Answer
        if 'port_new' in item:
            for port in item['port_new']:
                if port['port_type'] == '1-hop':
                    port['port_type'] = 'comp'
                    
                    # Update Answer with textual_edit.alt
                    if 'textual_edit' in item and 'alt' in item['textual_edit']:
                        alt = item['textual_edit']['alt']
                        # alt가 리스트인 경우 처리 (문자열로 변환)
                        if isinstance(alt, list):
                            alt = " ".join(alt)
                        port['Q&A']['Answer'] = alt
                        updated_count += 1
        
        # Update textual_edit.port_new as well if exists
        if 'textual_edit' in item and 'port_new' in item['textual_edit']:
             for port in item['textual_edit']['port_new']:
                if port['port_type'] == '1-hop':
                    port['port_type'] = 'comp'
                    
                    # Update Answer with textual_edit.alt
                    if 'alt' in item['textual_edit']:
                        alt = item['textual_edit']['alt']
                        if isinstance(alt, list):
                            alt = " ".join(alt)
                        port['Q&A']['Answer'] = alt

    if output_path is None:
        output_path = input_path
        
    print(f"Saving to: {output_path}")
    print(f"Updated {updated_count} items.")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default='/home/9115jin/temp/datasets/CCKEB_train.json')
    parser.add_argument('--eval_path', default='/home/9115jin/temp/datasets/CCKEB_eval.json')
    args = parser.parse_args()
    
    if os.path.exists(args.train_path):
        update_dataset(args.train_path)
    else:
        print(f"File not found: {args.train_path}")
        
    if os.path.exists(args.eval_path):
        update_dataset(args.eval_path)
    else:
        print(f"File not found: {args.eval_path}")

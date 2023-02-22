#!/usr/bin/env python
# author = 'ZZH'
# time = 2023/2/9
# project = huggingface_python
import json
from datasets import load_dataset
from transformers import RobertaTokenizer
import argparse

# print(get_dataset_config_names("code_x_glue_ct_code_to_text"))
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--lang', default='python', help='dataset_langs')
parser.add_argument('--split', default='test', help='train or validation or test')

args = parser.parse_args()

# Load the Code2Glue_python python dataset
dataset = load_dataset("code_x_glue_ct_code_to_text", args.lang, split=args.split)
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
with open('test.jsonl', 'w') as f:
    for row in dataset:
        code = ' '.join(row['code_tokens']).replace('\n', ' ')
        nl = ' '.join(row['docstring_tokens']).replace('\n', '')
        code = ' '.join(code.strip().split())
        nl = ' '.join(nl.strip().split())

        # if len(tokenizer(code)['input_ids']) + len(tokenizer(nl)['input_ids'])> 128: continue

        f.write(json.dumps({"src": code, 'trg': nl}) + '\n')


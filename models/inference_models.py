import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


# get model and tokenizer
def get_inference_model(model_dir):
    inference_tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    inference_model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).half().cuda()
    inference_model.eval()
    return inference_tokenizer, inference_model


# get model response
def get_local_response(query, model, tokenizer, max_length=2048, truncation=True, do_sample=False, max_new_tokens=512, temperature=0.7):
    cnt = 5
    all_response = ''
    while cnt:
        try:
            inputs = tokenizer([query], padding=True, return_tensors="pt", truncation=truncation,
                               max_length=max_length).to('cuda')
            output_ = model.generate(**inputs, do_sample=do_sample, max_new_tokens=max_new_tokens, temperature=temperature)
            output = output_.tolist()[0][len(inputs["input_ids"][0]):]
            response = tokenizer.decode(output)

            print(f'获得回复:{response}\n')
            all_response = response
            break
        except Exception as e:
            print('重新获取回复...\n')
            cnt -= 1
    if not cnt:
        return []
    split_response = all_response.strip().split('\n')
    return split_response

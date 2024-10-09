import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


# define your value model class if needed
class ValueModel(nn.Module):
    def __init__(self, base_model):
        super(ValueModel, self).__init__()
        self.base = base_model
        pass

    def forward(self, input_ids, attention_mask):
        pass


# get value model
def get_value_model(base_model_dir, state_dict_file):
    value_tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
    value_base_model = AutoModel.from_pretrained(base_model_dir, trust_remote_code=True).bfloat16().cuda()
    if state_dict_file is None:
        return value_tokenizer, value_base_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device is set to: ", device, '\n')
    VM = ValueModel(value_base_model)
    VM.load_state_dict(torch.load(state_dict_file))
    VM.to(device)
    VM.eval()
    return value_tokenizer, VM


# local value model: str->digit in [low, high]
def get_local_value(prompt_answer, model, tokenizer, max_length=2048, low=0, high=1):
    encoded_pair = tokenizer.encode_plus(
        prompt_answer,
        padding='max_length',
        max_length=max_length,  # Set the max length
        truncation=True,
        return_tensors='pt',  # Return PyTorch Tensor format
    )
    input_ids = encoded_pair['input_ids'].to('cuda')
    # print(input_ids)
    attention_mask = encoded_pair['attention_mask'].to('cuda')
    value = model(input_ids, attention_mask).item()
    value = min(high, max(value, low))
    return value

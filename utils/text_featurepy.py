# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 14:21:28 2023

@author: Janus_yu
"""
from transformers import BertModel, BertTokenizer
import torch 
import numpy as np


data_list = []
model_name = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained('C:/Users/Janus_yu/.cache/huggingface/hub/bert-base-chinese')
model = BertModel.from_pretrained('C:/Users/Janus_yu/.cache/huggingface/hub/bert-base-chinese')
model.eval()
filename = ""
with open(filename, "r", encoding="utf-8") as file:
    lines = file.readlines()
    for line in lines:
        line = line.strip()  
        text=line
        encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            output = model(**encoded_input)
            embeddings = output.last_hidden_state
            embeddings2 = output.pooler_output
            embeddings2=embeddings2.reshape(-1).data.numpy()
            data_list.append(embeddings2)
            data_npy = np.array(data_list)
            print(data_npy.shape)
#         #print(line)  
# np.save('list_text.npy',data_list)#存储为.npy文件


# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 14:07:54 2023

@author: Janus_yu
"""

import os.path
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image
IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD =[0.229, 0.224, 0.225]
features_dir = './picturefeatures'  
data_list = []# 定义一个空的存放数据的列表
path=
for filename in os.listdir(path):#

    transform1 = transforms.Compose([
    transforms.Resize(256),  # 缩放
    transforms.CenterCrop(224),   
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ]  # 转换成Tensor
)
    print(filename)
    img = Image.open(path + '/' + filename).convert('RGB') 
    print(img)
    img1 = transform1(img)  


    resnet50_feature_extractor = models.resnet50(pretrained=True)
 
    print(resnet50_feature_extractor.fc)
    resnet50_feature_extractor.fc = nn.Linear(2048, 2048)  
    torch.nn.init.eye(resnet50_feature_extractor.fc.weight)  

    for param in resnet50_feature_extractor.parameters():
        param.requires_grad = False

    x = Variable(torch.unsqueeze(img1, dim=0).float(), requires_grad=False)
    y = resnet50_feature_extractor(x)
    y = y.data.numpy()
    data_list.append(y)
    data_npy = np.array(data_list)
    print(data_npy.shape)
np.save)#存储为.npy文件


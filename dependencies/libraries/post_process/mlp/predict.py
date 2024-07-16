import argparse
import cv2
import numpy as np
import torch
import json
from torchvision import transforms
import os
from mlp.model import MLP,THCModel
from devlib._base_ import DatasetBase_ 
from mlp.tools import Stage2

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('data_dir', action='store', help='the dir to save original dataset')
    parser.add_argument('result_dir', action='store', help='the dir to save Processed dataset')
    parser.add_argument('save_dir', action='store', help='the dir to save Processed dataset')
    parser.add_argument('checkpoint', action='store', help='the size of a patch')
    parser.add_argument('patch_size', type=int, action='store', nargs='+', help='the size of a patch')
    parser.add_argument('--batchsize', action='store', default=1, help='the description of this dataset')
    args = parser.parse_args()
    
    return args



def main():
    # args = parse_args()
    
    # 输入大小
    patch_size = [56,56] 
    # 模型文件
    checkpoint = 'logs/MA_Detection/res_cls/56patch_test/epoch_0.pth'
    # 二阶段后处理json文件保存路径
    save_dir = 'fortest/post'
    # 指向一阶段模型验证后输出的json文件目录路径
    json_dir = "fortest/result"
    # 原MA数据集文件
    data_dir = '/home/hyh/Documents/quanyi/project/Data/e_optha_MA/MA_ex'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    model = THCModel()
    print(model)
    model.load_state_dict(torch.load(checkpoint))
    
    
    dataset_cfg = DatasetBase_("VOC").voc_dict
    image_dir = os.path.join(data_dir, dataset_cfg['Image_Dir'])
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    
    pp = Stage2(image_dir, json_dir, save_dir, model, patch_size, transform)
    pp.forward()


if __name__ == "__main__":
    main()
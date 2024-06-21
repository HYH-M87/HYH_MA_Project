import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import os
from torch.utils.tensorboard import SummaryWriter 

from mlp.dataset import MA_patch
from mlp.model import MLP

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('data_dir', action='store', help='the dir to save original dataset')
    parser.add_argument('save_dir', action='store', help='the dir to save Processed dataset')
    parser.add_argument('checkpoint', action='store', help='the size of a patch')
    parser.add_argument('patch_size', type=int, action='store', nargs='+', help='the size of a patch')
    parser.add_argument('--batchsize', action='store', default=1, help='the description of this dataset')
    args = parser.parse_args()
    
    return args

def main():
    args = parse_args()
    
    # 超参数
    input_size = 3 * args.patch_size[0] * args.patch_size[1]  # 输入层大小
    output_size = 2  
    batch_size = args.batchsize

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    res_log_path = os.path.join(args.save_dir, 'res.txt')

    # 数据预处理和加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = MA_patch(args.data_dir, False, transform)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    model = MLP(input_size, output_size)

    model.load_state_dict(args.checkpoint)
    
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        Positive=[]
        for item_info in dataloader:
            images,labels,files = item_info['image'], item_info['label'], item_info['file']
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            Positive.extend(files[predicted==0])
            
        with open(res_log_path,"w") as f:
            for p in Positive:
                f.write(p+'\n')


import cv2
import numpy as np
import torch
import json
from torchvision import transforms
import os
from tqdm import tqdm
from devlib._base_ import NumpyDecoder, NumpyEncoder
from devlib._base_ import DataProcessBase_

class Stage2():
    def __init__(self, image_dir, json_dir, save_dir, model, patch_size, 
                transforms=transforms.Compose([
                        transforms.ToTensor()
                    ])
                ) -> None:
        '''

            image_dir : 保存MA图片文件的路径，不是Patch
            result_dir : 保存JSON文件的目录路径
        '''
        self.image_dir = image_dir
        self.json_dir = json_dir
        self.save_dir = save_dir
        self.patch_size = patch_size
        self.transforms = transforms
        self.device = torch.device('cuda')
        self.model = model.to(self.device)
    
    def crop_patch(self, image, boxes, size=None, if_pad=True):
        
        patches = []
        for b in boxes:
            x, y, x2, y2 = map(int,b)

            croped_image = image[y:y2, x:x2, :]
            if if_pad:
                croped_image = DataProcessBase_().resize_and_pad(croped_image, size)
            else:
                croped_image = cv2.resize(croped_image, size)

            patches.append(croped_image)
        return patches
    
    def load_result(self,json_dir):
        '''
        读入数据
        '''
        result_lists=[]
        json_files = [os.path.join(json_dir, i) for i in os.listdir(json_dir)]
        for i in json_files:
            with open(i,'r') as f:
                data = json.load(f, cls=NumpyDecoder)
            result_lists.append(data)
        
        return result_lists


    def forward(self):
        # 读入json文件数据
        result_lists = self.load_result(self.json_dir)
        # result_lists = result_lists[3698:3700]
        self.model.eval() 
        with torch.no_grad():
            for i in tqdm(result_lists, colour='green'):
                image = cv2.imread(os.path.join(self.image_dir, i['source']))
                # 如果该patch存在预测框
                if len(i['annotation']['pred']) != 0:
                    # 预测框坐标修正
                    boxes = i['annotation']['pred'] + i['annotation']['offset']
                    # 构建模型输入
                    patches = self.construct_input(self.crop_patch(image, boxes, self.patch_size)).to(self.device)
                    outputs = self.model(patches)
                    _, predicted = torch.max(outputs.data, 1)
                    # 筛选出预测为MA的检测框
                    mask = (predicted.to(torch.device('cpu')) == 0).numpy()
                    i['annotation']['pred'] =  i['annotation']['pred'][mask]
        # 保存筛选后的数据
        self.save(result_lists)

    def save(self,result_lists):
        for i in tqdm(result_lists,colour='green'):
            with open(os.path.join(self.save_dir, i['name']+'.json'), 'w') as f:
                json.dump(i, f, cls=NumpyEncoder)
        

    def construct_input(self, images):
        '''
        step1: 对数据进行必要的transforms
        step2: 构建模型输入张量 -> (b,c,h,w)
        '''
        results=[]
        if self.transforms:
            for img in images:
                results.append(self.transforms(img))
                
        return torch.stack(results)

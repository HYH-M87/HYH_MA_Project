import argparse
import cv2
import numpy as np
import torch
import json
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from mlp.dataset import MA_patch
from mlp.model import MLP
from devlib._base_ import NumpyDecoder, NumpyEncoder
from devlib._base_ import DatasetBase_ 
from devlib._base_ import DataProcessBase_

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


class Stage2():
    def __init__(self, data_dir, result_dir, save_dir, model, patch_size, 
                transforms=transforms.Compose([
                        transforms.ToTensor()
                    ])
                ) -> None:
        self.data_dir = data_dir
        self.result_dir = result_dir
        self.save_dir = save_dir
        self.patch_size = patch_size
        self.transforms = transforms
        self.device = torch.device('cuda')
        self.model = model.to(self.device)
        
        
    
        
    # def crop_patch(self, image, boxes, size, if_contain=True):
    #     patches = []
    #     for b in boxes:
    #         x, y, x2, y2 = map(int,b)
    #         cx, cy = (x+x2)//2, (y+y2)//2
    #         h, w = size
            
    #         if if_contain:
    #             assert w > (x2-x) and h > (y2-y), "target size is bigger than patch size"
                
    #         croped_image = image[cy-(h//2):cy+(h//2), cx-(w//2):cx+(w//2), :]
                
    #         patches.append(croped_image)
    #     return patches
    
    def crop_patch(self, image, boxes, size, if_contain=True):
        
        patches = []
        for b in boxes:
            x, y, x2, y2 = map(int,b)

            croped_image = image[y:y2, x:x2, :]
            croped_image = DataProcessBase_().resize_and_pad(croped_image,[56,56])

            patches.append(croped_image)
        return patches
    
    def load_result(self,result_dir):
        '''
        读入数据
        '''
        result_lists=[]
        json_files = [os.path.join(result_dir, i) for i in os.listdir(result_dir)]
        for i in json_files:
            with open(i,'r') as f:
                data = json.load(f, cls=NumpyDecoder)
            result_lists.append(data)
        
        return result_lists


    def forward(self):
        result_lists = self.load_result(self.result_dir)
        # result_lists = result_lists[3698:3700]
        self.model.eval() 
        with torch.no_grad():
            for i in tqdm(result_lists, colour='green'):
                image = cv2.imread(os.path.join(self.data_dir, i['source']))
                if len(i['annotation']['pred']) != 0:
                    boxes = i['annotation']['pred'] + i['annotation']['offset']
                    patches = self.construct_input(self.crop_patch(image, boxes, self.patch_size)).to(self.device)
                    outputs = self.model(patches)
                    _, predicted = torch.max(outputs.data, 1)
                    mask = (predicted.to(torch.device('cpu')) == 0).numpy()
                    i['annotation']['pred'] =  i['annotation']['pred'][mask]
        self.save(result_lists)

    def save(self,result_lists):
        for i in tqdm(result_lists,colour='green'):
            with open(os.path.join(self.save_dir, i['name']+'.json'), 'w') as f:
                json.dump(i, f, cls=NumpyEncoder)
        

    def construct_input(self, images):
        results=[]
        if self.transforms:
            for img in images:
                results.append(self.transforms(img))
                
        return torch.stack(results)

def main():
    # args = parse_args()
    
    patch_size = [56,56] # 输入层大小
    inchanel = 3
    output_size = 2  
    model = MLP(patch_size, output_size)
    checkpoint = 'logs/MA_Detection/mlp_cls/exp1/epoch_261.pth'
    model.load_state_dict(torch.load(checkpoint))
    
    save_dir = 'dependencies/libraries/post_process/mlp/output/res50_dcn_cabm'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    result_dir = "logs/MA_Detection/hyh_ma_det_exp006/res50_cbam_dcn/model_eval_0.2_0.4_asd/result"
    
    data_dir = '/home/hyh/Documents/quanyi/project/Data/e_optha_MA/MA_ex'
    dataset_cfg = DatasetBase_("VOC").voc_dict
    data_dir = os.path.join(data_dir, dataset_cfg['Image_Dir'])
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    
    pp = Stage2(data_dir, result_dir, save_dir, model, patch_size, transform)
    pp.forward()


if __name__ == "__main__":
    main()
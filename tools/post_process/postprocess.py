# 读入预测框
# 读入fundus mask
# 读入vessel mask
# 初步筛选出 预测框
# mlp预测
# 得出最终预测框
import json
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import torch
from torchvision import transforms
import os
from mmdet.apis import init_detector, inference_detector
from mlp.model import THCModel
from mlp.tools import Stage2
from devlib._base_ import DatasetBase_ 
from devlib._base_ import DatasetBase_ 



class PostProcess(Stage2):
    def __init__(self, image_dir, json_dir, save_dir, model, patch_size, fundus_mask_dir, vessel_mask_dir, 
                transforms=transforms.Compose([
                        transforms.ToTensor()
                    ]) ) -> None:
        super().__init__(image_dir, json_dir, save_dir, model, patch_size, 
                transforms=transforms)
        self.fundus_mask_dir = fundus_mask_dir
        self.vessel_mask_dir = vessel_mask_dir
    
    def forward(self):
        ins_list = self.load_result(self.json_dir)
        for ins in ins_list:
            if len(ins['annotation']['pred']) == 0:
                continue
            source = ins['source']
            pred_boxes = ins['annotation']['pred'] + ins['annotation']['offset']
            fmask = self.read_fundus_mask(source)
            vmask = self.read_vessel_mask(source)
            filtered_pred_boxes = self.screen_out(pred_boxes, fmask, vmask)
            if len(filtered_pred_boxes) == 0:
                ins['annotation']['pred'] = np.empty((0,4))
                continue
            self.model.eval() 
            with torch.no_grad():

                image = cv2.imread(os.path.join(self.image_dir, ins['source']))
                # 预测框坐标修正
                # 构建模型输入
                patches = self.construct_input(self.crop_patch(image, filtered_pred_boxes, self.patch_size)).to(self.device)
                outputs = self.model(patches)
                _, predicted = torch.max(outputs.data, 1)
                # 筛选出预测为MA的检测框
                mask = (predicted.to(torch.device('cpu')) == 0).numpy()
                filtered_pred_boxes = (filtered_pred_boxes[mask]).reshape((-1,4)) 
                ins['annotation']['pred'] =  filtered_pred_boxes - ins['annotation']['offset']
        # 保存筛选后的数据
        self.save(ins_list)
        


    def read_fundus_mask(self, file):
        return cv2.imread(os.path.join(self.fundus_mask_dir,file),cv2.IMREAD_GRAYSCALE)
    
    def read_vessel_mask(self, file):
        return cv2.imread(os.path.join(self.vessel_mask_dir,file),cv2.IMREAD_GRAYSCALE)
    
    def screen_out(self, boxes, fundus_mask, vessel_mask):
        
        # cv2.imshow("fm",cv2.resize(fundus_mask,None,fx=0.4,fy=0.4))
        # cv2.imshow("vm",cv2.resize(vessel_mask,None,fx=0.4,fy=0.4))
        # cv2.waitKey(0)
        
        # 保留眼底图像掩膜内的预测框
        boxes = self.filter_boxes(boxes, fundus_mask, 0)
        # 筛除血管掩膜内的预测框
        boxes = self.filter_boxes(boxes, vessel_mask, 255)
        
        return boxes

    def filter_boxes(self, boxes, mask, threshold):
        filtered_boxes = []

        for box in boxes:
            x, y, x2, y2 = box

            mask_roi = mask[y:y2, x:x2]

            if np.any(mask_roi == threshold):
                continue
            filtered_boxes.append(box)
            

        return np.array(filtered_boxes)
    
        

def parse_args():
    parser = argparse.ArgumentParser(description=' ')
    parser.add_argument('checkpoint', action='store', help='the mode to ger pedcidt result set to "model" or "load" ')
    parser.add_argument('json_dir', action='store',  help='if mode is "model" it will receive two prarams, checkpoint and model cfg, respectively.')
    parser.add_argument('data_dir', action='store', help='the name of original dataset')
    parser.add_argument('fundus_mask_dir', action='store', help='the name of original dataset')
    parser.add_argument('vessel_mask_dir', action='store', help='the name of original dataset')
    parser.add_argument('patch_size', type=int, action='store', nargs=2, help='the size of a patch')
    parser.add_argument('save', action='store', help='the directory to save Processed dataset, insted of the name of dataset')
    parser.add_argument('--descripe', action='store', default="", help='the description of this dataset')

    args = parser.parse_args()
    
    return args

def main():
    args = parse_args()
    for arg, value in vars(args).items():
        print(f"{arg}: {value} \n")
        
    fundus_mask_dir=args.fundus_mask_dir
    vessel_mask_dir=args.vessel_mask_dir
    # 输入大小
    patch_size = args.patch_size
    # 模型文件
    checkpoint = args.checkpoint
    # 二阶段后处理json文件保存路径
    save_dir = os.path.join(args.save,args.descripe)
    # 指向一阶段模型验证后输出的json文件目录路径
    json_dir = args.json_dir
    # 原MA数据集文件
    data_dir = args.data_dir
    
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
    
    
    pp = PostProcess(image_dir, json_dir, save_dir, model, patch_size, fundus_mask_dir, vessel_mask_dir, transform)
    pp.forward()
    
if __name__ == "__main__":
    main()
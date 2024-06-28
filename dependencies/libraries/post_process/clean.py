# 读入预测框
# 读入fundus mask
# 读入vessel mask
# 初步筛选出 预测框
# mlp预测
# 得出最终预测框
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import os
from torch.utils.tensorboard import SummaryWriter 
from mmdet.apis import init_detector, inference_detector
from devlib._base_ import DatasetConstructor


class PostProcess():
    def __init__(self, eval_data) -> None:
        self.post_dataset = DatasetConstructor("VOC")
        self.post_dataset.makedirs(os.path.join(eval_data,'post_process'))
        self.fundus_mask_dir='/home/hyh/Documents/quanyi/project/Data/e_optha_MA/MA_mask'
        self.annotation_dir=os.path.join(eval_data,'pred sample')
        self.save_dir = os.path.join(eval_data,'post_process')
        self.vessel_mask_dir='dependencies/libraries/vessel_seg/output'
        self.img_dir = "/home/hyh/Documents/quanyi/project/Data/e_optha_MA/MA_ex/VOC2012/JPEGImages"
        self.dataset = '/home/hyh/Documents/quanyi/project/Data/e_optha_MA/MA_ex/VOC2012/ImageSets/Main/test.txt'
        self.gt_annotation_dir = '/home/hyh/Documents/quanyi/project/Data/e_optha_MA/MA_ex/VOC2012/Annotations_Txt'
        self.data = self.read_data()
        
        
    def read_data(self):
        with open(self.dataset,'r') as f:
            data = f.read().splitlines()
        return data
    
    def read_pred_box(self, file):
        with open(os.path.join(self.annotation_dir,file+'.txt')) as f:
            data = np.array(f.read().split(),dtype=np.uint32).reshape((-1,5))
        return data

    def read_fundus_mask(self, file):
        return cv2.imread(os.path.join(self.fundus_mask_dir,file+'.jpg'),cv2.IMREAD_GRAYSCALE)
    
    def read_vessel_mask(self, file):
        return cv2.imread(os.path.join(self.vessel_mask_dir,file+'.jpg'),cv2.IMREAD_GRAYSCALE)
    
    def screen_out(self, boxes, fundus_mask, vessel_mask):
        
        # cv2.imshow("fm",cv2.resize(fundus_mask,None,fx=0.4,fy=0.4))
        # cv2.imshow("vm",cv2.resize(vessel_mask,None,fx=0.4,fy=0.4))
        # cv2.waitKey(0)
        boxes = self.filter_boxes(boxes, fundus_mask, 0)
        boxes = self.filter_boxes(boxes, vessel_mask, 255)
        
        return boxes
    
    def visualize(self, file, f_box):
        if not os.path.exists(os.path.join(self.save_dir,"vis")):
            os.makedirs(os.path.join(self.save_dir,"vis"))
        oriimg = cv2.imread(os.path.join(self.img_dir,file+'.jpg'))
        oriimg = cv2.cvtColor(oriimg,cv2.COLOR_BGR2RGB)
        boxes = self.read_pred_box(file)

        for b in boxes:
            x,y,x2,y2,cls = b
            pt1 = (x,y)
            pt2 = (x2,y2)
            before = cv2.rectangle(oriimg,pt1,pt2,(124, 255 ,156),thickness=1)
            
        oriimg = cv2.imread(os.path.join(self.img_dir,file+'.jpg'))
        oriimg = cv2.cvtColor(oriimg,cv2.COLOR_BGR2RGB)
        for b in f_box:
            x,y,x2,y2,cls = b
            pt1 = (x,y)
            pt2 = (x2,y2)
            after = cv2.rectangle(oriimg,pt1,pt2,(45, 252 ,34),thickness=1)
        
        with open(os.path.join(self.gt_annotation_dir,file+'.txt')) as f:
            gt = np.array(f.read().split(),dtype=np.uint32).reshape((-1,5))
        for b in gt:
            x,y,w,h,cls = b
            pt1 = (x,y)
            pt2 = (x+w,y+h)
            gt_img = cv2.rectangle(oriimg,pt1,pt2,(0, 0 ,255),thickness=1)
            
        # 创建一个图形窗口
        fig, axs = plt.subplots(2, 2, figsize=(20, 10))

        # 在第一个子图中显示图像1
        axs[0,0].imshow(before)
        axs[0,0].set_title('before')
        axs[0,0].axis('off')  # 关闭坐标轴

        # 在第二个子图中显示图像2
        axs[0,1].imshow(after)
        axs[0,1].set_title('after')
        axs[0,1].axis('off')  # 关闭坐标轴

        # 在第二个子图中显示图像2
        axs[1,0].imshow(gt_img)
        axs[1,0].set_title('gt_img')
        axs[1,0].axis('off')  # 关闭坐标轴
        # 调整布局
        plt.tight_layout()

        # 显示图形
        # plt.show()
        plt.savefig(os.path.join(self.save_dir,"vis","C0001275.png"))
        plt.clf()
        plt.close()
            
            
    def filter_boxes(self, boxes, mask, threshold):
        filtered_boxes = []

        for box in boxes:
            x, y, x2, y2, cls = box

            mask_roi = mask[y:y2, x:x2]

            if np.any(mask_roi == threshold):
                continue
            filtered_boxes.append(box)
            

        return np.array(filtered_boxes)
    
    def save_res(self,file, f_box):
        if not os.path.exists(os.path.join(self.save_dir,'post_txt')):
            os.makedirs(os.path.join(self.save_dir,'post_txt'))
            
        path = os.path.join(self.save_dir,'post_txt',file+'.txt')
        with open(path , "w") as f:
            f.writelines([i+' ' for i in f_box.astype(str).flatten()])
        
with open('/home/hyh/Documents/quanyi/project/Data/e_optha_MA/MA_ex/VOC2012/ImageSets/Main/test.txt', 'r') as f:
    files = f.read().splitlines()

for file in files:
    pp = PostProcess('logs/MA_Detection/hyh_ma_det_exp006/res50_cbam_dcn/iou0.2_score0.3_VOC0')
    boxes = pp.read_pred_box(file)
    fundus_mask = pp.read_fundus_mask(file)
    vessel_mask = pp.read_vessel_mask(file)
    filter_boxes = pp.screen_out(boxes,fundus_mask,vessel_mask)

    pp.save_res(file, filter_boxes)

    pp.visualize(file, filter_boxes)

from typing import Callable
import cv2
import os
import numpy as np
import xml.etree.ElementTree as ET
import random
import math
import argparse
from tqdm import tqdm
from .._base_ import DataProcessBase_
from .._base_ import DataConstructor
import gc
# from .._base_ import DataBase_

# class Process(DataBase_):
#     pass

class patch_data(DataConstructor):
    def __init__(self) -> None:
        super().__init__()
        self.numerical_data_template = None
        self.annotation_template = {'type':None, 'gt':None, 'offset':None}
        self.imgdata_template = {
            'origin':None
        }
        
        self._re_construct()

class GetImagePatch(DataProcessBase_):
    def __init__(self, datastructure=patch_data) -> None:
        self.datastructure = datastructure()
        super().__init__()
        pass
    
    def contain_(self, **kwargs):
        mask = np.all(np.stack([
            kwargs['box_data'][:, 0] >= kwargs['area'][0],
            kwargs['box_data'][:, 1] >= kwargs['area'][1],
            kwargs['box_data'][:, 2] <= kwargs['area'][2],
            kwargs['box_data'][:, 3] <= kwargs['area'][3]
        ], axis=-1), axis=-1)
        return mask
    
    def soft_contain_(self, **kwargs):
        '''
            area = [x_start,y_start,x_end,y_end]
        '''
        box_centre = (kwargs['box_data'][:,0:2] + kwargs['box_data'][:,2:4]) // 2
        mask = np.all(np.stack([
            kwargs['area'][0] < box_centre[:,0],
            kwargs['area'][2] > box_centre[:,0],
            kwargs['area'][1] < box_centre[:,1],
            kwargs['area'][3] > box_centre[:,1]
        ], axis=-1), axis=-1)
        return mask
    
    def cut_patch(self, image_path, annotation_path , patch_size:tuple, overlap:float, screen_out:list=None) -> list:
        '''
        params:
        
            screen_out:list[func], the function to screen out the block data
                input: block image(cv2 mat), area(list) ; output: a 1d mask    
        
        statement:
            annotation : (5*N)-> (x,y,x2,y2,cls)
        
        '''
        patch_lists = []
        
        image = cv2.imread(image_path)
        box_data = []
        if annotation_path:
            with open(annotation_path,"r") as f:
                box_data = np.array(f.read().split(),dtype=int)
                box_data = box_data.reshape((-1,5))
            
        # get image name
        source = os.path.split(image_path)[-1]
        
        # get the central coordinates for each GTbox
        
        step = tuple(int(i*(1-overlap)) for i in patch_size)
        height, width, _ = image.shape
        
        num_blocks_height = math.ceil((height-patch_size[1]) / step[1]) + 1 
        num_blocks_width = math.ceil((width-patch_size[0]) / step[0]) + 1
        
        block_count = 0
        for i in range(num_blocks_height):
            for j in range(num_blocks_width):
                # 计算裁剪边界
                y_start = i * step[1]
                y_end = int(min(y_start + patch_size[1], height))
                x_start = j * step[0]
                x_end = int(min(x_start + patch_size[0], width))
                
                # 修正裁剪边界，若超出图片尺寸，采用从边界往回裁剪的方式
                if(y_start + patch_size[1] > height):
                    y_start = y_end - patch_size[1]
                if(x_start + patch_size[0] > width):
                    x_start = x_end - patch_size[0]

                area = [x_start,y_start,x_end,y_end]
                # 裁剪图片块
                patch_img = image[y_start:y_end, x_start:x_end]
                # 判断该图片块内是否含有目标框，并生成掩膜，用于筛选出在该图片块内的目标框
                block_data = []
                if len(box_data) > 0:
                    mask = self.soft_contain_(patch_img = patch_img, box_data = box_data, area = area)
                    block_data = box_data[mask]
                
                # 若存在其余筛选条件，则进行额外的筛选
                # flag 表示是否保留当前的裁剪
                if screen_out:
                    flag = True
                    for func, params in screen_out:
                        flag = np.logical_and(flag, func(*params, patch_img = patch_img, box_data = box_data, area = area))
                        
                    if not flag:
                        continue
                
                if len(block_data) != 0:
                    # 坐标修正，保存相对坐标
                    block_data[:, 0] = np.maximum(0, block_data[:, 0] - x_start)
                    block_data[:, 1] = np.maximum(0, block_data[:, 1] - y_start)
                    block_data[:, 2] = np.minimum(x_end, block_data[:, 2] - x_start)
                    block_data[:, 3] = np.minimum(y_end, block_data[:, 3] - y_start)
                    classes = block_data[:,4]
                else:
                    classes = np.array([])
                    classes.resize([0,1])
                    block_data = np.array([])
                    block_data.resize([0,5])
                    
                # 保存数据
                name = source[:-4] + f"_block_{block_count}"
                offset = np.array([x_start,y_start,x_start,y_start])
                patch_lists.append(self.datastructure.get_item(source, name, classes, {'origin':patch_img},{'type':"LTRBxyxy",'gt':block_data,'offset':offset}))
                block_count += 1
    
        return patch_lists
    
    
    def cut_image(self, image_dir, annotation_dir, patch_size:tuple, overlap:float, screen_out:list[list[Callable,tuple]]=None):
        '''
        params
            image_dir
            annotation_dir
            patch_size
            overlap
            screen_out：二维列表[[func,params]...],func为自定义的筛选函数,params为额外参数,默认已有参数为,(图片块数据,ma目标框坐标,图片快坐标偏置)
        '''
        # 构建数据路径
        img_list=[]
        annotation_list=[]
        for i in os.listdir(image_dir):
            img_list.append(os.path.join(image_dir,i))
            annotation_list.append(os.path.join(annotation_dir,i[:-4]+".txt"))
            
        # 数据集构建Sp
        data_set=[]
        print('*'*10+'cut patches'+'*'*10)
        for index,(ip,ap) in tqdm(enumerate(zip(img_list,annotation_list)),colour='green'):
            # #test:
            # if index > 3:
            #     break
            
            # 图片裁剪
            patch_lists = self.cut_patch(ip, ap, patch_size, overlap, screen_out)
            data_set.extend(patch_lists)
        return data_set
class MergeImagePatch(GetImagePatch):
    def __init__(self, datastructure=patch_data) -> None:
        super().__init__(datastructure=datastructure)
        
    def merge(self, patch_lists, patch_size, target_size, expend_index):
        '''
        将多个patch合成一个图片, 例如 56*56 合成 112*112，则需要4个patch，通过将patch_lists经过多次随机打乱生成不同的patch_lists保存在candidates
        再从左到右，从上到下进行组合
        params：
        '''
        
        # 创建merge patch 返回列表
        merge_lists = []
        # 记录原图片名字
        source = patch_lists[0]['source']
        # 数据扩充
        patch_lists *= int(expend_index)
        random.shuffle(patch_lists)
        # candidates ，dim= k*len(patch_lists), k表示目标图片由多少个patch构成
        merge_step = (target_size[0] // patch_size[0], target_size[1] // patch_size[1])
        patchs_num = merge_step[0] * merge_step[1]

        # 构建candidates
        '''
        warning: 这种方式在进行非单张图的merge时，也即patch_lists很大，可能会导致内存占用过大，可以采取
                    边拼接边shuffle的方式，不进行candidates的构建吗，对于patch_lists，依次取样拼接，当索引超出时，将其shuffle再从头采样
        有空改一下
        '''
        candidates = []
        for i in range(patchs_num):
            random.seed(i)
            candidates.append(random.sample(patch_lists, k=len(patch_lists)))
        
        for k in range(len(patch_lists)):
            name = f"{source[:-4]}_mix_{k}"
            # 创建空图片，box，classes，用于构建数据字典
            res_image = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
            res_boxes = []
            res_classes = []
            
            # 组合成第k张图片
            for i in range(merge_step[0]):
                for j in range(merge_step[1]):
                    patch_index = i * merge_step[1] + j
                    patch = candidates[patch_index][k]
                    patch_img = patch['image']['origin']
                    patch_boxes = patch['annotation']['gt']
                    patch_classes = patch['classes']
                    
                    # 放置patch到对应位置
                    x_start, y_start = j * patch_size[0], i * patch_size[1]
                    res_image[y_start:y_start + patch_size[1], x_start:x_start + patch_size[0], :] = patch_img

                    # 调整boxes坐标
                    adjusted_boxes = patch_boxes + np.array([x_start, y_start, x_start, y_start, 0])
                    res_boxes.extend(adjusted_boxes)
                    res_classes.extend(patch_classes)
            
            # 构建数据字典，并保存
            res_boxes = np.array(res_boxes)
            res_classes = np.array(res_classes)
            merged_item = self.datastructure.get_item(source, name, res_classes, {'origin':res_image},{'type':"LTRBxyxy",'gt':res_boxes})
            merge_lists.append(merged_item)
        
        return merge_lists
    
    def merge_image(self, image_dir, annotation_dir, patch_size:tuple, overlap:float, target_size:tuple, expend_index:float, screen_out:list=None):
        img_list=[]
        annotation_list=[]
        for i in os.listdir(image_dir):
            img_list.append(os.path.join(image_dir,i))
            annotation_list.append(os.path.join(annotation_dir,i[:-4]+".txt"))
            
        data_set=[]
        print('*'*10+'cut and merge patches'+'*'*10)
        for index,(ip,ap) in tqdm(enumerate(zip(img_list,annotation_list)),colour='green'):
            # #test:
            # if index > 3:
            #     break
            patch_lists = self.cut_patch(ip, ap, patch_size, overlap, screen_out)
            merge_lists = self.merge(patch_lists, patch_size, target_size, expend_index)
            data_set.extend(merge_lists)
        return data_set
    
    def merge_image_out(self, image_dir, annotation_dir, patch_size:tuple, overlap:float, target_size:tuple, expend_index:float, screen_out:list=None):
        '''
        patch 的选取不限于同一张图片
        '''
        img_list=[]
        annotation_list=[]
        for i in os.listdir(image_dir):
            img_list.append(os.path.join(image_dir,i))
            annotation_list.append(os.path.join(annotation_dir,i[:-4]+".txt"))
            
        data_set=[]
        print('*'*10+'cut and merge patches'+'*'*10)
        for index,(ip,ap) in tqdm(enumerate(zip(img_list,annotation_list)),colour='green'):
            # #test:
            # if index > 3:
            #     break
            patch_lists = self.cut_patch(ip, ap, patch_size, overlap, screen_out)
            data_set.extend(patch_lists)
        
        # 裁剪完全部patch，再进行随机拼接
        merge_lists = self.merge(data_set, patch_size, target_size, expend_index)
        return merge_lists

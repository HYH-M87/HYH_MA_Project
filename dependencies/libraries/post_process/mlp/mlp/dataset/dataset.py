
import numpy as np
import os
import shutil
from torch.utils.data import Dataset
import cv2
import torch
from devlib._base_ import DatasetConstructor, DataProcessBase_
class Generate():
    def __init__(self,source_dir ,save_dir, type, patch_size) -> None:
        self.source_dir = source_dir
        self.save_dir = save_dir
        self.type = type
        self.patch_size = patch_size
        self.datasetConstruct = DatasetConstructor("VOC")
        self.source_data_path = self.datasetConstruct.get_path(source_dir)
        self.datasetConstruct.makedirs(save_dir)
        self.dst_data_path = self.datasetConstruct.get_path(save_dir)
        

    # TODO: 对训练集用一阶段的模型预测后的json文件作为构建数据集的依据，可能更有针对性，效果更好 
    def load_json():
        '''
        instances = []
        
        1. 读入json文件
        2. 读入图片
        3. 基于predbox，进行裁剪
        4. 获取predbox的预测类别（可以根据其score设定一个阈值来划分背景和MA）
        5. 构建一个实例，并保存到instances
        6. 返回instances
        
        return instances
        '''
        pass

    
    def get_instances(self):
        '''
        从带背景框的数据集中读取数据
        
        '''
        # 构建实例列表
        instances=[]
        
        # 从原数据集获取数据
        with open(self.source_data_path["TrainSet_Path"], 'r') as f:
            train_data_lists = f.read().splitlines()
            
        with open(self.source_data_path["TestSet_Path"], 'r') as f:
            test_data_lists = (f.read().splitlines())
            
        data_dict = {'trainval.txt':train_data_lists, 'test.txt':test_data_lists}
        
        for k,v in data_dict.items():
            dataset_f = open(os.path.join(self.dst_data_path["ImageSets_Dir"],k), 'w')
            for d in v:
                img_path = os.path.join(self.source_data_path["Image_Dir"], d+".jpg")
                image = cv2.imread(img_path)
                
                txt_path = os.path.join(self.source_data_path["Annotation_Txt_Dir"], d+".txt")
                with open(txt_path,"r") as f:
                    boxes = np.array(f.readline().split()).reshape((-1,5))
                    
                for i, b in enumerate(boxes):
                    cls = b[-1]
                    # 对裁剪的图片resize, 按比例，0填充
                    croped = DataProcessBase_().crop_image(image, b[0:4], self.patch_size)
                    name = d+f"_{i}"
                    dataset_f.write(name+'\n')
                    instances.append([croped, cls, name])
            dataset_f.close()

        return instances
    
    def save(self, instances):
        # 保存图像和标签数据
        image_dir = self.dst_data_path["Image_Dir"]
        txt_dir = self.dst_data_path["Annotation_Txt_Dir"]
        
        for img, cls, name in instances:
            cv2.imwrite(os.path.join(image_dir,name+'.jpg'), img)
            with open(os.path.join(txt_dir, name+'.txt'), 'w') as f:
                f.write(f"{cls}")

    def forward(self):
        instances = self.get_instances()
        self.save(instances)
        

class MA_patch(Dataset):
    def __init__(self, data_dir,is_train ,transform) -> None:
        self.transform = transform
        if is_train:
            with open(os.path.join(data_dir, "VOC2012/ImageSets/Main/trainval.txt"), "r") as f:
                self.files = f.read().splitlines()
        else:
            with open(os.path.join(data_dir, "VOC2012/ImageSets/Main/test.txt"), "r") as f:
                self.files = f.read().splitlines()
        
        self.img_dir = os.path.join(data_dir,"VOC2012/JPEGImages")
        self.txt_dir = os.path.join(data_dir,"VOC2012/Annotations_Txt")
        self.img_files = os.listdir(self.img_dir)      
        
    def __getitem__(self, index):
        file = self.files[index]
        
        image_path = os.path.join(self.img_dir, file+'.jpg')
        txt_path = os.path.join(self.txt_dir, file+'.txt')
        
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Failed to load image: {image_path}")
        
        
        if self.transform:
            image = self.transform(image)
        
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"Annotation file not found for image: {image_path}")
        with open(txt_path,"r") as f:
            data = f.read().split()
        label = torch.tensor([1,0],dtype=torch.float32) if int(data[-1])==0 else torch.tensor([0,1],dtype=torch.float32)
        
        item_info = {
            'image':image,
            'label':label,
            'file':file,
        }
        
        return item_info
    
    
    def __len__(self):
        return len(self.files)


import os
from torch.utils.data import Dataset
import cv2
import torch


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
        label = torch.tensor([1,0],dtype=torch.float32) if int(data[0])==0 else torch.tensor([0,1],dtype=torch.float32)
        
        item_info = {
            'image':image,
            'label':label,
            'file':file
        }
        
        return item_info
    
    
    def __len__(self):
        return len(self.files)
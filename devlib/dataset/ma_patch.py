import random
import numpy as np
from devlib.process_data import GetImagePatch, MergeImagePatch
from devlib._base_ import DatasetConstructor
from tqdm import tqdm
import os
import cv2

class MAPatchDataset():
    def __init__(self, ori_dir, dst_dir, patch_size, overlap, split_by, split_index) -> None:
        '''
        params:
        
        '''
        super().__init__()
        self.dst_dir = dst_dir
        self.ori_dir = ori_dir
        self.dataset_constructor = DatasetConstructor('VOC', Annotation_Txt_Dir='VOC2012/Annotations_Txt')  # 数据集格式
        self.source_path = self.dataset_constructor.get_path(ori_dir)
        self.dst_path = self.dataset_constructor.get_path(dst_dir)
        self.gip = GetImagePatch()
        self.patch_size = patch_size
        self.overlap = overlap
        self.split_by = split_by
        self.split_index = split_index
        
    def get_cfg(self):
        
        cfg={
            "dst_dir":self.dst_dir,
            "ori_dir":self.ori_dir,
            "dst_path":self.dst_path,
            "ori_path":self.source_path,
            "patch_size":self.patch_size,
            "overlap":self.overlap,
            "split_by":self.split_by,
            "split_index":self.split_index
        }
        
        return cfg
        
    def contain_(self, *args, **kwargs):
        mask = np.all(np.stack([
            kwargs['box_data'][:, 0] >= kwargs['area'][0],
            kwargs['box_data'][:, 1] >= kwargs['area'][1],
            kwargs['box_data'][:, 2] <= kwargs['area'][2],
            kwargs['box_data'][:, 3] <= kwargs['area'][3]
        ], axis=-1), axis=-1)
        return mask

    def soft_contain_(self, *args, **kwargs):
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
    
    def ma_filter(self, *args, **kwargs):
        box_centre = (kwargs['box_data'][:,0:2] + kwargs['box_data'][:,2:4]) // 2
        flag = np.any(np.all(np.stack([
            kwargs['area'][0] < box_centre[:,0],
            kwargs['area'][2] > box_centre[:,0],
            kwargs['area'][1] < box_centre[:,1],
            kwargs['area'][3] > box_centre[:,1]
        ], axis=-1), axis=-1))
        return flag
    
    def forward(self):
        
        self.dataset_constructor.makedirs(self.dst_dir)
        data_set = self.gip.cut_image(self.source_path["Image_Dir"],  self.source_path["Annotation_Txt_Dir"], \
                                        self.patch_size, self.overlap, [[self.ma_filter,()]])
        self.cal_img_info(data_set)
        self.save(data_set)
        train_set,test_set = self.data_split(data_set, "image", self.split_index)
        self.dataset_constructor.txt2voc(self.dst_path["Annotation_Txt_Dir"], self.dst_path["Annotation_Dir"], ("MA",))
        
        
    def data_split(self, data_set, split_by, split_index):
        train_set = [] 
        test_set = []
        train_set_path = os.path.join(self.dst_path['ImageSets_Dir'],'trainval.txt')
        test_set_path = os.path.join(self.dst_path['ImageSets_Dir'],'test.txt')
        print('*'*10 + f"spliting dataset"+'*'*10)

        if split_by == "patch":
            random.shuffle(data_set)
            for index,item in tqdm(enumerate(data_set),colour='green'):
                if index <  int(len(data_set)*split_index):
                    train_set.append(item)
                    with open(train_set_path,"a") as f:
                        f.write(item["name"]+"\n")
                else:
                    test_set.append(item)
                    with open(test_set_path,"a") as f:
                        f.write(item["name"]+"\n")

        elif split_by == "image":
            with open(os.path.join(self.source_path['ImageSets_Dir'],'test.txt')) as f:
                flag = f.read().splitlines()
            # flag = os.listdir(self.source_path["Image_Dir"])
            # flag = flag[:int(len(flag)*0.8)+1]
            for index,item in tqdm(enumerate(data_set),colour='green'):
                if item["source"][:-4] in flag:
                    test_set.append(item)
                    with open(test_set_path,"a") as f:
                        f.write(item["name"]+"\n")
                else:
                    train_set.append(item)
                    with open(train_set_path,"a") as f:
                        f.write(item["name"]+"\n")
                    
        
        return train_set, test_set
                    
    def save(self, data_set):
        print('*'*10 + f"saving data"+'*'*10)
        for d in tqdm(data_set,colour='green'):
            img = d["image"]["origin"]
            img_path = os.path.join(self.dst_path["Image_Dir"],d["name"]+".jpg")
            cv2.imwrite(img_path, img)
            
            txt_path = os.path.join(self.dst_path["Annotation_Txt_Dir"], d["name"]+".txt")
            txt_data =d['annotation']['gt'].flatten().astype(str)
            
            with open(txt_path,"w") as f:
                f.writelines([ i+" " for i in txt_data])
                
    def cal_img_info(self, data_set):
        print('*'*10 + f"calculating means and vars for dataset"+'*'*10)
        means = []
        std_devs = []
        for d in tqdm(data_set,colour='green'):
            img = d["image"]["origin"]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mean, std_dev = self.gip.cal_mean_var(img)
            means.append(mean)
            std_devs.append(std_dev)
        all_mean = np.mean(means, axis=0)
        all_var = np.mean(std_devs, axis=0)
        save_path = os.path.join(self.dst_dir, "info.txt")
        with open(save_path, "w") as f:
            f.writelines(f"means=[{all_mean[0]}, {all_mean[1]}, {all_mean[2]}]\n")
            f.writelines(f"vars=[{all_var[0]}, {all_var[1]}, {all_var[2]}]")
        


class MergePatchDataset(MAPatchDataset):
    def __init__(self, ori_dir, dst_dir, 
                patch_size, overlap, target_size, 
                expend_index, split_by, split_index) -> None:
        '''
        params:
        
        '''
        super().__init__(ori_dir,dst_dir,patch_size,overlap,split_by,split_index)
        self.target_size = target_size
        self.expend_index = expend_index
        self.mip = MergeImagePatch()
    
    def forward(self):
        
        self.dataset_constructor.makedirs(self.dst_dir)
        data_set = self.mip.merge_image(self.source_path["Image_Dir"],  self.source_path["Annotation_Txt_Dir"], \
                                        self.patch_size, self.overlap, self.target_size, self.expend_index, [[self.ma_filter,()]])
        self.cal_img_info(data_set)
        self.save(data_set)
        train_set,test_set = self.data_split(data_set, "image", self.split_index)
        self.dataset_constructor.txt2voc(self.dst_path["Annotation_Txt_Dir"], self.dst_path["Annotation_Dir"], ("MA",))
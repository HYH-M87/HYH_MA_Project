import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Any
import json
from .._base_ import DataConstructor
from .._base_ import DataProcessBase_
'''
进度：
    数据保存:待完成
    测试：待完成

'''




class hard_sample(DataConstructor):
    def __init__(self) -> None:
        super().__init__()
        self.annotation_template = {'hard_sample': None}
        self._re_construct()


class HardSampleAnalysis(DataProcessBase_):
    def __init__(self, dataset_cfg:dict, result_dir:str, save_dir:str) -> None:
        super().__init__()
        self.dataconstructor = hard_sample()
        self.dataset_cfg = dataset_cfg
        self.data_path = dataset_cfg['dst_path']
        self.result_dir = result_dir
        self.save_dir = save_dir

    def forward(self):
        hardsample_lists = self.extract_hardsample()
        self.analysis(hardsample_lists)
        self.save_instance_png(hardsample_lists)
        
    def analysis(self, instances, gray:bool=True, rgb:bool=True, fft:bool=True):
        for i in instances:
            if self.save_dir:
                save_path = os.path.join(self.save_dir, i['name'])
                
            img = i['image']['origin']
            
            if gray:
                self.togray(i, img)    
            if rgb:
                self.RGB_DataStatics(i, img)
            if fft:
                r,g,b = cv2.split(img)
                self.fft_DataStatics(i, 0.1*r+0.8*g+0.1*b)
        

    def extract_hardsample(self, patch_size=None):
        '''
        impor 
        读入json
        筛选hard-sample
        提取数据，patchsize，img_name，offset，hard-sample，对应的gt，iou，score
        构建数据
        返回数据列表
        '''
        sample_lists=[]
        files = [os.path.join(self.result_dir,i) for i in os.listdir(self.result_dir)]
        for file_path in files:
            with open(file_path, 'r') as file:
                data = json.load(file)
            hard_sample = data['hard_sample']
            
            if hard_sample is not None:
                source = data['source']
                offset = data['annotation']['offset']
                classes = data['classes']
                name = data['name']
                x_s, y_s = offset[0], offset[1]
                image = cv2.imread(os.path.join(self.data_path['Image_Dir'], source))
                for gt,pred,iou in hard_sample:
                    gt
                if  patch_size:
                    # Error
                    annotation = data['annotation']['gt']
                    annotation[:,0:4] += offset
                    patch = self.cut_patche(image, annotation, patch_size)
                else :
                    patch_size = data['image']
                    patch = image[y_s:y_s+patch_size[1], x_s:x_s+patch_size[0],:]
                
                sample = self.dataconstructor.get_item(source, name, classes, {'origin':patch}, {'hard_sample':hard_sample})
                sample_lists.append(sample)
            
        return sample_lists
    
    
    def cut_patche(self, image, annotations, patch_size = 28):
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        (x1, y1, x2, y2, cls) = (annotations)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        half_size = patch_size // 2
        start_x = max(center_x - half_size, 0)
        start_y = max(center_y - half_size, 0)
        end_x = min(center_x + half_size, image.shape[1])
        end_y = min(center_y + half_size, image.shape[0])
        patch = image[start_y:end_y, start_x:end_x]

        return patch
            

    def togray(self, i:dict, img:np.ndarray):
        i['image']['gray'] = self.gray_img(img)
        
    def RGB_DataStatics(self, i, img):
        '''
        RGB :max, min, mean, std, histogram
        '''
        means, vars = self.cal_mean_var(img)
        mins, maxs = self.cal_min_max(img)
        hists = self.histogram(img)

        types = ['r','g','b']

        for (t,max,min,mean,var,hist) in zip(types,maxs,mins,means,vars,hists):
            i['numerical'][t] = list((max,min,mean,var,hist))

    def fft_DataStatics(self, i, img):
        img_dft = self.dft(img, shift=False)
        
        magnitude_spectrum = 20 * np.log(cv2.magnitude(img_dft[:,:,0], img_dft[:,:,1]))
        means, vars = self.cal_mean_var(magnitude_spectrum)
        mins, maxs = self.cal_min_max(magnitude_spectrum)
        
        rows, cols = img.shape
        # 生成频率网格
        u = np.fft.fftfreq(rows).reshape(-1, 1)
        v = np.fft.fftfreq(cols).reshape(1, -1)
        freq = np.sqrt(u**2 + v**2)

        # Flatten the frequency and magnitude arrays
        freq_flat = freq.flatten()
        magnitude_flat = magnitude_spectrum.flatten()

        # Sort by frequency
        sorted_indices = np.argsort(freq_flat)
        i['numerical']['f'] = magnitude_flat[sorted_indices]
        
        img_dft  = np.fft.fftshift(img_dft)
        i['image']['frequency'] = np.uint8(cv2.normalize(cv2.magnitude(img_dft[:, :, 0], img_dft[:, :, 1]), None, 0, 255, cv2.NORM_MINMAX)) 
        
        low_mask, high_mask = self.dft_mask(img.shape, 0.95)
        
        def f_(img, mask=None):
            if mask is not None:
                img = img * mask
            img = self.idft(img)
            img = cv2.magnitude(img[:, :, 0], img[:, :, 1])
            img = np.uint8(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX))
            return img
        
        hpf_img = f_(img_dft, high_mask)
        lpf_img = f_(img_dft, low_mask)
        i['image']['HPF'] = hpf_img
        i['image']['LPF'] = lpf_img

    def read_annotations(self, file_path):
        with open(file_path, 'r') as file:
            lines = np.array([int(i) for i in file.readline().split()]).reshape((-1,5))
        annotations = []
        for line in lines:
            x1, y1, x2, y2, cls = map(int, line)
            annotations.append((x1, y1, x2, y2, cls))
        return annotations

    def save_instance_scale(self):
        
        pass
    
    def save_instance_png(self, instances):
        for ins in instances:
            image_num = len(ins['image'])
            c = math.ceil(image_num / 2)
            r = math.ceil(image_num / c)
            
            fig, axs = plt.subplots(r, c, figsize=(8, 16)) 
            image_data = ins['image'].copy()
            hard_samples=ins['annotation']['hard_sample']
            
            for i in range(r):
                for j in range(c):
                    
                    image_data.popitem()
                    
                    
                    
                    if i*c+j < len(keys) :
                        
                        res_img = cv2.rectangle(ins['image'][keys[i*c+j]],pt1, pt2,(0,0,0),1)
                        axs[i, j].imshow(res_img)
                        axs[i, j].set_title(keys[i*c+j])
                        
                    if len(keys)  <= i*c+j < len(keys)+1:
                        for color in ['r','g','b']: 
                            hist = (ins['numerical'][color]['histogram']).reshape((-1,1))
                            axs[i, j].plot(hist,color = color)
                    if len(keys)+1  <= i*c+j < len(keys)+2:
                        hist = (ins['numerical']['fft_all']['histogram']).reshape((-1,1))
                        axs[i, j].plot(hist,color = 'r')
                        
            plt.tight_layout()
            plt.savefig(os.path.join(self.save,ins["source"]+ins["name"]+".png"))
            # plt.show()
            plt.clf()
            plt.close()



# 测试函数
if __name__ == "__main__":

    # 设置图像和注释文件所在的目录
    image_directory = 'test/hard_sample_res/img'
    annotation_directory = 'test/hard_sample_res/txt'
    save_dir = "test/hard_sample_res/save"
    
    analye = HardSampleAnalysis("VOC",image_directory,annotation_directory,save_dir)
    analye.analysis()
    # process_images_and_annotations(image_directory, annotation_directory, save_dir)

import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Any
import json
from .._base_ import DataConstructor
from .._base_ import DataProcessBase_
from .._base_ import NumpyDecoder

class hard_sample(DataConstructor):
    def __init__(self) -> None:
        super().__init__()
        self.imgdata_template = {
            'origin': None,
            'gray': None,
            'frequency': None,
            'HPF':None,
            'LPF':None
        }
        self.annotation_template = {'hard_sample': None}
        self._re_construct()


class HardSampleAnalysis(DataProcessBase_):
    def __init__(self, dataset_cfg:dict, json_dir:str, save_dir:str) -> None:
        super().__init__()
        self.dataconstructor = hard_sample()
        self.dataset_cfg = dataset_cfg
        self.data_path = dataset_cfg['dst_path']
        self.json_dir = json_dir
        self.save_dir = save_dir

    def forward(self):
        hardsample_lists = self.extract_hardsample()
        self.analysis(hardsample_lists)
        self.save_instance_png(hardsample_lists)
        
    def analysis(self, instances):
        for i in instances:
            # 创建每个实例的保存目录
            if self.save_dir:
                save_path = os.path.join(self.save_dir, i['name'])
                os.makedirs(save_path)
                
            img = i['image']['origin']

            self.togray(i, img)    
            
            self.RGB_DataStatics(i, img)
            
            self.fft_DataStatics(i, self.gray_img(img))
        

    def extract_hardsample(self):
        # 构建hard sample 数据列表
        sample_lists=[]
        files = [os.path.join(self.json_dir,i) for i in os.listdir(self.json_dir)]
        for file_path in files:
            with open(file_path, 'r') as file:
                data = json.load(file, cls=NumpyDecoder)
            hard_sample = data['annotation']['hard_sample']
            
            # 筛选出带有hard sample 的数据块，并构建数据列表
            if hard_sample is not None and len(hard_sample) !=0 :
                source = data['source']
                offset = data['annotation']['offset']
                classes = data['classes']
                name = data['name']
                x_s, y_s = offset[0], offset[1]
                image = cv2.imread(os.path.join(self.data_path['Image_Dir'], source))
                patch_size = data['image']
                # 也可以使用self.cut_patch 进行裁剪
                # bbox = data['annotation']['hard_sample'][0][0][0:4]
                # patch = self.cut_patch(image, bbox+offset, 56)
                patch = image[y_s:y_s+patch_size[1], x_s:x_s+patch_size[0],:]
                
                sample = self.dataconstructor.get_item(source, name, classes, {'origin':patch}, {'hard_sample':hard_sample})
                sample_lists.append(sample)
            
        return sample_lists
    
    
    def cut_patch(self, image, annotations, patch_size = 28):
        '''
            通过hard sample bbox进行任意大小patch的裁剪
        '''
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
        '''
        转灰度图
        '''
        i['image']['gray'] = self.gray_img(img)
        
    def RGB_DataStatics(self, i, img):
        '''
        统计RGB三通道的最大值，最小值，均值，标准差和直方图
        RGB :max, min, mean, std, histogram
        '''
        means, vars = self.cal_mean_var(img)
        mins, maxs = self.cal_min_max(img)
        hists = self.histogram(img)

        types = ['r','g','b']

        for (t,max,min,mean,var,hist) in zip(types,maxs,mins,means,vars,hists):
            i['numerical'][t] = list((max,min,mean,var,hist))
        
    def fft_DataStatics(self, i, img):
        '''
        对图像进行高通滤波和低通滤波，并保存
        '''
        # 傅里叶变换
        img_dft = self.dft(img, shift=False)
        # 计算幅值，并做数值统计
        magnitude_spectrum = 20 * np.log(cv2.magnitude(img_dft[:,:,0], img_dft[:,:,1]))
        means, vars = self.cal_mean_var(magnitude_spectrum)
        mins, maxs = self.cal_min_max(magnitude_spectrum)
        
        rows, cols = img.shape
        # 生成频率网格
        u = np.fft.fftfreq(rows).reshape(-1, 1)
        v = np.fft.fftfreq(cols).reshape(1, -1)
        freq = np.sqrt(u**2 + v**2)

        # 展平频率和幅值数据
        freq_flat = freq.flatten()
        magnitude_flat = magnitude_spectrum.flatten()

        # 将频率从小到大排序，并且幅值也按对于频率进行排序
        sorted_indices = np.argsort(freq_flat)
        i['numerical']['f'] = magnitude_flat[sorted_indices]
        
        # 中心偏移
        img_dft  = np.fft.fftshift(img_dft)
        i['image']['frequency'] = np.uint8(cv2.normalize(cv2.magnitude(img_dft[:, :, 0], img_dft[:, :, 1]), None, 0, 255, cv2.NORM_MINMAX)) 
        
        # 计算高低通掩膜
        low_mask, high_mask = self.dft_mask(img.shape, 0.95)
        
        # 应用高低通滤波器
        def f_(img, mask=None):
            if mask is not None:
                img = img * mask
            img = self.idft(img)
            img = cv2.magnitude(img[:, :, 0], img[:, :, 1])
            img = np.uint8(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX))
            return img
        
        # 保存
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



    def save_instance_png(self, instances):
        '''
            保存实例数据，每个图片patch为一个目录，其中保存其分析数据
        '''
        for ins in instances:
            # for image data
            name = ins['name']
            boxes = ins['annotation']['hard_sample']
            for k,img in ins['image'].items():
                save_path = os.path.join(self.save_dir, name, k+'.jpg')
                for b in boxes:
                    x,y,x2,y2 = map(int, b[0])
                    cv2.imwrite(save_path, img)
                    img = cv2.rectangle(img, (x,y), (x2,y2), (255,0,0),1)
                    cv2.imwrite(save_path, img)



if __name__ == "__main__":
    # for debug
    # 设置图像和注释文件所在的目录
    image_directory = 'test/hard_sample_res/img'
    annotation_directory = 'test/hard_sample_res/txt'
    save_dir = "test/hard_sample_res/save"
    
    analye = HardSampleAnalysis("VOC",image_directory,annotation_directory,save_dir)
    analye.analysis()
    # process_images_and_annotations(image_directory, annotation_directory, save_dir)

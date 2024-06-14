import cv2
import numpy as np


from .DataBased import DataBase_


class DataProcessBase():
    def __init__(self) -> None:
        pass
        
    
    def gray_img(self, img:np.ndarray):
    
        b,g,r = cv2.split(img)
        return 0.1*b+0.8*g+0.1*r
    
    def cal_min_max(self, img:np.ndarray):
        # 检查图像是否读取成功
        if img is None:
            raise ValueError("load img failed")

        min_values = []
        max_values = []

        # 对每个通道分别计算最小值和最大值
        if len(img.shape)==3:
            for i in range(img.shape[2]):
                channel = img[:, :, i]
                min_values.append(np.min(channel))
                max_values.append(np.max(channel))
        else:
            min_values.append(np.min(img))
            max_values.append(np.max(img))
        return min_values, max_values

    def histogram(self, img:np.ndarray):

        if img is None:
            raise ValueError("load img failed")
        
        hist_data=[]
        if len(img.shape)==3:
            for i in range(img.shape[2]):
                hist = cv2.calcHist([img], [i], None, [256], [0, 256])
                hist_data.append(hist.flatten())
        else: 
            hist = cv2.calcHist([img], [0], None, [256], [0, 256])
            hist_data.append(hist.flatten())

        return hist_data
    
    def cal_mean_var(self, img:np.ndarray):
        '''
        calculate the mean value and variance value of a batch of images
        '''
        image_float = img.astype(np.float32)
        mean, std_dev = cv2.meanStdDev(image_float)

        mean = mean.flatten()
        std_dev = std_dev.flatten()
        
        return mean, std_dev
    
    def dft(self, img, shift:bool=True):
        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        if shift:
            dft_shift = np.fft.fftshift(dft)
            return dft_shift
        return dft
    
    def idft(self, dft ,shift:bool=True):
        if shift:
            dft = np.fft.ifftshift(dft)

        return cv2.idft(dft)
    
    def dft_mask(self, shape, k):

        # 获取图像大小gray
        rows, cols = shape
        crow, ccol = rows // 2, cols // 2  # 中心位置

        # 创建高通滤波器和低通滤波器的掩码
        mask_low = np.zeros((rows, cols, 2), np.uint8)
        mask_high = np.ones((rows, cols, 2), np.uint8)

        # 定义滤波器的半径
        rad = [rows*(k),rows*(1-k)]
        center = (crow, ccol)
        cv2.circle(mask_low, center, int(rad[0]), (1, 1), thickness=-1)
        cv2.circle(mask_high, center, int(rad[1]), (0, 0), thickness=-1) 

        return mask_low, mask_high
    
    def HPF(self, dft_shift, mask):
        return dft_shift * mask
    
    def LPF(self, dft_shift, mask):
        return dft_shift * mask
    
    
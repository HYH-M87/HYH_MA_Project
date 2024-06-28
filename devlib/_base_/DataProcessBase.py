import os
import cv2
import numpy as np


class DataProcessBase_():
    def __init__(self) -> None:
        pass
    def resize_and_pad(self, image, target_size):
        # 获取输入图像的尺寸
        h, w = image.shape[:2]
        target_h, target_w = target_size
        
        if h==target_h and w ==target_w:
            return image
        
        # 计算缩放比例
        scale = min(target_w / w, target_h / h)
        
        # 缩放图像
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # 创建一个目标尺寸的全0数组（黑色背景）
        new_image = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # 计算填充的位置
        pad_left = (target_w - new_w) // 2
        pad_top = (target_h - new_h) // 2
        
        # 将缩放后的图像放置到新图像中
        new_image[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized_image
        
        return new_image

    
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
    
    def mask2box(self, cls:int, gtmask_dir:str, txt_dir:str=None):
        '''
        convert the binary mask image to bt box coordinate (LTRBxyxy) and save as txt file
        params:
                cls : the class of the mask
                gtmask_dir: dir path of binary mask images
                txt_dir: save path of txt files
        return: 
                None
        '''
        if txt_dir is None:
            txt_dir = self.Annotation_Txt
            
        if(not os.path.exists(txt_dir)):
            os.makedirs(txt_dir)
            
        lists = os.listdir(gtmask_dir)
        

        for l in lists:
                
            img_path = os.path.join(gtmask_dir,l)
            try:
                mask = (cv2.imread(img_path))[:,:,0]
            except:
                print(img_path)
            box_num, labels, boxes, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            boxes = (boxes[boxes[:,4].argsort()])[:-1]
            boxes[:,2:4] = boxes[:,0:2] + boxes[:,2:4]
            boxes[:,4] = cls
            with open(os.path.join(txt_dir,l[:-4])+".txt","w") as f:
                for b in boxes:
                    b = [str(i)+" " for i in b]
                    f.writelines(b)
    
    def HPF(self, dft_shift, mask):
        return dft_shift * mask
    
    def LPF(self, dft_shift, mask):
        return dft_shift * mask
    
    
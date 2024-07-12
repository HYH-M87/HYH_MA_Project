import os
import cv2
import numpy as np
import json
from devlib._base_ import NumpyDecoder
from devlib._base_ import DatasetConstructor
def make_dir(name):
    img="VOC2012/JPEGImages"
    label="VOC2012/Annotations_Txt"
    imageset = "VOC2012/ImageSets/Main"
    for p in [img,label,imageset]:
        if not os.path.exists(os.path.join(name,p)):
            os.makedirs(os.path.join(name,p))

def resize_and_pad(image, target_size):
    # 获取输入图像的尺寸
    h, w = image.shape[:2]
    target_h, target_w = target_size

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

class extract_sample():
    def __init__(self) -> None:
        self.save_dir = "fortest/extract"
        self.image_dir="/home/hyh/Documents/quanyi/project/Data/e_optha_MA/MA_ex/VOC2012/JPEGImages"
        self.json_dir="fortest/model_eval_0.2_0.6_t/result"
        self.size = [56,56]
        self.dataset = DatasetConstructor('VOC')
        self.dataset.makedirs(self.save_dir)
        
    
    def load_result(self,result_dir):
        '''
        读入数据
        '''
        result_lists=[]
        json_files = [os.path.join(result_dir, i) for i in os.listdir(result_dir)]
        for i in json_files:
            with open(i,'r') as f:
                data = json.load(f, cls=NumpyDecoder)
            result_lists.append(data)
        
        return result_lists
    

    def forward(self):
        '''
        读入json文件，并转为字典，返回列表
        '''
        patch_lists = self.load_result(self.json_dir)
        for p in patch_lists:
            image = cv2.imread(os.path.join(self.image_dir, p["source"]))
            k=0
            offset = p['annotation']['offset']
            for i in p['annotation']['gt']:
                i[0:4] += offset
                save_name = f"{p['name']}_ins{k}"
                x,y,x2,y2,cls = map(int, i)
                crop = image[y:y2,x:x2,:]
                crop = resize_and_pad(crop, self.size)
                cv2.imwrite(os.path.join(self.save_dir, "VOC2012/JPEGImages", save_name+'.jpg'),crop)
                with open(os.path.join(self.save_dir, "VOC2012/Annotations_Txt", save_name+'.txt'), 'w') as f:
                    f.write(f'{x} {x2} {y} {y2} {cls}')
                k+=1
            for i, box in enumerate(p['annotation']['pred']):
                box += offset
                save_name = f"{p['name']}_ins{k}"
                x,y,x2,y2 = map(int, box)
                crop = image[y:y2,x:x2,:]
                crop = resize_and_pad(crop, self.size)
                cv2.imwrite(os.path.join(self.save_dir,"VOC2012/JPEGImages", save_name+'.jpg'),crop)
                if p['annotation']['iou'] is None:
                    cls = 1
                else:
                    if int(p['annotation']['iou'][i][1]) == -1:
                        cls = 1
                    else:
                        cls = 0
                with open(os.path.join(self.save_dir, "VOC2012/Annotations_Txt", save_name+'.txt'), 'w') as f:
                    f.write(f'{x} {x2} {y} {y2} {cls}')
                k+=1

        
        
        
    


def process_images_and_boxes(image_dir, txt_dir, output_dir, dataset, size=(24, 24)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(dataset,"r") as f:
        image_files = f.read().splitlines()
    
    # 获取目录中的所有图像文件
    # image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
    
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file+".jpg")
        txt_path = os.path.join(txt_dir, image_file+".txt") 
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue
        
        # 读取样本框数据
        if not os.path.exists(txt_path):
            print(f"Annotation file not found for image: {image_path}")
            continue
        
        with open(txt_path, 'r') as f:
            boxes = f.readline().strip().split()
        
        # 检查是否有数据
        if len(boxes) % 5 != 0:
            print(f"Annotation data format error in file: {txt_path}")
            continue
        
        boxes = np.array(boxes, dtype=np.float32).reshape(-1, 5)
        
        for i, b in enumerate(boxes):
            # 裁剪样本框
            # x, y, x2, y2, cls = map(int, b)
            # cropped = image[y:y2, x:x2]
            x, y, x2, y2, cls = map(int, b)
            cropped = image[y:y2, x:x2]
            # 调整大小
            resized = resize_and_pad(cropped,(112,112))
            # resized = cv2.resize(cropped, size ,interpolation=cv2.INTER_CUBIC)
            
            # 保存处理后的图像
            output_filename = f"{os.path.splitext(image_file)[0]}_box{i}_cls{int(cls)}"
            img_output_path = os.path.join(output_dir, "VOC2012/JPEGImages",output_filename+".jpg")
            txt_output_path = os.path.join(output_dir, "VOC2012/Annotations_Txt",output_filename+".txt")
            # projection_path = os.path.join(output_dir, "projection/projection.txt",output_filename+".txt")
            cv2.imwrite(img_output_path, resized)
            
            
            with open(txt_output_path,"w+") as f:
                f.write(f'{x} {y} {x2} {y2} {cls}')
            with open(os.path.join(output_dir, "VOC2012/ImageSets/Main", os.path.split(dataset)[-1]),"a+") as f:
                f.write(output_filename+'\n')
                

            print(f"Saved processed image: {img_output_path}")

# image_directory = 'MA_healthy_ex/VOC2012/JPEGImages' 
# # txt_dir="/home/hyh/Documents/quanyi/project/MA_Project/logs/MA_Detection/hyh_ma_det_exp006/res50_cbam_dcn/iou0.2_score0.3_VOC0/post_process/post_txt"
# txt_dir="MA_healthy_ex/VOC2012/Annotations_Txt"
# dataset = "MA_healthy_ex/VOC2012/ImageSets/Main/trainval.txt"
# # output_directory = '/home/hyh/Documents/quanyi/project/MA_Project/logs/MA_Detection/hyh_ma_det_exp006/res50_cbam_dcn/iou0.2_score0.3_VOC0/post_process'  
# output_directory = "extract_sample_112"
# make_dir(output_directory)
# process_images_and_boxes(image_directory, txt_dir, output_directory, dataset)

e = extract_sample()
e.forward()

import cv2
import os
import numpy as np
import xml.etree.ElementTree as ET
import random
import math
import argparse


class DataProcess:
    
    def __init__(self, Type:str=None) -> None:
        '''
            params:
                    Type: the type of dataset,("VOC","COCO")
                    data_name: the topest directory of the dataset, 
        '''
        self.Type = Type
        if Type=="VOC":
            self.Image_Path = "VOC2012/JPEGImages"
            self.Annotation_Path = "VOC2012/Annotations"
            self.Train_Data = "VOC2012/ImageSets/Main/trainval.txt"
            self.Test_Data = "VOC2012/ImageSets/Main/test.txt"
            self.ImageSets = "VOC2012/ImageSets/Main"
            self.Info = "VOC2012/info.txt"
            self.Annotation_Txt = "VOC2012/Annotations_Txt"
        pass

    def make_dir(self, name):
        if not os.path.exists(name):
            os.makedirs(name)
            if(self.Type=="VOC"):
                os.makedirs(os.path.join(name,"VOC2012"))
                os.makedirs(os.path.join(name,self.Annotation_Path))
                os.makedirs(os.path.join(name,self.Image_Path))    
                os.makedirs(os.path.join(name,self.Annotation_Txt))        
                os.makedirs(os.path.join(name,self.ImageSets))  
        
    
    def calculate_mean_variance(self, img_dir:str, info_out:str) -> list[int]:
        '''
        calculate the mean value and variance value of a batch of images
        
        params:
                image_paths : dir of the .jpg file. the default value is self.Image_Path
        return:
                means: means of each chanels (BGR)
                variances: variances of each chanels (BGR)
        '''
        means = [0, 0, 0]
        variances = [0, 0, 0]
        
        if(not os.path.exists(img_dir)):
            print("{} is not exists".format(img_dir))
            return
        
        
        images_list = [os.path.join(img_dir,i) for i in os.listdir(img_dir)]
        
        for image_path in images_list:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Unable to load image {image_path}")
                continue
            
            image_float = image.astype(np.float32)
            
            mean = np.mean(image_float, axis=(0, 1))
            variance = np.var(image_float, axis=(0, 1))
            
            means = [m + mv for m, mv in zip(means, mean)]
            variances = [v + vv for v, vv in zip(variances, variance)]
        
        total_images = len(os.listdir(img_dir))
        means = [m / total_images for m in means]
        variances = [v / total_images for v in variances]
        
        with open(info_out,"w+") as f:
            m = [str(i)+"," for i in means]
            v = [str(i)+"," for i in variances]
            f.writelines(["mean: "]+m)
            f.writelines("\n")
            f.writelines(["variances: "]+v)
        
        return means, variances
    
    def txt2xml(self,txt_dir:str ,xml_dir:str ,cn:tuple) -> None:
        '''
        convert the txt file to VOC Dataset xml file;
        txt format:
            824 1498 7 8 0 1142 988 11 13 0  -> N*(x,y,w,h,classes) .....
            
        params:
                txt_dir: dir path of  txt files
                xml_dir: save path of xml files
                cn: class name ,example: ("MA","background")
        return:
                None
        '''
        
        if(not os.path.exists(txt_dir)):
            print("{} is not exists".format(txt_dir))
            return
        if(not os.path.exists(xml_dir)):
            os.makedirs(xml_dir)
        
        txt_list = os.listdir(txt_dir)
        for t in txt_list:

            with open(os.path.join(txt_dir,t), 'r') as f:
                data_line = np.array(f.readline().split())  

            bboxes = data_line.reshape((-1,5))

            xml_root = ET.Element('annotation')

            for bbox in bboxes:
                x, y, width, height, index= map(int, bbox)
                bndbox = ET.SubElement(xml_root, 'object')
                ET.SubElement(bndbox, 'name').text = cn[int(index)]
                ET.SubElement(bndbox, 'pose').text = 'Unspecified'
                ET.SubElement(bndbox, 'truncated').text = '0'
                ET.SubElement(bndbox, 'difficult').text = '0'

                bndbox_elem = ET.SubElement(bndbox, 'bndbox')
                ET.SubElement(bndbox_elem, 'xmin').text = str(x)
                ET.SubElement(bndbox_elem, 'ymin').text = str(y)
                ET.SubElement(bndbox_elem, 'xmax').text = str(x + width)
                ET.SubElement(bndbox_elem, 'ymax').text = str(y + height)

            tree = ET.ElementTree(xml_root)
            tree.write(os.path.join(xml_dir,t[:-4]+".xml"), encoding='utf-8', xml_declaration=True)
    
    
    def mask2gtbox(self, gtmask_dir:str, txt_dir:str) -> None:
        '''
        convert the binary mask image to bt box coordinate and save as txt file
        params:
                gtmask_dir: dir path of binary mask images
                txt_dir: save path of txt files
        return: 
                None
        '''

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
            boxes[:,4] = 0
            with open(os.path.join(txt_dir,l[:-4])+".txt","w") as f:
                for b in boxes:
                    b = [str(i)+" " for i in b]
                    f.writelines(b)
    
    
    def data_split(self, datalist_dir:str, out_path:tuple, k:int=0.8) -> None:
        '''
        split the Dataset into train dataset and eval dataset
        params:
                datalist_dir: dir of the data list 
                out_path: save path of the result tx file, a tuple ("trainval.txt","text.txt")
                k: the proportion of train dataset (0~1)
        return:
                None
        
        '''
        image_files = os.listdir(datalist_dir)
        
        train_count = int(len(image_files) * k)
        test_count = len(image_files) - train_count

        train_files = random.sample(image_files, train_count)
        test_files = [f for f in image_files if f not in train_files]

        with open(out_path[0], 'w') as f:
            for file_name in train_files:
                f.write(file_name[:-4] + '\n')

        with open(out_path[1], 'w') as f:
            for file_name in test_files:
                f.write(file_name[:-4] + '\n')
        print("-"*10+"dataset split complete"+"-"*10)
        

    def image_cut(self, img_dir:tuple, label_dir:tuple, block:list, overlap:float) -> tuple:
        '''
        cut the image block that contains target from origin picture
        params:
                img_dir: dir path of origin images 
                label_dir: dir path of origin txt annotations defined as ...
                save_dir: the parent directory of the result composed of two Dirs, images and  annotations_txt
                block: the image block size 
                overlap: the overlap index (0~1) 
                creat_xml: creat xml Annotations file 
        return: 
                a tuple contains image block save path and image xml annotations save path
        '''


        image_block_save_path = img_dir[1]
        annotation_block_save_path = label_dir[1]
        
        if not os.path.exists(image_block_save_path):
            os.makedirs(image_block_save_path)
        if not os.path.exists(annotation_block_save_path):
            os.makedirs(annotation_block_save_path)
        
        block_size = block
        step = [int(i*overlap) for i in block]
        # step = 80

        img_list = [i[:-4]for i in os.listdir(img_dir[0])]
        img_path = [os.path.join(img_dir[0],i+".jpg") for i in img_list]
        box_path = [os.path.join(label_dir[0],i+".txt") for i in img_list]

        print("-"*10+"data loading complete"+"-"*10)
        
        for index in range(len(img_list)):
            block_count = 0
            # read the image and load its GT box
            image = cv2.imread(img_path[index])
            with open(box_path[index],"r") as f:
                box_data = np.array(f.readline().split(),dtype=int)
                box_data = (box_data.reshape((-1,5)))
                
            
            # get the central coordinates for each GTbox
            box_num = box_data.shape[0]
            box_centre = np.zeros((box_num,2))
            for i in range(box_num):
                box_centre[i] = box_data[i,0:2] + box_data[i,2:4]/2

            height, width, _ = image.shape
            
            num_blocks_height = math.ceil((height-block_size[1]) / step[1]) + 1 
            num_blocks_width = math.ceil((width-block_size[0]) / step[0]) + 1

            for i in range(num_blocks_height):
                for j in range(num_blocks_width):
                    
                    # cut the block from image
                    y_start = i * step[1]
                    y_end = int(min(y_start + block_size[1], height))
                    x_start = j * step[0]
                    x_end = int(min(x_start + block_size[0], width))
                    
                    # if the block exceed the image resolution, then cut it back from the boundary
                    
                    if(y_start + block_size[1] > height):
                        y_start = y_end - block_size[1]
                    if(x_start + block_size[0] > width):
                        x_start = x_end - block_size[0]
                    
                    block = image[y_start:y_end, x_start:x_end]
   
                    
                    
                    # select all the block that contain MA
                    x = np.array([ (x_start<i[0]<x_end) for i in box_centre])
                    y = np.array([ (y_start<i[1]<y_end) for i in box_centre])
                    cobj = x*y
                    if(sum((x*y).astype(int))==0):
                        continue
                    
                    # Coordinate correction
                    block_data = box_data[cobj]
                    for bb in block_data:
                        bb[0] = max(0,bb[0]-x_start)
                        bb[1] = max(0,bb[1]-y_start)
                        bb[2] = min(x_end-bb[0],bb[2])
                        bb[3] = min(y_end-bb[1],bb[3])

                    
                    
                    # save block
                    block_filename = img_list[index] + f"_block_{block_count}"
                    
                    
                    cv2.imwrite(os.path.join(image_block_save_path,block_filename+".jpg"), block)
                    with open(os.path.join(annotation_block_save_path,block_filename+".txt"),"w") as f:
                        for box in block_data:
                            f.writelines([str(i)+" " for i in box])
                    

                    # print(f"Saved block {img_list[index]}_{block_count} as {block_filename}")
                    block_count += 1
        print("-"*10+"image cutting complete"+"-"*10)

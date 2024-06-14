from .DataBased import DataBase_
import os
import xml.etree.ElementTree as ET
import numpy as np
from typing import Union, Any

import cv2

class DatasetConstruct(DataBase_):
    def __init__(self, Type: str | dict) -> None:
        super().__init__(Type)
        
    def makedirs(self, name:str):
        '''
        param: 
            name: the path to the topest directory; 
        
        '''
        if not os.path.exists(name):
            if not os.path.exists(name):
                os.makedirs(os.path.join(name,self.Annotation_Path))
                os.makedirs(os.path.join(name,self.Image_Path))    
                os.makedirs(os.path.join(name,self.Annotation_Txt))        
                os.makedirs(os.path.join(name,self.ImageSets))  

    
    def mask2box(self, cls:int, gtmask_dir:str, txt_dir:str=None):
        '''
        convert the binary mask image to bt box coordinate and save as txt file
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
            boxes[:,4] = cls
            with open(os.path.join(txt_dir,l[:-4])+".txt","w") as f:
                for b in boxes:
                    b = [str(i)+" " for i in b]
                    f.writelines(b)
    
    def convert(self):
        pass
    
    def coco2voc(self):
        pass
    
    def generate_txt(self):
        pass
    
    def txt2voc(self, txt_dir:str ,xml_dir:str ,cn:tuple) -> None:
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
                data_line = np.array(f.readline().split()).astype(int)  

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
            
            
    def txt2coco(self):
        pass



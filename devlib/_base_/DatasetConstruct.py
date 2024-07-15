import copy

from tqdm import tqdm
from .DataBase import DatasetBase_
import os
import xml.etree.ElementTree as ET
import numpy as np
from typing import Union, Any

import cv2

class DatasetConstructor(DatasetBase_):
    def __init__(self, Cfg:Union[str, dict], **kwargs) -> None:
        '''
        数据集构建
        初始化参数：
            Cfg:     传入字符串['VOC', ]，表示使用已定义的数据集格式例如“VOC”否则传入自定义配置文（dict）
            **kwargs 可针对特殊数据集添加新的路径信息
            
        example：
            # 传入额外路径
            my_dataset = DatasetConstructor('VOC', {'additional_path': Path})
            
        '''
        super().__init__(Cfg)
        if len(kwargs) != 0:
            self.add_(kwargs)
            
    # 添加额外路径
    def add_(self, more_dir:dict):
        for k,v in more_dir.items():
            self.data_path_cfg[k] = v
            
    # 构建目录
    def makedirs(self, name:str):
        '''
        param: 
            name: the path to the topest directory; 
        '''
        print('*'*10+'making dataset directory'+'*'*10)
        if not os.path.exists(name):
            for k,v in self.data_path_cfg.items():
                if k.endswith('Dir'):
                    os.makedirs(os.path.join(name,v))
                    print(f'make : {os.path.join(name,v)}')
        else:
            print('dataset has already existed')
            
    # 获取该数据集的所有路径信息
    def get_path(self, name):

        path_dict = copy.deepcopy(self.data_path_cfg)
        for k,v in self.data_path_cfg.items():
            path_dict[k] = os.path.join(name,v)
        return path_dict
    

    
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
            N*(x,y,x,y,classes) .....
            
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
        print('*'*10 + 'changing txt to voc xml file'+'*'*10)
        for t in tqdm(txt_list,colour='green'):

            with open(os.path.join(txt_dir,t), 'r') as f:
                data_line = np.array(f.readline().split()).astype(int)  

            bboxes = data_line.reshape((-1,5))

            xml_root = ET.Element('annotation')

            for bbox in bboxes:
                x, y, x2, y2, index= map(int, bbox)
                bndbox = ET.SubElement(xml_root, 'object')
                ET.SubElement(bndbox, 'name').text = cn[int(index)]
                ET.SubElement(bndbox, 'pose').text = 'Unspecified'
                ET.SubElement(bndbox, 'truncated').text = '0'
                ET.SubElement(bndbox, 'difficult').text = '0'

                bndbox_elem = ET.SubElement(bndbox, 'bndbox')
                ET.SubElement(bndbox_elem, 'xmin').text = str(x)
                ET.SubElement(bndbox_elem, 'ymin').text = str(y)
                ET.SubElement(bndbox_elem, 'xmax').text = str(x2)
                ET.SubElement(bndbox_elem, 'ymax').text = str(y2)

            tree = ET.ElementTree(xml_root)
            tree.write(os.path.join(xml_dir,t[:-4]+".xml"), encoding='utf-8', xml_declaration=True)
            
            
    def txt2coco(self):
        pass



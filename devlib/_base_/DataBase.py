'''
    the base class for data processing and analysis, which define the type of dataset, such as VOC, COCO and etc, the dircetory structure of specific dataset 
    and the method to fetch data and annotation
'''

import json
from typing import Union, Any
import copy
import numpy as np

class custom_dict(dict):
    def __setitem__(self, key, value):
        if key not in self:
            raise KeyError(f"Key '{key}' does not exist in the dictionary")
        super().__setitem__(key, value)

class NumpyDecoder(json.JSONDecoder):
    def decode(self, s, **kwargs):
        result = super().decode(s, **kwargs)
        return self._convert(result)

    def _convert(self, obj):
        if isinstance(obj, list):
            try:
                return np.array(obj)
            except ValueError:
                return obj
        elif isinstance(obj, dict):
            return {key: self._convert(value) for key, value in obj.items()}
        return obj

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
    
class DatasetBase_ :
    def __init__(self, Cfg:Union[str, dict]) -> None:
        '''
            params:
                    Type: str->the type of dataset,("VOC","COCO") ; dict -> cfg dict of dataset's relavent path
                    custom: custom dataset , not recommending to use
        '''
        self.Cfg = Cfg
        self.voc_dict = {
                                'Image_Dir':"VOC2012/JPEGImages",
                                'Annotation_Dir':"VOC2012/Annotations",
                                'ImageSets_Dir':"VOC2012/ImageSets/Main",
                                'Annotation_Txt_Dir':"VOC2012/Annotations_Txt",
                                'TestSet_Path':"VOC2012/ImageSets/Main/test.txt",
                                'TrainSet_Path':"VOC2012/ImageSets/Main/trainval.txt"
                            }
        self.coco_dict = {
                                ''
            
                        }
        self.data_path_cfg={}
        if isinstance(self.Cfg,str):
            if self.Cfg=="VOC":
                self.data_path_cfg = self.voc_dict
        else:
            # TODO: need to check the vaildation of Cfg 
            
            self.data_path_cfg = Cfg

class DataConstructor:
    def __init__(self) -> None:
        
        self.imgdata_template = {
                'origin': None,
                'gray': None,
                'frequency': None,
                'HPF':None,
                'LPF':None
            }
        
        self.annotation_template = {'type': None, 'gt':None}
        
        self.numerical_data_template = {
                'r': None,
                'g': None,
                'b': None,
                'f':None,
            }
        
        self.datadict = {
            'source': None,
            'name': None,
            'annotation': self.annotation_template,
            'classes': None,
            'image': self.imgdata_template,
            'numerical': self.numerical_data_template
        }
    
    def _re_construct(self):
        self.datadict = {
            'source': None,
            'name': None,
            'annotation': self.annotation_template,
            'classes': None,
            'image': self.imgdata_template,
            'numerical': self.numerical_data_template
        }
    
    def get_item(self, source, name, classes, img:dict=None, annotation:dict=None, numerical:dict=None) -> dict:
        '''
        
        '''
        res = copy.deepcopy(self.datadict)
        res['classes'] = classes
        res['source'] = source
        res['name'] = name

        if img:
            for k,v in img.items():
                assert k in res['image'].keys() , f"img dict doesn't contain {k}, please re-construct or delete it"
                res['image'][k]=v
        if annotation:
            for k,v in annotation.items():
                assert  k in res['annotation'].keys(), f"annotation dict doesn't contain {k}, please re-construct or delete it"
                res['annotation'][k]=v
        if numerical:
            for k,v in numerical.items():
                assert  k in res['numerical'].keys(), f"numerical dict doesn't contain {k}, please re-construct or delete it"
                res['numerical'][k]=v
            
        return res
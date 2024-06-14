from typing import Union, Any
import numpy as np
import copy


class DataConstructure_:
    def __init__(self, boxencoder:str ) -> None:
        statistics_template = {'max': None, 'min': None, 'mean': None, 'std':None, 'histogram': None}
        imgdata_template = {
                'origin': None,
                'gray': None,
                'frequency': None,
                'HPF':None,
                'LPF':None
            }
        box_template = {'LTRBxyxy': None}
        
        self.datadict = {
            'source': None,
            'name': None,
            'annotation': box_template[boxencoder],
            'classes': None,
            'img': imgdata_template,
            'data': {
                'r': statistics_template,
                'g': statistics_template,
                'b': statistics_template,
                'fft_all': statistics_template
            },
        }
    
    def __call__(self, source: str, name: str, img: np.ndarray, annotation: Union[list, np.ndarray]) -> Any:
        '''
        img: RGB
        '''
        res = copy.deepcopy(self.datadict)
        res['img']['origin'] = img
        res['source'] = source
        res['name'] = name
        res['annotation'] = annotation[:-1]
        res['classes'] = annotation[-1]
        
        return res
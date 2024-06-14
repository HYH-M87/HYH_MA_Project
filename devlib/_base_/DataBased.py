'''
    the base class for data processing and analysis, which define the type of dataset, such as VOC, COCO and etc, the dircetory structure of specific dataset 
    and the method to fetch data and annotation
'''

from typing import Union, Any


class DataBase_ :
    def __init__(self, Type:Union[str, dict]) -> None:
        '''
            params:
                    Type: str->the type of dataset,("VOC","COCO") ; dict -> cfg dict of dataset's relavent path
                    custom: custom dataset , not recommending to use
        '''
        self.Type = Type
        if isinstance(self.Type,str):
            if self.Type=="VOC":
                self.Image_Path = "VOC2012/JPEGImages"
                self.Annotation_Path = "VOC2012/Annotations"
                self.Train_Data = "VOC2012/ImageSets/Main/trainval.txt"
                self.Test_Data = "VOC2012/ImageSets/Main/test.txt"
                self.ImageSets = "VOC2012/ImageSets/Main"
                self.Info = "VOC2012/info.txt"
                self.Annotation_Txt = "VOC2012/Annotations_Txt"
        else:
            self.Image_Path = Type['Image_Path']
            self.Annotation_Path = Type['Annotation_Path']
            self.Train_Data = Type['Train_Data']
            self.Test_Data = Type['Test_Data']
            self.ImageSets = Type['ImageSets']
            self.Info = Type['Info'] 
            self.Annotation_Txt = Type['Annotation_Txt']

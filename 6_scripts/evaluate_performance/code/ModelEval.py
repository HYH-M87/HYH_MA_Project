import cv2
import os
import numpy as np
from mmdet.apis import init_detector, inference_detector
import xml.etree.ElementTree as ET
from torch.utils.tensorboard.writer import SummaryWriter
import torch
from PIL import Image
import random

class ModelEval():
    
    def __init__(self, Type) -> None:
        self.Type = Type
        if Type=="VOC":
            self.Image_Path = "VOC2012/JPEGImages"
            self.Annotation_Path = "VOC2012/Annotations"
            self.Train_Data = "VOC2012/ImageSets/Main/trainval.txt"
            self.Test_Data = "VOC2012/ImageSets/Main/test.txt"
            self.ImageSets = "VOC2012/ImageSets/Main"
            self.Info = "VOC2012/info.txt"
            self.Annotation_Txt = "VOC2012/Annotations_Txt"
        
    
    def predict_batch(self, datasetdir:str, modelcfg:tuple, samplenum:int, logdir:str):
        '''
            random sampling on a set of datasets and make predictions. You have the option to save the results or visualize them through TensorBoard.
            param:
                datasetdir:
                modelcfg: a tuple contain the model file path and model cfg file  ("xxx.py","xxx.pth")
                samplenum: The number of samples
                logdir: the dir path to save tensorboard log. usually is the experiment log path. example: "9_logs/MA_Detection/hyh_ma_det_exp001"
        '''
        if not os.path.exists(os.path.join(logdir,"Negtive_Data")):
            os.mkdir(os.path.join(logdir,"Negtive_Data"))
            
        writer = SummaryWriter(os.path.join(logdir,"tb"))
        with open(os.path.join(datasetdir,self.Test_Data)) as f:
            image_files=f.read().splitlines()
        
        # image_files = [f for f in os.listdir(os.path.join(datasetdir,self.Test_Data))]
        selected_images = random.sample(image_files, k=samplenum)
        
        imagePaths = [os.path.join(datasetdir,self.Image_Path,i+".jpg") for i in selected_images]
        annotationPaths = [os.path.join(datasetdir,self.Annotation_Path,i+".xml") for i in selected_images]

        model = init_detector(modelcfg[0],modelcfg[1])
        

        for index,(img_file,xml_file) in enumerate(zip(imagePaths,annotationPaths)):
            
            
            img = cv2.imread(img_file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # get the gt box and draw
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for obj in root.iter('object'):
                bbox = obj.find('bndbox')
                gtx1 = int(bbox.find('xmin').text)
                gty1 = int(bbox.find('ymin').text)
                gtx2 = int(bbox.find('xmax').text)
                gty2 = int(bbox.find('ymax').text)
                img = cv2.rectangle(img,(gtx1,gty1),(gtx2,gty2),(255,0,0),1)
                
            # inference
            result = inference_detector(model,img)
            pred_box = result.pred_instances.bboxes
            pred_score = result.pred_instances.scores
            # draw predict box
            for i in range(pred_box.shape[0]):
                img = cv2.rectangle(img,(int(pred_box[i][0]),int(pred_box[i][1])),(int(pred_box[i][2]),int(pred_box[i][3])),(0,255,0),1)
            
            # save the negitive result
            if(len(pred_score)==0 or torch.min(pred_score)<0.7):
                writer.add_image("Negtive",img,global_step=index,dataformats="HWC")
                writer.add_text("Negtive",img_file,global_step=index)
                cv2.imwrite(os.path.join(logdir,"Negtive_Data",img_file),cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            # visualize the result
            writer.add_image(modelcfg[1], img, global_step=index,dataformats="HWC") 
            writer.add_text(modelcfg[1]+"txt",img_file,global_step=index)
                
        writer.close()
        
    

if __name__ =="__main__":
    # For Debug
    img_dir = "../Data/e_optha_MA/ProcessedData/MAimages_CutPatch(112,112)_overlap70.0"
    model = ["9_logs/MA_Detection/hyh_ma_det_exp001/run.py","9_logs/MA_Detection/hyh_ma_det_exp001/epoch_58.pth"]
    log_dri = "9_logs/MA_Detection/hyh_ma_det_exp001"
    eval = ModelEval("VOC")
    eval.predict_batch(img_dir, model, 5, log_dri)
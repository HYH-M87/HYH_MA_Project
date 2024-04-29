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
    
    def __init__(self) -> None:
        pass
    
    def predict_batch(self, images:str, annotations:str , modelcfg:tuple, samplenum:int, logdir:str):
        '''
            random sampling on a set of datasets and make predictions. You have the option to save the results or visualize them through TensorBoard.
            param:
                images: the directory of the images
                annotations: the directory of the annotations
                modelcfg: a tuple contain the model file path and model cfg file  ("xxx.py","xxx.pth")
                samplenum: The number of samples
                logdir: the dir path to save tensorboard log. usually is the experiment log path. example: "9_logs/MA_Detection/hyh_ma_det_exp001"
        '''
        writer = SummaryWriter(logdir)
        
        image_files = [f for f in os.listdir(images)]
        selected_images = random.sample(image_files, k=samplenum)
        annotation_files = [a for a in os.listdir(annotations) if a[:-4]+".jpg" in selected_images]
        print(modelcfg)
        model = init_detector(modelcfg[0],modelcfg[1])
        
        selected_images.sort()
        annotation_files.sort()
        for index,(img_file,xml_file) in enumerate(zip(selected_images,annotation_files)):
            
            
            img = cv2.imread(os.path.join(images,img_file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            tree = ET.parse(os.path.join(annotations,xml_file))
            root = tree.getroot()
            for obj in root.iter('object'):
                bbox = obj.find('bndbox')
                gtx1 = int(bbox.find('xmin').text)
                gty1 = int(bbox.find('ymin').text)
                gtx2 = int(bbox.find('xmax').text)
                gty2 = int(bbox.find('ymax').text)
                img = cv2.rectangle(img,(gtx1,gty1),(gtx2,gty2),(255,0,0),1)
            
            result = inference_detector(model,img)
            pred_box = result.pred_instances.bboxes
            pred_score = result.pred_instances.scores
            for i in range(pred_box.shape[0]):
                img = cv2.rectangle(img,(int(pred_box[i][0]),int(pred_box[i][1])),(int(pred_box[i][2]),int(pred_box[i][3])),(0,255,0),1)
                
            if(len(pred_score)==0 or torch.max(pred_score)<0.7):
                writer.add_image("Negtive",img,global_step=index,dataformats="HWC")
                
            writer.add_image(modelcfg[1], img, global_step=index,dataformats="HWC") 
            writer.add_text(modelcfg[1]+"txt",img_file,global_step=index)
                
        writer.close()

if __name__ =="__main__":
    # For Debug
    img_dir = "../Data/e_optha_MA/ProcessedData/MAimages_CutPatch(112,112)_overlap70.0/VOC2012/JPEGImages"
    label_dir = "../Data/e_optha_MA/ProcessedData/MAimages_CutPatch(112,112)_overlap70.0/VOC2012/Annotations"
    model = ["9_logs/MA_Detection/hyh_ma_det_exp001/run.py","9_logs/MA_Detection/hyh_ma_det_exp001/epoch_58.pth"]
    log_dri = "9_logs/MA_Detection/hyh_ma_det_exp001/tb"
    eval = ModelEval()
    eval.predict_batch(img_dir, label_dir, model, 5, log_dri)
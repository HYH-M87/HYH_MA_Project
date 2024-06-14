import cv2
import os
import numpy as np
from mmdet.apis import init_detector, inference_detector
import xml.etree.ElementTree as ET
from torch.utils.tensorboard.writer import SummaryWriter
import torch
from PIL import Image
import math
import random

class Metric():
    def __init__(self) -> None:
        
        pass
    

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
        
    def _init_predictor(self):
        '''
        初始化 模型，导入数据和标签，创建所需指标
        '''
        pass
        
    def _patch(self, patch):
        '''
        输入patch，进行预测，统计基于bbox的指标并更新
        
        '''
        
        pass
    
    def _image(self, image):
        '''
        
        '''
        pass
    
    def predict(self, dataset, model_cfg, samplenum, log):
        
        # load data
        
        # build model
        
        # create metric saver
        
        ## predict loop
            # predict a image
            # m,c = self._image()
        
        pass
    
    
    def predict_patch(self, datasetdir:str, modelcfg:tuple, samplenum:int, logdir:str):
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
    
    
    def predict_whole(self, datasetdir:tuple,  exp_dir:str, block_size:list, overlap:int, samplenum:int=-1, save:str=None, score_ther:float=0.3, iou_ther:float=0.5):
        '''
        1. 读入图像和标签
        2. 切片
        3. 预测
        4. 计算指标
        5. 保存
        
        '''

        assert os.path.exists(os.path.join(exp_dir,"latest","vis_data","config.py"))
        assert os.path.exists(os.path.join(exp_dir,"last_checkpoint"))
        
        with open(os.path.join(exp_dir,"last_checkpoint"),"r") as f:
            modelpath = f.readline()
        modelcfg=[]
        modelcfg.append(os.path.join(exp_dir,"latest","vis_data","config.py"))
        modelcfg.append(modelpath)

        step = [int(i*(1-overlap)) for i in block_size]
        print(f"Model Eval\nscore_threshold:{score_ther}\niou_threshold:{iou_ther}")

        img_dir = os.path.join(datasetdir,self.Image_Path)
        txt_dir = os.path.join(datasetdir,self.Annotation_Txt)
        dataset = os.path.join(datasetdir,self.Test_Data)
        
        with open(dataset,"r") as f:
            img_list = np.array(f.read().splitlines()).flatten()
        
        
        if samplenum > 0:
            img_list = img_list[0:samplenum]
        else:
            samplenum = len(img_list)
        
        # the metrics based on bbox, patch, image level respectively
        bbox_metric =  np.array([0,0,0,0,0,0,0]).astype(float) #(TP.FP.TM.FN,Recall,Ap,SP)
        patch_metric =  np.array([0,0,0,0,0,0,0]).astype(float)
        image_metric = np.array([0,0,0,0,0,0,0]).astype(float)
        
        healthy_img = 0
        ma_image = 0
        model = init_detector(modelcfg[0],modelcfg[1])
        
        # model inference for every image 
        for id,img in enumerate(img_list):
            print(f"all: {samplenum};  now:{id}")
            Ori_image = cv2.imread(os.path.join(img_dir,img+".jpg"))
            image = Ori_image.copy()
            
            healthy = not os.path.exists(os.path.join(txt_dir,img+".txt"))
            
            if not healthy:
                ma_image += 1
                with open(os.path.join(txt_dir,img+".txt"), 'r') as f:
                    data_line = np.array(f.readline().split()).astype(int)  
                
                bboxes = data_line.reshape((-1,5))
                bboxes = bboxes[bboxes[:,4]==0]
                bboxes[:,2:4] = bboxes[:,2:4] + bboxes[:,0:2]
            else:
                healthy_img += 1
                bboxes=[]
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, _ = image.shape
            num_blocks_height = math.ceil((height-block_size[1]) / step[1]) + 1 
            num_blocks_width = math.ceil((width-block_size[0]) / step[0]) + 1
            
            
            bbox_metric_ = np.array([0,0,0,0,0,0,0]).astype(float) #(TP.FP.TM.FN,Recall,Ap,SP)
            patch_metric_ = np.array([0,0,0,0,0,0,0]).astype(float)
            
            pred_boxes=[]
            hard_sample=[]
            hard_sample_iou=[]
            # the number of patch/block/'s GT is positive or negitive
            P_block=0.
            N_block=0.
            
            for i in range(num_blocks_height):
                for j in range(num_blocks_width):
                    metric_= np.array([0,0,0,0,0,0,0]).astype(float)
                    
                    # cut the block from image                    
                    y_start = i * step[1]
                    y_end = int(min(y_start + block_size[1], height))
                    x_start = j * step[0]
                    x_end = int(min(x_start + block_size[0], width))

                    block = image[y_start:y_end, x_start:x_end,:]

                    # discard the background area
                    if (block.flatten()<15).astype(int).sum()/len(block.flatten()) > 0.85:
                        continue
                    else:
                        # model inference
                        result = inference_detector(model,block)
                        pred_label = (result.pred_instances.labels).cpu().numpy()
                        pred_box = (result.pred_instances.bboxes).cpu().numpy().astype(np.uint32)
                        pred_score = (result.pred_instances.scores).cpu().numpy()
                        pred_box = pred_box[pred_score>score_ther] + np.array([x_start,y_start]*2).astype(np.uint32)
                        # get the GT box in current area
                        masked_gt = []
                        if not healthy:
                            masked_gt = bboxes[self.contain_(bboxes,[x_start,y_start,x_end,y_end])]

                        for b in pred_box:
                            pred_boxes.append(b)
                        
                        if(len(masked_gt)==0):
                            N_block += 1.
                            bbox_metric_[1] += len(pred_box)
                            if len(pred_score) == 0:
                                patch_metric_[2] += 1
                            else:
                                # if np.max(pred_score.flatten()) < 0.2:
                                #     patch_metric_[2] += 1
                                # else:
                                patch_metric_[1] += 1
                            continue
                        
                        P_block += 1.
                        if not healthy:
                            gtnum = len(masked_gt)
                            metric_[0:4], hard_sample_,hiou_ = self.fp_tp_bbox(masked_gt,pred_box,iou_ther)
                            hard_sample += hard_sample_
                            hard_sample_iou += hiou_
                            match_gt = gtnum-len(hard_sample_)
                            if metric_[0] != 0:
                                patch_metric_[0] += 1
                            else:
                                patch_metric_[3] += 1
                        
                        metric_[4] = match_gt/(gtnum+1e-9)  
                        metric_[5] = metric_[0]/(metric_[0]+metric_[1]+1e-9)
                        
                        bbox_metric_ += metric_
                        # print(f"block{num_block}  Recall={recall}  mAP={ap}  [TP,FP,TN,FN]={metric} gtnum={gtnum}  matchgt={match_gt}")
                    
            # for image
            bbox_metric_[4] = bbox_metric_[4] / (P_block+1e-9)
            bbox_metric_[5] = bbox_metric_[0] / (bbox_metric_[0]+bbox_metric_[1]+1e-9)
            bbox_metric += bbox_metric_
            
            patch_metric_[4] = patch_metric_[0] / (P_block+1e-9)
            patch_metric_[5] = patch_metric_[0] / (patch_metric_[0]+patch_metric_[1]+1e-9)
            patch_metric_[6] = patch_metric_[2] / (P_block+N_block+1e-9)
            patch_metric += patch_metric_
            
            if healthy:
                if patch_metric_[1] != 0:
                    image_metric[1] += 1
                    img_pt = "FP"
                else:
                    image_metric[2] += 1
                    img_pt = "TN"
            else:
                if patch_metric_[0] != 0:
                    image_metric[0] += 1
                    img_pt = "TP"
                else:
                    image_metric[3] += 1
                    img_pt = "FN"
                    
            offset=3
            for bbox in pred_boxes:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(Ori_image,(x1-offset,y1-offset),(x2+offset,y2+offset),(255,255,0),1)
            hard_sample_iou=np.array(hard_sample_iou)
            
            for bbox in bboxes:
                x1, y1, x2, y2, index= map(int, bbox)
                flag = (bbox==hard_sample).all(axis=1) if len(hard_sample)!=0 else [False]
                if np.any(flag):
                    cv2.rectangle(Ori_image,(x1-offset,y1-offset),(x2+offset,y2+offset),(255,12,215),1)
                    cv2.putText(Ori_image,f"iou:{round(hard_sample_iou[flag][0],2)}",(x2+10,y2+10),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 255, 255),1)
                else:
                    cv2.rectangle(Ori_image,(x1-offset,y1-offset),(x2+offset,y2+offset),(0,0,0),1)
                    
                cv2.rectangle(Ori_image,(x1-10,y1-10),(x2+10,y2+10),(255,0,0),2)

            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if save is not None:
    
                if not os.path.exists(os.path.join(exp_dir,save)):
                    os.makedirs(os.path.join(exp_dir,save))
                if not os.path.exists(os.path.join(exp_dir,save,"hard sample")):
                    os.makedirs(os.path.join(exp_dir,save,"hard sample"))
                    
                save_path = os.path.join(exp_dir,save,"pred_"+img+".jpg")
                cv2.imwrite(save_path,Ori_image)
                # with open(os.path.join(exp_dir,save,"eval.txt"),"w") as f:
                #     pass
                # with open(os.path.join(exp_dir,save,"pred_"+img+".txt"),"w+") as f:
                #     f.write(f"Bbox Based------------\nRecall={round(bbox_metric_[4],3)}  mAP={round(bbox_metric_[5],3)} [TP,FP,TN,FN]={bbox_metric_[0:4].astype(int)}\n")
                #     f.write(f"Patch Based-----------\nRecall={round(patch_metric_[4],3)}  mAP={round(patch_metric_[5],3)}  SP={patch_metric_[6]}  [TP,FP,TN,FN]={patch_metric_[0:4].astype(int)}\n")
                #     f.write("Image Based-----------\nGT={}\nPred={}".format("healthy" if healthy else "MA",img_pt))
                with open(os.path.join(exp_dir,save,"eval.txt"),"a+") as f:
                    f.write("*"*15+f"{img}"+"*"*15+"\n")
                    f.write(f"Bbox Based------------\n   Recall={round(bbox_metric_[4],3)}  mAP={round(bbox_metric_[5],3)} [TP,FP,TN,FN]={bbox_metric_[0:4].astype(int)}\n")
                    f.write(f"Patch Based-----------\n   Recall={round(patch_metric_[4],3)}  mAP={round(patch_metric_[5],3)}  SP={patch_metric_[6]}  [TP,FP,TN,FN]={patch_metric_[0:4].astype(int)}\n")
                    f.write("Image Based-----------\n   GT={}  Pred={}\n".format("healthy" if healthy else "MA",img_pt))
                    
                with open(os.path.join(exp_dir,save,"hard sample",f"{img}.txt"),"a+") as f:
                    for bbox in hard_sample:
                        x1, y1, x2, y2, index= map(int, bbox)
                        f.write(f"{x1} {y1} {x2} {y2} {index} ")
            # print(f"img:{img}  all block={N_block+P_block}")
            # print("bbox based------------")
            # print(f"Recall={bbox_metric_[4]}  mAP={bbox_metric_[5]} [TP,FP,TN,FN]={bbox_metric_[0:4]}")
            # print("patch based------------")
            # print(f"Recall={patch_metric_[4]}  mAP={patch_metric_[5]}  SP={patch_metric_[6]}  [TP,FP,TN,FN]={patch_metric_[0:4]}")
            # print("Image Based-----------\nGT={}\nPred={}".format("healthy" if healthy else "MA",img_pt))
        
        bbox_metric /= ma_image
        patch_metric /= ma_image
        
        # print("*"*15+"OVER ALL"+"*"*15)
        # print("bbox based------------")
        # print(f"Recall={bbox_metric[4]}  mAP={bbox_metric[5]} [TP,FP,TN,FN]={bbox_metric[0:4]}")
        # print("patch based------------")
        # print(f"Recall={patch_metric[4]}  mAP={patch_metric[5]}  SP={patch_metric[6]}  [TP,FP,TN,FN]={patch_metric[0:4]}")
        
        if save is not None:
            with open(os.path.join(exp_dir,save,"eval.txt"),"a+") as f:
                f.write("*"*15+"OVER ALL"+f"model : {modelcfg[1]} "+"*"*15+"\n")
                f.write(f"Bbox Based------------\n   Recall={round(bbox_metric[4],3)}  mAP={round(bbox_metric[5],3)} [TP,FP,TN,FN]={bbox_metric[0:4].astype(int)}\n")
                f.write(f"Patch Based-----------\n   Recall={round(patch_metric[4],3)}  mAP={round(patch_metric[5],3)}  SP={patch_metric[6]}  [TP,FP,TN,FN]={patch_metric[0:4].astype(int)}\n")
                f.write(f"Image Based-----------\n   Recall={round(image_metric[0] / ma_image,3)}  mAP={round(image_metric[0]/(image_metric[0]+image_metric[1]),3)}  SP={image_metric[2]/(image_metric[2]+image_metric[1])}  [TP,FP,TN,FN]={image_metric[0:4].astype(int)}")
        
    def contain_(self,box1,area):
        return ((box1[:,0]>=area[0]).astype(int) * (box1[:,1]>=area[1]).astype(int) * (box1[:,2]<=area[2]).astype(int) * (box1[:,3]<=area[3]).astype(int)).astype(bool)
    
    def IOU(self, box1, box2):
        '''
        params:
            box1 & box2 : (N*4) -> ([[x1,y1,x2,y2] ...])
        result:
            iou : N*1
        '''    
        iw = np.minimum(box1[:, 2], box2[:, 2]) - np.maximum(box1[:, 0], box2[:, 0])
        ih = np.minimum(box1[:, 3], box2[:, 3]) - np.maximum(box1[:, 1], box2[:, 1])
        iw = np.maximum(iw, 0)
        ih = np.maximum(ih, 0)
        intersection = iw * ih
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        union = area1 + area2 - intersection
        # iou = (intersection / union+1e-9)*2*(intersection/area1)
        iou = (intersection / union)
        return iou
    
    def DIOU(self, gt_boxes, pred_boxes):
        iou = self.IOU(gt_boxes, pred_boxes)
        # Calculate the center points of each box
        center_gt = np.column_stack(((gt_boxes[:, 0] + gt_boxes[:, 2]) / 2, (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2))
        center_pred = np.column_stack(((pred_boxes[:, 0] + pred_boxes[:, 2]) / 2, (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2))
        
        # Calculate the euclidean distance between the center points
        d = np.linalg.norm(center_gt - center_pred, axis=1)
        
        # Calculate the diagonal length of the smallest enclosing box
        x1_enclose = np.minimum(gt_boxes[:, 0], pred_boxes[:, 0])
        y1_enclose = np.minimum(gt_boxes[:, 1], pred_boxes[:, 1])
        x2_enclose = np.maximum(gt_boxes[:, 2], pred_boxes[:, 2])
        y2_enclose = np.maximum(gt_boxes[:, 3], pred_boxes[:, 3])
        c = np.linalg.norm(np.column_stack((x2_enclose, y2_enclose)) - np.column_stack((x1_enclose, y1_enclose)), axis=1)
        
        # Calculate DIoU
        diou = iou - (d ** 2) / (c ** 2)

        return diou
    
    def fp_tp_bbox(self,GT:np.array,Pred:np.array, iou_ther):
        TP = 0
        FP = 0
        TN = 0
        hard_sample = []
        hard_sample_iou = []
        for gtbox in GT:
            pl = Pred.shape[0]
            gtboxs = np.tile(gtbox, (pl, 1))
            iou = self.IOU(gtboxs,Pred)
            Nmatch_mask = (iou < iou_ther)
            TP += np.sum((iou >= iou_ther).astype(int))
            if(np.sum((iou >= iou_ther).astype(int))==0):
                hard_sample.append(gtbox)
                hard_sample_iou.append(np.max(iou) if len(iou)>0 else 0)
                
            Pred = Pred[Nmatch_mask]
            
        FP = Pred.shape[0]
        FN = len(hard_sample)
        # return [TP,FP,TN,FN]
        return [TP,FP,TN,FN],hard_sample,hard_sample_iou
    
    # def fp_tp_(self,GT:np.array,Pred:np.array):
    #     TP = 0
    #     FP = 0
    #     TN = 0
    #     iou_threshold = 0.5
    #     gt_matched = 0
    #     for gtbox in GT:
    #         pl = Pred.shape[0]
    #         gtboxs = np.tile(gtbox, (pl, 1))
    #         iou = self.IOU(gtboxs,Pred)
    #         Nmatch_mask = (iou < iou_threshold)
    #         TP += np.sum((iou >= iou_threshold).astype(int))
    #         if(np.sum((iou >= iou_threshold).astype(int))!=0):
    #             gt_matched += 1
                
    #         Pred = Pred[Nmatch_mask]
            
    #     FP = Pred.shape[0]
    #     FN = GT.shape[0] - gt_matched
    #     # return [TP,FP,TN,FN]
    #     return [TP,FP,TN,FN],gt_matched
    
    def recall(metric):
        pass
    
    def ap():
        pass
    
    
    

if __name__ =="__main__":
    
    # For Debug
    # img_dir = "../Data/e_optha_MA/ProcessedData/MAimages_CutPatch(112,112)_overlap70.0"
    # model = ["9_logs/MA_Detection/hyh_ma_det_exp001/run.py","9_logs/MA_Detection/hyh_ma_det_exp001/epoch_58.pth"]
    # log_dri = "9_logs/MA_Detection/hyh_ma_det_exp001"
    # eval = ModelEval("VOC")
    # eval.predict_batch(img_dir, model, 5, log_dri)
    
        
        
    # For Debug 
    # TODO: wait for test
    s=0.2
    i=0.2
    tool = ModelEval("VOC")
    data="../Data/e_optha_MA/MA"
    exp="9_logs/MA_Detection/hyh_ma_det_exp005/merge56to112_res50/pretrain"
    save=f"score{s}_iou{i}"
    tool.predict_whole(data,exp,[112,112],0.1,-1,save,s,i)

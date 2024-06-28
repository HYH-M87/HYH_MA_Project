import json
import sys
import cv2
import os
import numpy as np
from mmdet.apis import init_detector, inference_detector
import xml.etree.ElementTree as ET
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from .._base_ import DataConstructor
from ..process_data import GetImagePatch
from .._base_ import DatasetConstructor, NumpyEncoder, NumpyDecoder

class eval_data(DataConstructor):
    def __init__(self) -> None:
        super().__init__()
        self.annotation_template = {'type':None, 'gt':None, 'pred':None, 'hard_sample':None, 'score':None, 'label':None, 'iou':None, 'offset':None}
        self.imgdata_template = {'origin':None}
        self.numerical_data_template=None
        self._re_construct()
        
    
class Metric():
    def __init__(self) -> None:
        self.bbox_metric =  np.array([0,0,0,0,0,0,0]).astype(float) #(TP.FP.TM.FN,Recall,Ap,SP)
        self.patch_metric =  np.array([0,0,0,0,0,0,0]).astype(float)
        self.image_metric = np.array([0,0,0,0,0,0,0]).astype(float)
        
    

class ModelEval():
    
    def __init__(self, type, dataset_cfg, iou_threshold, score_threshold, save_dir, checkpoint=None, model_cfg=None, pred_result:str=None) -> None:
        assert type in ['load','model'], "type must be 'model' or 'load' "
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.gip = GetImagePatch(datastructure=eval_data)
        self.dataset_cfg = dataset_cfg
        self.dataset_constructor = DatasetConstructor('VOC', Annotation_Txt_Dir='VOC2012/Annotations_Txt') 
        self.data_path = dataset_cfg['dst_path']
        self.pred_result = pred_result
        self.bbox_metric =  np.array([0,0,0,0,0,0,0]).astype(float) #(TP.FP.TM.FN,Recall,Ap,SP)
        self.patch_metric =  np.array([0,0,0,0,0,0,0]).astype(float)
        self.image_metric = np.array([0,0,0,0,0,0,0]).astype(float)
        
        if type=='model':
            assert (checkpoint) and (model_cfg), 'type model must required checkpoint and model_cfg'
            self.model = self._init_predictor(model_cfg, checkpoint)
            self.get_pred = self.model_inference
        else:
            assert pred_result, 'type load must required pred_result'
            self.get_pred = self.load_pred
            
    def _init_predictor(self, cfg, checkpoint):
        '''
        初始化模型
        '''
        model = init_detector(cfg, checkpoint)
        return model
    
    def black_filter(self, *args, **kwargs):
        img = kwargs['patch_img']
        
        return not (np.sum(img.flatten()<15) / len(img.flatten()) > 0.85)
    
    
    def fp_tp_bbox(self, GT:np.array, Pred:np.array):

        hard_sample = []
        pred_iou = np.zeros((Pred.shape[0],2))
        pred_iou[:,1] = -1
        for index, gtbox in enumerate(GT):
            gtbox = gtbox[0:4]
            pl = Pred.shape[0]
            gtboxs = np.tile(gtbox, (pl, 1))
            iou = self.IOU(gtboxs,Pred)
            
            mask = iou > pred_iou[:,0]
            
            if np.any(mask):
                pred_iou[mask, 0] = iou[mask]
                pred_iou[mask, 1] = index
                if np.max(iou[mask]) < self.iou_threshold:
                    max_index = np.argmax(iou)
                    hard_sample.append([gtbox, np.hstack((Pred[max_index], iou[max_index])) ])
            else:
                if len(Pred) == 0:
                    hard_sample.append([gtbox, np.full_like(gtbox, -1) ])
                else:
                    max_index = np.argmax(iou)
                    hard_sample.append([gtbox, np.hstack((Pred[max_index], iou[max_index])) ])
                
        return pred_iou, hard_sample
    
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
    
    def model_inference(self, patch):
        img = patch['image']['origin']
        result = inference_detector(self.model, img)
        pred_label = (result.pred_instances.labels).cpu().numpy()
        pred_box = (result.pred_instances.bboxes).cpu().numpy().astype(np.uint32)
        pred_score = (result.pred_instances.scores).cpu().numpy()
        score_mask = pred_score>self.score_threshold
        pred_box = pred_box[score_mask]
        pred_score = pred_score[score_mask]
        pred_label = pred_label[score_mask]
        return pred_label, pred_box, pred_score

    def load_pred(self, patch):
        
        json_path = os.path.join(self.pred_result,  patch['name']+'.json')
        with open(json_path, 'r') as f:
            data = json.load(f, cls=NumpyDecoder)
        pred_label = data['annotation']['label']
        pred_box = data['annotation']['pred']
        pred_score = data['annotation']['score']
        pred_box.resize([pred_box.shape[0],4])
        
        return pred_label, pred_box, pred_score
    


    def cal_bbox_metric(self, patch):
        # 当前patch不含ma目标
        if len(patch['annotation']['gt']) == 0:
            TP = 0
            FP = len(patch['annotation']['pred'])
            TN = 0
            FN = 0
        else:
            gt = patch['annotation']['gt']
            pred = patch['annotation']['pred']
            pred_iou, hard_sample = self.fp_tp_bbox(gt, pred)
            patch['annotation']['hard_sample'] = hard_sample
            patch['annotation']['iou'] = pred_iou
            TP = np.sum(pred_iou[:,0] >= self.iou_threshold)
            FP = np.sum(pred_iou[:,0] < self.iou_threshold)
            TN = 0
            FN = len(hard_sample)
        
        return TP,FP,TN,FN

    def _patch(self, patch):
        '''
        输入patch，进行预测，统计基于bbox的指标并更新'''
        
        Recall = 0
        Ap = 0
        Sp = 0
        
        
        pred_label, pred_box, pred_score = self.get_pred(patch)
        patch['annotation']['pred'] = pred_box
        patch['annotation']['score'] = pred_score
        patch['annotation']['label'] = pred_label
        TP,FP,TN,FN = self.cal_bbox_metric(patch)
        
        gt_num = len(patch['annotation']['gt'])
        if gt_num > 0:

            Recall = (gt_num - FN)/(gt_num+1e-9)
        
        Ap = TP/(TP+FP+1e-9)
        Sp = TN / (FP+TN+1e-9)
        
        bbox_metric = np.array([TP,FP,TN,FN,Recall,Ap,Sp])
        
        
        return bbox_metric

    def _image(self, patch_lists):
        bbox_metric =  np.array([0,0,0,0,0,0,0]).astype(float)
        TP=0
        FP=0
        TN=0
        FN=0
        P_patch = 0
        N_patch = 0
        for p in patch_lists:
            bbox_metric_ = self._patch(p)
            
            if len(p['annotation']['gt']) > 0:
                P_patch += 1
                if bbox_metric_[0] != 0:
                    TP += 1
                else:
                    FN += 1
            else:
                N_patch += 1
                if  bbox_metric_[1] > 2:
                    FP += 1
                else:
                    TN += 1
            bbox_metric += bbox_metric_
        
        bbox_metric[4] /= (P_patch+1e-9)
        bbox_metric[5] /= (P_patch+N_patch+1e-9)
        bbox_metric[6] /= (P_patch+N_patch+1e-9)
        
        Recall = TP / (P_patch+1e-9)
        Ap = TP / (TP+FP+1e-9)
        Sp = TN / (N_patch+1e-9)
        
        patch_metric =  np.array([TP,FP,TN,FN,Recall,Ap,Sp]).astype(float)
        return np.around(patch_metric,3), np.around(bbox_metric,3)
    
    def save_img(self, path, image, pred_boxes, gt_boxes):
        '''
        画框
        '''
        offset = 4
        for (x,y,x2,y2) in pred_boxes:
            cv2.rectangle(image,(x-offset,y-offset), (x2+offset,y2+offset), (0,255,0),1)
        for (x,y,x2,y2) in gt_boxes:
            cv2.rectangle(image,(x-offset,y-offset), (x2+offset,y2+offset), (255,0,0),2)
        cv2.imwrite(path, image)

    def save_instance(self, instances):
        source = instances[0]['source']
        img_path = os.path.join(self.data_path['Image_Dir'],source)
        img_save_path = os.path.join(self.save_dir, source)
        
        log_path = os.path.join(self.save_dir, 'model_eval.log')
        img = cv2.imread(img_path)
        pred_boxes = []
        gt_boxes = []
        json_dir = os.path.join(self.save_dir, "result")
        if not os.path.exists(json_dir):
            os.makedirs(json_dir)
        for i in instances:
            json_path = os.path.join(json_dir,i['name']+'.json')
            i['image'] = self.dataset_cfg['patch_size']
            self.save_json(json_path, i)
            
            
            pred_boxes.extend(i['annotation']['pred'] + i['annotation']['offset'])
            if len(i['annotation']['gt']) > 0:
                gt_boxes.extend(i['annotation']['gt'][:,0:4] + i['annotation']['offset'])
        
        self.save_img(img_save_path, img, pred_boxes, gt_boxes)
        
        
    
    def save_json(self, path, dict):
        with open(path, 'w') as json_file:
            json.dump(dict, json_file,cls=NumpyEncoder)
    
    def predict(self):
        
        with open(self.data_path['TestSet_Path'],"r") as f:
            test_data = f.read().splitlines()
        
        img_list=[]
        annotation_list=[]
        for i in test_data:
            img_list.append(os.path.join(self.data_path['Image_Dir'],i+'.jpg'))
            annotation_list.append(os.path.join(self.data_path['Annotation_Txt_Dir'],i+".txt"))
            
        # img_list=img_list[9:11]
        # annotation_list=annotation_list[9:11]
        bbox_metric =  np.array([0,0,0,0,0,0,0]).astype(float)
        patch_metric =  np.array([0,0,0,0,0,0,0]).astype(float)
        TP=0
        FP=0
        TN=0
        FN=0
        P_image=0
        N_image=0
        
        for index,(ip,ap) in tqdm(enumerate(zip(img_list,annotation_list)), colour='green'):
            # #test:
            # if index > 3:
            #     break
            
            # 图片裁剪
        
            if not os.path.exists(ap):
                ap = None
            patch_lists = self.gip.cut_patch(ip, ap, self.dataset_cfg['patch_size'], self.dataset_cfg['overlap'] , [[self.black_filter,()]])
            
            patch_metric_, bbox_metric_ = self._image(patch_lists)
            
            if os.path.exists(ap):
                P_image += 1
                if patch_metric_[0] != 0:
                    TP += 1
                else:
                    FN += 1
            else:
                N_image += 1
                if patch_metric_[1] > 0:
                    FP += 1
                else:
                    TN += 1
            
            patch_metric += patch_metric_
            bbox_metric += bbox_metric_
            with open(os.path.join(self.save_dir,'model_eval.txt'),'a') as f:
                f.write(os.path.split(ip)[-1]+'\n')
                f.write('Bbox based: TP:{} FP:{} TN:{} FN:{} Recall:{} AP:{} SP:{} \n'.format(*tuple(bbox_metric_)))
                f.write('Patch based: TP:{} FP:{} TN:{} FN:{} Recall:{} AP:{} SP:{} \n'.format(*tuple(patch_metric_)))
                
            self.save_instance(patch_lists)
            
            
        Recall = TP / P_image+1e-9
        Ap = TP / (TP+FP+1e-9)
        Sp = TN / (N_image+1e-9)
        image_metric = np.around(np.array([TP,FP,TN,FN,Recall,Ap,Sp]).astype(float),3)
        bbox_metric /= (P_image+1e-9)
        patch_metric /= (P_image+1e-9)
        
        with open(os.path.join(self.save_dir,'model_eval.txt'),'a') as f:
                f.write('Summaries'+'\n')
                f.write('Bbox based: TP:{} FP:{} TN:{} FN:{} Recall:{} AP:{} SP:{} \n'.format(*tuple(bbox_metric)))
                f.write('Patch based: TP:{} FP:{} TN:{} FN:{} Recall:{} AP:{} SP:{} \n'.format(*tuple(patch_metric)))
                f.write('Image based: TP:{} FP:{} TN:{} FN:{} Recall:{} AP:{} SP:{} \n'.format(*tuple(image_metric)))

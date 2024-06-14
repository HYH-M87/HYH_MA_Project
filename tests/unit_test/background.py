

import cv2
import numpy as np
import os
import random
dir = "MA_healthy/VOC2012/JPEGImages"
img_list = os.listdir(dir)
for i in img_list:
    


    image = cv2.imread('MA/VOC2012/JPEGImages/'+i, 0) 
    #MA 二进制掩膜
    an_mask = cv2.imread("Annotation_MA/"+i[:-4]+".png")[:,:,0]
    # 生成MA目标框
    box_num, labels, boxes, centroids = cv2.connectedComponentsWithStats(an_mask, connectivity=8)
    boxes = (boxes[boxes[:,4].argsort()])[:-1]
    boxes[:,4] = 0
    gtnum = len(boxes)
    allbox = boxes
    # canny 边缘检测
    edges = cv2.Canny(image, 7, 50)
    binary_mask = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY)[1]
    # 十字腐蚀
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    fs = cv2.erode(binary_mask, kernel, iterations=1)
    
    # 在背景类候选中除去MA
    an_mask = cv2.dilate(an_mask, np.ones((9, 9), dtype=np.uint8), 1)
    an_mask[an_mask<0]=0
    an_mask[an_mask>0]=255
    fs = fs.astype(float) - an_mask.astype(float) 
    fs[fs<0]=0
    fs=fs.astype(np.uint8)
    # 将背景类目标膨胀至与MA大小近似
    fs = cv2.dilate(fs, cv2.getStructuringElement(cv2.MORPH_ELLIPSE , (3, 3)), 1 )
    fs = cv2.dilate(fs, cv2.getStructuringElement(cv2.MORPH_RECT , (5, 5)), 1 )
    # cv2.imshow("d_fs",fs+an_mask)
    # cv2.imshow("fs",fs)
    # cv2.waitKey(0)
    
    #计算背景类目标框
    box_num, labels, boxes, centroids = cv2.connectedComponentsWithStats(fs, connectivity=8)
    boxes = (boxes[boxes[:,4].argsort()])[:-1]
    boxes[:,4] = 1
    boxes = boxes[np.random.choice(boxes.shape[0], min(gtnum,len(boxes)), replace=False)]
    allbox = np.vstack((allbox,boxes))
    
    # 保存
    with open(os.path.join("MA _healthy/VOC2012/Annotations_Txt",i[:-4])+".txt","w") as f:
        for b in allbox:
            b = [str(i)+" " for i in b]
            f.writelines(b)
    
    


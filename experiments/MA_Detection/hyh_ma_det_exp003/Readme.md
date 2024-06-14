# Experiment : hyh_ma_det_exp001
|  Id   | Date  | Author |
|  ----  | ----  | ---   |
| 001  | 24/04/30 |  Hyh  |

## Introduction
### Brief
探究图像形态学操作对模型性能的影响
### Motivation
在阅读MA检测相关论文中，存在大量基于人工筛选特征和ML算法的MA检测，以及部分基于DL的也有做相关图像预处理，故进行尝试
### Procedure
先对整张图像做了均值滤波，基于对所有gtbox面积的均值（47.144）故设置kernel size为（7\*7）。接着运用自适应直方图均衡对图像对比度进行增强。
输入patch尺寸为（112\*112），resize为(224\*224)，overlap rate 50%
## Configuration 

### Dataset
e_optha_MA
### Model
FSAF
### Optimizor
SGD with momentum(0.9)
LR : 0.01
LR decay: k=0.1 , step=[8,12]
### Loss Function
IOU Loss
### Training Parameter
batch size = 32
training epoch = 50
CenterRegionAssigner:  pos_scale=0.8, neg_scale=0.8

## Conclusion
|Recall	|Map|
|---|---|
|0.733	|0.702|
实验结果并不理想，这种处理确实在一定程度上删除了很多特征，可能是影响模型性能的原因，有个想法就是保存原图像的RGB三通道，并额外添加这些特征，需要修改模型
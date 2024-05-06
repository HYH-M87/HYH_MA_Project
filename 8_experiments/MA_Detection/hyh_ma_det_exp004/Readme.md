# Experiment : hyh_ma_det_exp001
|  Id   | Date  | Author |
|  ----  | ----  | ---   |
| 001  | 24/05/01 |  Hyh  |

## Introduction
### Brief
探究通过将多个含有MA点的patch进行拼接，形成一个(112\*112)的图像，对比于(112\*112)的patch其包含更多的MA目标点是否对模型性能有所提升
### Motivation
在对小目标检测的论文中对数据的处理提及到可以将对个图像进行拼接以提高目标点的数量，有助于小目标点的检测
### Procedure
对原始图像进行(56\*56)patch的裁剪，并采用四个小patch进行拼接成为一张(112\*112)的图像，resize to (224\*224)，overlap rate 50%
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
|0.904	|0.866|
实验结果算是比较成功，但是在进行整理时发现，可能该结果并不正确，因为处理图像的过程不严谨，在测试集中经过merge的图像可能有重重复。
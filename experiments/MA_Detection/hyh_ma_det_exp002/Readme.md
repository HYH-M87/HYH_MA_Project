# Experiment : hyh_ma_det_exp001
|  Id   | Date  | Author |
|  ----  | ----  | ---   |
| 001  | 24/04/30 |  Hyh  |

## Introduction
### Brief
探究CenterRegionAssigner的参数对recall和map的影响，主要是pos_scale 和 neg_scale；
pos_scale是一个0~1的参数，表示的是在gtbox中心向外的这个pos_scale范围内，若bbox的中心落入其中则分配为正样本
### Motivation
向通过调整这个参数实现recall的提升，按理说pos_scale越大，则recall相应会越高
### Procedure
输入patch尺寸为（112\*112），resize to (224\*2224)，overlap rate 50%
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

## Conclusion
pos_scale & neg_scale|Recall	|Map|
|---|---|---|
|0.8 & 0.8| 0.873  | 0.818 |
|0.5 & 0.5| 0.867  | 0.812 |
更大的pos_scale和neg_scale对模型性能有略微提升，但不明显
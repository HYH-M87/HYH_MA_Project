# Experiment : hyh_ma_det_exp001
|  Id   | Date  | Author |
|  ----  | ----  | ---   |
| 001  | 24/04/28 |  Hyh  |

## Introduction
### Brief
探究对输入图像后的resize处理，resize为两种尺寸 1. (112\*112) 2. (224\*224) 和不同的overlap rate 1. 50% 2. 70%对模型性能的影响
### Motivation
在一次配置训练参数时，发现我输入的图像是(112\*112)的patch，但是经过resize后为(224\*224)，因此取消掉resize的操作，但发现模型性能有明显下降，故做此实验，根据overlap——rate具体分为两大部分，以随便观察overlap——rate的影响
### Procedure
输入patch尺寸为（112\*112），overlap rate 50%和70%以及resize尺寸(112\*112),(224\*224)进行四组实验
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
|index	|input size|	data processing-resize|	overlap rate|	Recall|	Map|
|  ----  | ----  | ---   |  ----  | ----  | ---   |
|1	|112*112	|112*112	|70%	|0.728	|0.696|
|2	|112*112	|224*224|	70%	|0.862	|0.715|
|3	|112*112	|112*112|	50%|	0.689|	0.636|
|4	|112*112	|224*224	|50%	|0.875	|0.832|
|5	|224*224	|224*224|	50%	|0.714	|0.684|

经实验结果发现，对于实验4，5可知对于输入尺寸为112*112的模型其性能更好，并根据实验1和2，以及实验3和4可知对输入的patch resize为（224*224）后模型性能有提升，但以上实验对于overlap rate并没有明显的结果。已知目前实验4的效果最佳，因此后续对数据的实验将沿用这套参数
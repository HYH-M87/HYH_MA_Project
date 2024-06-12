
# Qucik Start

本项目基于MMdetection框架，进行眼底彩照的微动脉瘤（MA）检测

---


## 如何使用
完整的训练过程如下：
    1. 构建数据集
    2. 编写配置文件
    3. 开始训练  

**数据集构建**：目前支持VOC格式的数据集，若所用数据集为非VOC格式，需要进行转换。数据处理的python脚本位于：**"6_scripts/process_data/code"** 目录下；可直接使用的bash脚本位于 **"6_scripts/process_data/utils"** 目录下

1. 图片的切片：将一张完整的眼底彩照图片进行切片处理。

```python
cut_patch.sh 用于将原始图片切片
参数：
    TYPE #数据集类型（目前只支持VOC格式）
    ORI_DATA #原始数据集路径（数据集的最高级目录路径）
    PROCESSED_DATA #处理后数据集的保存路径
    PATCH_SIZE_H #切片大小的高
    PATCH_SIZE_W #切片大小的宽
    OVERLAP_RATE #切片间的重叠率(0~1)
    SPLIT #训练集划分的系数（0~1），可选参数，默认为0.8
    DESCRIPE #该数据集的描述信息

usage：
    bash 6_scripts/process_data/utils/cut_patch.sh "VOC" ../Data/e_optha_MA/MA ../Data/e_optha_MA/ProcessedData 112 112 0.5 0.8 nan

```
2. 图片的切片并进行拼接：将一张完整的眼底彩照图片进行切片处理再讲切片进行拼接。目前没有设置拼接后的size参数的接口，默认拼接位112*112，如需修改可到../code/merge_patch.py中进行修改

```python
merge_patch.sh 用于将原始图片切片
参数：
    TYPE #数据集类型（目前只支持VOC格式）
    ORI_DATA #原始数据集路径（数据集的最高级目录路径）
    PROCESSED_DATA #处理后数据集的保存路径
    PATCH_SIZE_H #切片大小的高
    PATCH_SIZE_W #切片大小的宽
    OVERLAP_RATE #切片间的重叠率(0~1)
    SPLIT #训练集划分的系数（0~1），可选参数，默认为0.8
    DESCRIPE #该数据集的描述信息

usage：
    bash 6_scripts/process_data/utils/merge_patch.sh "VOC" ../Data/e_optha_MA/MA ../Data/e_optha_MA/ProcessedData 56 56 0.5 0.8 nan

```

**实验配置文件**：项目基于MMdetection故，实验配置文件与mmdet一致。实验目录如下
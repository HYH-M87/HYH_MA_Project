# MA-Project

# Data

The datasets are all stored in the "data" directory. In addition to the dataset files, there are also some bash scripts for data processing. All datasets involved in the processing should first be converted into the specified dataset format, like "VOC","COCO".
## Bash Scripts
**1.cut_patch.sh**

The script calls the Python script in "project2_MA/6_scripts/process_data" to cut patch on images. The parameter list is as follows:
```
TYPE: Dataset formats, such as 'VOC' and 'COCO'

ORI_DATA: the top-level directory of the original images，Example for "VOC" the top-level directory is "VOCdevkit"

PROCESSED_DATA: The directory for storing processed datasets, which contain a series of datasets in a specified format within its subdirectories

PATCH_SIZE_H: the height value of a patch size

PATCH_SIZE_W: the width value of a patch size

OVERLAP_RATE: the overlap rate between two patches
```
```
usage: bash cut_patch.sh "VOC" "e_optha_MA/MA" "e_optha_MA/ProcessedData” 112 112 0.7 
```

# project2_MA
The directory for the main project of MA detection, for detailed information, see the 'documentation' directory

# How to do Data Processing
The directory 6_scripts/process_data contains materials related to data processing, which includes two subdirectories: code and utils. The code directory stores Python scripts, while the utils directory contains bash scripts.
## code

Within the "code" directory, there is a script named DataProcess.py that implements a DataProcess class. This class integrates multiple functions related to data processing. The remaining scripts are designed to utilize the DataProcess class to perform more specific data processing tasks.

## utils
The utils directory contains a series of bash scripts that invoke Python scripts located in the code directory to facilitate the execution of related operations.

## usage of bash scripts

### 1.cut_patch.sh

The script calls the Python script in "6_scripts/process_data/code/cut_patch_overlap.py" to cut patch on images. The parameter list is as follows:
```
TYPE: Dataset formats, such as 'VOC' and 'COCO'

ORI_DATA: the top-level directory of the original images，Example for "VOC" the top-level directory is "VOCdevkit"

PROCESSED_DATA: The directory for storing processed datasets, which contain a series of datasets in a specified format within its subdirectories

PATCH_SIZE_H: the height value of a patch size

PATCH_SIZE_W: the width value of a patch size

OVERLAP_RATE: the overlap rate between two patches
```
```
usage: bash 6_scripts/process_data/utils/cut_patch.sh "VOC" "../Data/e_optha_MA/MA" "../Data/e_optha_MA/ProcessedData” 112 112 0.7 
```
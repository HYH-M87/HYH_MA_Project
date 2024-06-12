from DataProcess import DataProcess
import argparse
import os

'''
cut origin image into patch
'''



def parse_args():
    parser = argparse.ArgumentParser(description='Cut patch from Image')
    parser.add_argument('data_type', action='store', help='the type of dataset', choices=['COCO', 'VOC'])
    parser.add_argument('data_dir', action='store', help='the dir to save original dataset')
    parser.add_argument('out_dir', action='store', help='the dir to save Processed dataset')
    parser.add_argument(
        'patch_size',
        type=int,
        action='store',
        nargs='+',
        help='the size of a patch')
    # parser.add_argument(
    #     'target_size',
    #     type=int,
    #     action='store',
    #     nargs='+',
    #     help='the size of a target')
    parser.add_argument(
        'overlap_rate',
        type=float,
        action='store',
        help='the overlap rate between adjacent patches')
    parser.add_argument(
        'split',
        type=float,
        action='store',
        help='the number of the ouput image')
    parser.add_argument('--descripe', action='store', default="", help='the description of this dataset')
    args = parser.parse_args()
    
    return args

def main():
    args = parse_args()
    tool = DataProcess(args.data_type)
    
    block_size = args.patch_size
    overlap = args.overlap_rate
    target_size = [112,112]
    OriData_path = args.data_dir
    
    ProcessedData_path= os.path.join(args.out_dir, "MAimages_MergePatch({},{})_to_({},{})_overlap{}_{}".format(block_size[0],block_size[1],target_size[0],target_size[1],overlap*100,args.descripe))
    
    
    ori_image_path = os.path.join(OriData_path,tool.Image_Path)
    ori_label_path = os.path.join(OriData_path,tool.Annotation_Txt)
    dst_image_path = os.path.join(ProcessedData_path,tool.Image_Path)
    dst_label_path = os.path.join(ProcessedData_path,tool.Annotation_Txt)
    dst_set_path = (os.path.join(ProcessedData_path,tool.Train_Data),os.path.join(ProcessedData_path,tool.Test_Data))
    
    
    tool.make_dir(ProcessedData_path)
    
    # tool.image_patch_merge((ori_image_path,dst_image_path),(ori_label_path,dst_label_path),block_size,overlap,args.split,dst_set_path)
    tool.image_patch_merge_witin((ori_image_path,dst_image_path),(ori_label_path,dst_label_path),block_size,target_size,overlap,args.split,dst_set_path)

    tool.txt2xml(dst_label_path,os.path.join(ProcessedData_path,tool.Annotation_Path),("MA","Background"))
    tool.calculate_mean_variance(dst_image_path,os.path.join(ProcessedData_path,tool.Info))
    pass


if __name__ == "__main__":
    main()
    
    
    ## for debug:
    # tool = DataProcess("VOC")
    # block_size=[56,56]
    # target_size = [112,112]
    # overlap=0.5
    # OriData_path="../Data/e_optha_MA/MA_healthy"
    
    # ProcessedData_path= os.path.join("../Data/e_optha_MA/", "MAimages_MergePatch({},{})_overlap{}_{}".format(block_size[0],block_size[1],overlap*100,"withhealthy"))
    
    
    # ori_image_path = os.path.join(OriData_path,tool.Image_Path)
    # ori_label_path = os.path.join(OriData_path,tool.Annotation_Txt)
    # dst_image_path = os.path.join(ProcessedData_path,tool.Image_Path)
    # dst_label_path = os.path.join(ProcessedData_path,tool.Annotation_Txt)
    # dst_set_path = (os.path.join(ProcessedData_path,tool.Train_Data),os.path.join(ProcessedData_path,tool.Test_Data))
    
    
    # tool.make_dir(ProcessedData_path)
    # tool.image_patch_merge_witin((ori_image_path,dst_image_path),(ori_label_path,dst_label_path),block_size,target_size,overlap,0.8,dst_set_path)
    # tool.txt2xml(dst_label_path,os.path.join(ProcessedData_path,tool.Annotation_Path),("MA","Background"))
    # tool.calculate_mean_variance(dst_image_path,os.path.join(ProcessedData_path,tool.Info))
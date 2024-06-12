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
    parser.add_argument('patch_size', type=int, action='store', nargs='+', help='the size of a patch')
    parser.add_argument('overlap_rate', type=float, action='store', help='the overlap rate between adjacent patches')
    parser.add_argument(
        'split',
        type=float,
        action='store',
        help='the ratio of train dataset')
    parser.add_argument('--descripe', action='store', default="", help='the description of this dataset')
    args = parser.parse_args()
    
    return args

def main():
    args = parse_args()
    tool = DataProcess(args.data_type)
    
    block_size = args.patch_size
    overlap = args.overlap_rate
    
    OriData_path = args.data_dir
    
    ProcessedData_path= os.path.join(args.out_dir, "MAimages_CutPatch({},{})_overlap{}_{}".format(block_size[0],block_size[1],overlap*100,args.descripe))
    
    if_whole=False
    
    ori_image_path = os.path.join(OriData_path,tool.Image_Path)
    ori_label_path = os.path.join(OriData_path,tool.Annotation_Txt)
    dst_image_path = os.path.join(ProcessedData_path,tool.Image_Path)
    dst_label_path = os.path.join(ProcessedData_path,tool.Annotation_Txt)
    dataset_path = (os.path.join(ProcessedData_path,tool.Train_Data),os.path.join(ProcessedData_path,tool.Test_Data))
    
    tool.make_dir(ProcessedData_path)
    if args.split != -1:
        tool.image_cut((ori_image_path,dst_image_path),(ori_label_path,dst_label_path),block_size,overlap,args.split,dataset_path,if_whole)
    else:
        tool.image_cut((ori_image_path,dst_image_path),(ori_label_path,dst_label_path),block_size,overlap,None,None,if_whole)
        tool.data_split(dst_label_path,dataset_path)
    tool.txt2xml(dst_label_path,os.path.join(ProcessedData_path,tool.Annotation_Path),("MA","Background"))
    tool.calculate_mean_variance(dst_image_path,os.path.join(ProcessedData_path,tool.Info))


if __name__ == "__main__":
    main()
    
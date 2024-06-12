from ModelEval import ModelEval
import argparse


# images:str, annotations:str , modelcfg:tuple, samplenum:int, logname:str

def parse_args():
    parser = argparse.ArgumentParser(description='Predict a batch of image and visualize in Tensorboard')
    parser.add_argument('data_type', action='store', help='the type of dataset',choices=["VOC","COCO"])
    parser.add_argument('dataset_dir', action='store', help='the top-level directory of dataset ')
    parser.add_argument('exp_dir', action='store', help='the directory path of experiment')
    parser.add_argument('patch_size', type=int, action='store', nargs='+', help='the size of a patch')
    parser.add_argument('overlap_rate', type=float, action='store', help='the overlap rate between adjacent patches')
    parser.add_argument('sample_num', type=int, action='store', help='the number of a images in a batch')
    parser.add_argument('score_threshold', type=float, action='store', help='the threshold to screen out the predict box')
    parser.add_argument('iou_threshold', type=float, action='store', help='the threshold to choose the TP predict box')
    parser.add_argument('--descripe', action='store', default="", help='the description added to the directory name')
    args = parser.parse_args()
    
    return args

def main():
    
    args = parse_args()
    eval = ModelEval(args.data_type)
    healthy=True
    log_dir = "iou{}_score{}_{}".format(args.iou_threshold, args.score_threshold, args.descripe)
    eval.predict_whole(args.dataset_dir, args.exp_dir, args.patch_size, args.overlap_rate, args.sample_num, log_dir ,args.score_threshold,args.iou_threshold)

if __name__ =="__main__":
    main()
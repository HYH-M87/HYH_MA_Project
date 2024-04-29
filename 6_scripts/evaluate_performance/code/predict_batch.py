from ModelEval import ModelEval
import argparse


# images:str, annotations:str , modelcfg:tuple, samplenum:int, logname:str

def parse_args():
    parser = argparse.ArgumentParser(description='Predict a batch of image and visualize in Tensorboard')
    parser.add_argument('data_type', action='store', help='the type of dataset',choices=["VOC","COCO"])
    parser.add_argument('dataset_dir', action='store', help='the top-level directory of dataset ')
    parser.add_argument('model_weight', action='store', help='the path of model weight .pth file')
    parser.add_argument('model_cfg', action='store', help='the path of model cfg .py file')
    parser.add_argument('sample_num', type=int, action='store', help='the number of a images in a batch')
    parser.add_argument('log_dir', type=str, action='store', help='the directory of tensorboard log file')
    args = parser.parse_args()
    
    return args

def main():
    args = parse_args()
    eval = ModelEval(args.data_type)
    eval.predict_batch(args.dataset_dir, [args.model_cfg,args.model_weight], args.sample_num, args.log_dir)

if __name__ =="__main__":
    main()
from ModelEval import ModelEval
import argparse


# images:str, annotations:str , modelcfg:tuple, samplenum:int, logname:str

def parse_args():
    parser = argparse.ArgumentParser(description='Predict a batch of image and visualize in Tensorboard')
    parser.add_argument('img_dir', action='store', help='the directory of .jpg files')
    parser.add_argument('annotation_dir', action='store', help='the directory of annotation .xml files')
    parser.add_argument('model_weight', action='store', help='the path of model weight .pth file')
    parser.add_argument('model_cfg', action='store', help='the path of model cfg .py file')
    parser.add_argument('sample_num', type=int, action='store', help='the number of a images in a batch')
    parser.add_argument('log_dir', type=str, action='store', help='the directory of tensorboard log file')
    args = parser.parse_args()
    
    return args

def main():
    args = parse_args()
    eval = ModelEval()
    eval.predict_batch(args.img_dir, args.annotation_dir, [args.model_cfg,args.model_weight], args.sample_num, args.log_dir)

if __name__ =="__main__":
    main()
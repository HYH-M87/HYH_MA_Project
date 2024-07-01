from devlib.dataset import MAPatchDataset
from devlib.evaluate_performance import ModelEval


from devlib.dataset import MAPatchDataset
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Cut patch from Image')
    parser.add_argument('mode', action='store', help='the name of original dataset')
    parser.add_argument('pred', action='store', nargs='+', help='if mode is "model" it will receive two prarams, checkpoint and model cfg, respectively.')
    parser.add_argument('data_dir', action='store', help='the name of original dataset')
    parser.add_argument('patch_size', type=int, action='store', nargs=2, help='the size of a patch')
    parser.add_argument('overlap_rate', type=float, action='store', help='the overlap rate between adjacent patches')
    parser.add_argument('split', type=float, action='store', help='the ratio of train dataset')
    parser.add_argument('score', action='store', type=float, help='the score threshold')
    parser.add_argument('iou', action='store', type=float, help='the iou threshold')
    parser.add_argument('save', action='store', help='the directory to save Processed dataset, insted of the name of dataset')
    parser.add_argument('--descripe', action='store', default="", help='the description of this dataset')

    args = parser.parse_args()
    
    return args

def main():
    args = parse_args()
    for arg, value in vars(args).items():
        print(f"{arg}: {value} \n")
    mode = args.mode
    pred = args.pred
    patch_size = args.patch_size
    overlap = args.overlap_rate
    score_threshold = args.score
    iou_threshold = args.iou
    ma_patch = MAPatchDataset('None', args.data_dir, patch_size, overlap, "image", args.split)
    datacfg=ma_patch.get_cfg()
    name = f'model_eval_{iou_threshold}_{score_threshold}_{args.descripe}'
    save_dir = os.path.join(args.save,name)
    
    if mode == 'model':
        e = ModelEval(mode, datacfg, iou_threshold, score_threshold, save_dir, checkpoint=pred[0], model_cfg=pred[1])
    else:
        e = ModelEval(mode, datacfg, iou_threshold, score_threshold, save_dir, pred_result=pred[0])
    
    e.predict()



if __name__ == "__main__":
    main()
    # DEBUG
    # ma_patch = MAPatchDataset('None', '../Data/e_optha_MA/MA_ex', [112,112], 0.1, "image", 0.8)
    # datacfg=ma_patch.get_cfg()
    # e = ModelEval("load", datacfg, 0.2, 0.4, 'tests/unit_test/modeleval',pred_result='tests/unit_test/modeleval/result')
    # e.predict()
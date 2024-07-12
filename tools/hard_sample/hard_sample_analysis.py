from devlib.analyze_data import HardSampleAnalysis
from devlib.dataset import MAPatchDataset
from devlib.dataset import MAPatchDataset
from devlib.evaluate_performance import ModelEval


from devlib.dataset import MAPatchDataset
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Hardsample Analysis')
    parser.add_argument('data_dir', action='store', help='the name of original dataset')
    parser.add_argument('json_dir', action='store', help='the name of original dataset')
    parser.add_argument('patch_size', type=int, action='store', nargs=2, help='the size of a patch')
    parser.add_argument('overlap_rate', type=float, action='store', help='the overlap rate between adjacent patches')
    parser.add_argument('split', type=float, action='store', help='the ratio of train dataset')
    parser.add_argument('save', action='store', help='the directory to save Processed dataset, insted of the name of dataset')
    parser.add_argument('--descripe', action='store', default="", help='the description of this dataset')

    args = parser.parse_args()
    
    return args

def main():
    args = parse_args()
    for arg, value in vars(args).items():
        print(f"{arg}: {value} \n")

    patch_size = args.patch_size
    overlap = args.overlap_rate

    ma_patch = MAPatchDataset('None', args.data_dir, patch_size, overlap, "image", args.split)
    datacfg=ma_patch.get_cfg()
    save_dir = os.path.join(args.save,args.descripe)
    analysis = HardSampleAnalysis(datacfg, args.json_dir, save_dir)
    analysis.forward()



if __name__ == "__main__":
    # main()
    # DEBUG
    ma_patch = MAPatchDataset('None', '../Data/e_optha_MA/MA_ex', [112,112], 0.1, "image", 0.8)
    datacfg=ma_patch.get_cfg()
    analysis = HardSampleAnalysis(datacfg, "fortest/test/model_eval_0.2_0.4_test/result", "fortest/test")
    analysis.forward()
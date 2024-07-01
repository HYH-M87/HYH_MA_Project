from devlib.dataset import MAPatchDataset
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Cut patch from Image')
    parser.add_argument('data_dir', action='store', help='the name of original dataset')
    parser.add_argument('out_dir', action='store', help='the directory to save Processed dataset, insted of the name of dataset')
    parser.add_argument('patch_size', type=int, action='store', nargs=2, help='the size of a patch')
    parser.add_argument('overlap_rate', type=float, action='store', help='the overlap rate between adjacent patches')
    parser.add_argument('split', type=float, action='store', help='the ratio of train dataset')
    parser.add_argument('--descripe', action='store', default="", help='the description of this dataset')
    args = parser.parse_args()
    
    return args

def main():
    args = parse_args()
    for arg, value in vars(args).items():
        print(f"{arg}: {value} \n")
    patch_size = args.patch_size
    overlap = args.overlap_rate
    dataset_name = "MAimages_CutPatch({},{})_overlap{}_{}".format(patch_size[0],patch_size[1],overlap*100,args.descripe)
    ProcessedData_path= os.path.join(args.out_dir, dataset_name)
    ma_patch = MAPatchDataset(args.data_dir, ProcessedData_path, patch_size, overlap, "image", args.split)
    # ma_patch = MAPatchDataset("/home/hyh/Documents/quanyi/project/Data/e_optha_MA/MA","/home/hyh/Documents/quanyi/project/Data/e_optha_MA/Test",[112,112],0.5,"image",0.8)

    ma_patch.forward()

if __name__ == "__main__":
    main()
    
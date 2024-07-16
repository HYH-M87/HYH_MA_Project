from mlp.dataset import Generate

if __name__ == "__main__":
    source_dir = "/home/hyh/Documents/quanyi/project/Data/e_optha_MA/MA_healthy"
    save_dir="/home/hyh/Documents/quanyi/project/Data/e_optha_MA/extract_sample_test"
    type="VOC"
    patch_size=[56,56]
    g = Generate(source_dir, save_dir, type, patch_size)
    g.forward()
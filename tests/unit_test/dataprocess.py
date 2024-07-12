from devlib import GetImagePatch
from devlib import DataConstructor

class patch_data(DataConstructor):
    def __init__(self) -> None:
        super().__init__()
        self.statistics_data_template = None
        self.imgdata_template = {
            'origin':None
        }
        
        self._re_construct()
tool = GetImagePatch(patch_data())

img_p = "tests/test_data/C0000886.jpg"
txt_p = "tests/test_data/C0000886.txt"

d = tool.cut_patch(img_p,txt_p,[112,112],0.5)
with open("test.txt","w") as f:
    f.writelines([ i+" " for i in d[0]['annotation']['value'].flatten().astype(str)])

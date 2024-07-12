import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from mlp.plugins import CBAM
import torch.nn.init as init
import torch.nn.functional as F


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.relu = nn.ReLU(inplace=True)
        self.cbam = CBAM(self.expansion * planes, kernel_size=3)
        
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
        #self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                init.constant_(m.bias, 0)
                
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.cbam(out)
        
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, deep_stem=False):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.deep_stem = deep_stem
        
        self._make_stem_layer(3, 64)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layre_names = []
        for i,nb in enumerate(num_blocks):
            if i==0:
                stride = 1
            else:
                stride = 2
                
            layer = self._make_layer(block, 64 * 2**i, nb, stride)
            self.add_module(f'layer{i+1}', layer)
            self.layre_names.append(f'layer{i+1}')
        
        # self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        # self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    
    def _make_stem_layer(self, in_channels, stem_channels):
        if self.deep_stem:
            self.stem = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False),
                nn.BatchNorm2d(stem_channels // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                nn.BatchNorm2d(stem_channels // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                nn.BatchNorm2d(stem_channels),
                nn.ReLU(inplace=True))
            
        else:
            self.conv1 = nn.Conv2d(
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
            self.norm1 = nn.BatchNorm2d(
                stem_channels)
            self.relu = nn.ReLU(inplace=True)
            
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        outs=[]
        x = self.maxpool(x)
        for layer_name in self.layre_names:
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            outs.append(x)
        return outs

class Neck(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.horizion_layer = []
        self.inplanes = [256,512,1024,2048]
        self.make_horizion()
        self.make_fusion_layer()
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                init.constant_(m.bias, 0)
        
    def make_fusion_layer(self):
        
        in_feature=  0
        for i in range(len(self.inplanes)): 
            in_feature += self.inplanes[i]
        out_feature = in_feature // 4
        conv1 = nn.Conv2d(in_feature, out_feature, 1, 1, 0)
        bn1 = nn.BatchNorm2d(out_feature)
        conv2 = nn.Conv2d(out_feature, out_feature, 3, 1, 1)
        bn2 = nn.BatchNorm2d(out_feature)
        self.add_module(f"fusion_layer", nn.Sequential(conv1, bn1, conv2 , bn2))
            
            
    def make_horizion(self):
        for i in self.inplanes:
            conv = nn.Conv2d(i, i, 3, 1, 1)
            bn = nn.BatchNorm2d(i)
            self.add_module(f"horizion_layer{i}", nn.Sequential(conv,bn))
            self.horizion_layer.append(f"horizion_layer{i}")
            
    def fusion_high(self, outs):
        _,c,h,w = outs[0].shape
        y = [outs[0]]
        for x in outs[1:]:
            x = F.interpolate(x ,size=(h,w), mode='bilinear')
            y.append(x)
        y_h = torch.cat(y,dim=1)
        

        
        return y_h
    
    def fusion_low(self, outs):

        _,c,h,w = outs[-1].shape
        y = [outs[-1]]
        for x in outs[:-1]:
            x = F.interpolate(x ,size=(h,w), mode='bilinear')
            y.append(x)
        y_l = torch.cat(y,dim=1)
        
        return y_l
    
    def forward(self, ins:list[torch.Tensor]):
        '''
        tuple(x1,x2,x3,x4)
        x0, x1,x2,x3,x4,x5
        for 
        x1 -> x1(h)
        
        '''
        outs=[]
        for i, name in enumerate(self.horizion_layer):
            layer = getattr(self, name)
            ins[i] = layer(ins[i])
        
        x_h = self.fusion_high(ins)
        x_l = self.fusion_low(ins)
        
        layer = getattr(self, "fusion_layer")
        x_h = layer(x_h)
        x_l = layer(x_l)
        outs.extend([x_h, x_l])
            
        return outs

class Head(nn.Module):
    def __init__(self, levels=[4, 2, 1], in_planes=960, out_planes=2) -> None:
        super().__init__()
        self.levels = levels
        
        extend=0
        for i in levels:
            extend += (i**2)
            
        fc1 = nn.Linear(in_planes*extend, in_planes*extend // 4)
        relu = nn.ReLU()
        fc2 = nn.Linear(in_planes*extend // 4, out_planes)
        self.add_module("classifier", nn.Sequential(fc1, relu, fc2))
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                init.constant_(m.bias, 0)
    
    def spp(self, x):
        b, c, h, w = x.size()
        output = []

        for level in self.levels:
            kernel_size = (h // level, w // level)
            stride = kernel_size
            pooling = nn.AdaptiveMaxPool2d((level, level))
            pooled = pooling(x)  
            output.append(pooled.view(b, -1))  

        output = torch.cat(output, dim=-1)  
        
        # b*960*21
        return output 
    
    def forward(self,ins):
        y1 = self.spp(ins[0])
        y2 = self.spp(ins[1])
        out = y1 + y2
        out = getattr(self, "classifier")(out)
        return out

def ResNet50(num_classes=1000,deep_stem=False):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes, deep_stem)
    
    pretrained_model = models.resnet50(weights='IMAGENET1K_V1')
    pretrained_dict = pretrained_model.state_dict()
    pretrained_dict.pop('fc.weight')
    pretrained_dict.pop('fc.bias')
    
    model.load_state_dict(pretrained_dict,False)
    return model

class THCModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.backbone = ResNet50(2, True)
        self.neck = Neck()
        self.head = Head()
    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x


if __name__ == "__main__":
    # for debug
    import cv2
    img = cv2.imread("/home/hyh/Documents/quanyi/project/Data/e_optha_MA/ProcessedData/MAimages_CutPatch(112,112)_overlap50.0_ex/VOC2012/JPEGImages/C0000886_block_1.jpg")
    t = transforms.ToTensor()
    img = t(img)
    img = img.unsqueeze(0)
    # model = ResNet50(2,True)
    
    # output = model(img)
    # neck = Neck()
    # print(neck)
    # output = neck(output)
    # head = Head()
    # print(head)
    # output = head(output)
    # print(output.shape)
    
    model = THCModel()
    print(model)
    x = model(img)
    print(x.shape)
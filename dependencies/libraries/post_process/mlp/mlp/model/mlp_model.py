import torch
import torch.nn as nn





# 定义 MLP 模型
class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.ef1 = nn.Sequential( nn.Conv2d(3,32,3),nn.BatchNorm2d(32), nn.ReLU())
        self.ef2 = nn.Sequential( nn.Conv2d(32,32,3),nn.BatchNorm2d(32), nn.ReLU())
        self.ef3 = nn.Sequential( nn.Conv2d(32,3,3),nn.BatchNorm2d(3), nn.ReLU())
        self.ef4 = nn.Sequential( nn.Conv2d(32,3,3),nn.BatchNorm2d(3), nn.ReLU())
        self.pool1 = nn.AvgPool2d(2,2)
        
        self.fc1 = nn.Linear(1215, 1215*2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1215, 1215*2)
        self.fc3 = nn.Linear(1215*2, 512)
        self.fc4 = nn.Linear(512, 2)
        self.sigmoid = nn.Sigmoid()
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        out = self.ef1(x)
        out_y = self.pool1(out)
        out = self.ef2(out)
        out = self.ef3(out)
        out_y = self.ef4(out_y)  #b*(3*12*12)
        out = out.view((out.shape[0],-1))  #b*(3*24*24)
        out_y = out_y.view((out_y.shape[0],-1))
        out = torch.cat((out,out_y),dim=1)
        out_y = self.fc2(out)
        out = self.relu(self.fc1(out))
        out = out*out_y
        out = self.fc4(self.fc3(out))
        out = self.sigmoid(out)
        
        return out

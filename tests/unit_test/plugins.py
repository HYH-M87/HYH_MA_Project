import torch
import torch.nn as nn
import torchvision.ops as ops

class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(DeformableConv2d, self).__init__()
        self.offset = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.deform_conv = ops.DeformConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.offset.weight, mode='fan_out', nonlinearity='relu')
        if self.offset.bias is not None:
            nn.init.constant_(self.offset.bias, 0)
        
        nn.init.kaiming_normal_(self.deform_conv.weight, mode='fan_out', nonlinearity='relu')
        if self.deform_conv.bias is not None:
            nn.init.constant_(self.deform_conv.bias, 0)
    
    def forward(self, x):
        offset = self.offset(x)
        x = self.deform_conv(x, offset)
        return x

# 示例用法
if __name__ == "__main__":
    input_tensor = torch.randn(1, 3, 64, 64)  # 示例输入张量
    model = DeformableConv2d(3, 64, kernel_size=3, padding=1)
    output = model(input_tensor)
    print(output.shape)

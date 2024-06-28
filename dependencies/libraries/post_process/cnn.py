import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection
from torchvision import models

from mlp.dataset import MA_patch

# 定义数据增强和预处理
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = MA_patch("/home/hyh/Documents/quanyi/project/Data/e_optha_MA/extract_sample",True,transform)
test_dataset = MA_patch("/home/hyh/Documents/quanyi/project/Data/e_optha_MA/extract_sample",False,transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

# 加载预训练的 ResNeXt50 模型
model = models.resnext50_32x4d(pretrained=True)

# 修改最后的全连接层用于二分类
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

# 使用GPU（如果可用）
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# 定义优化器
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = optimizer(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # 验证模型
    model.eval()
    corrects = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)
    
    accuracy = corrects.double() / len(test_loader.dataset)
    print(f"Validation Accuracy: {accuracy:.4f}")

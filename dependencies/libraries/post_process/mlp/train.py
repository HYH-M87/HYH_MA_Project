
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import os
from torch.utils.tensorboard import SummaryWriter 

from mlp.dataset import MA_patch
from mlp.model import MLP

# 超参数
input_size = 3 * 24 * 24  
output_size = 2  
learning_rate = 0.001
num_epochs = 300
batch_size = 32

log_dir = "logs/MA_Detection/mlp_cls_test"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    
log_path = os.path.join(log_dir,'log.txt')
checkpoint = os.path.join(log_dir,'final_model.pth')
writer = SummaryWriter(log_dir=os.path.join(log_dir,'tf'))

# 数据预处理和加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = MA_patch("/home/hyh/Documents/quanyi/project/Data/e_optha_MA/extract_sample",True,transform)
test_dataset = MA_patch("/home/hyh/Documents/quanyi/project/Data/e_optha_MA/extract_sample",False,transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model = MLP(input_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)



# model.load_state_dict(checkpoint)
for epoch in range(num_epochs):
    tl=0
    sl=0
    for i, (item_info) in enumerate(train_loader):

        images,labels = item_info['image'], item_info['label']
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        tl += loss.item()
        sl += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {sl/10:.4f}')
            writer.add_scalar('training loss', (sl/10), epoch * len(train_loader) + i)
            with open(log_path,"a") as f:
                f.write(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {sl/10:.4f}\n')
            sl=0
    
    with open(log_path,"a") as f:
        f.write(f'model save to '+f'{log_dir}/epoch_{epoch}.pth\n')
    torch.save(model.state_dict(), os.path.join(log_dir, f'epoch_{epoch}.pth'))

    # 
    model.eval() 
    with torch.no_grad():
        correct = 0
        total = 0
        gt_total = 0
        tp_total = 0
        for item_info in test_loader:
            images,labels = item_info['image'], item_info['label']
            label = torch.max(labels.data, 1)[1]
            gt_total += (label==0).sum().item()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == torch.max(labels.data, 1)[1]).sum().item()
            
            tp_total += ((predicted == torch.max(labels.data, 1)[1]) * (label==0)).sum().item()
        with open(log_path,"a") as f:
            f.write(f'Accuracy: {100 * correct / total:.2f}%  |  Recall: {100 * tp_total / gt_total:.2f}%\n')
            writer.add_scalar('Accuracy', (correct / total), epoch)
            writer.add_scalar('Recall', (tp_total / gt_total), epoch)
        print(f'Accuracy: {100 * correct / total:.2f}%  |  Recall: {100 * tp_total / gt_total:.2f}%')

# 保存模型
writer.close()
torch.save(model.state_dict(), os.path.join(log_dir, f'final_model.pth'))
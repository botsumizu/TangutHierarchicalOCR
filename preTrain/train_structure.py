import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os
import time
import copy
import matplotlib.pyplot as plt

# --- 配置部分 (你可以根据需要修改这些参数) ---
data_dir = 'TangutDataset'  # 你的数据集根目录
model_save_path = 'tangut_structure_classifier.pth'  # 训练好的模型保存路径
num_epochs = 15  # 训练轮数，对于小数据集，15-25轮通常足够观察趋势
batch_size = 4  # 批量大小，由于你数据很少，用小的批量大小
learning_rate = 0.001  # 学习率

# --- 1. 数据预处理与增强 ---
# 定义训练和验证阶段不同的数据变换（Data Transforms）
data_transforms = {
    'train': transforms.Compose([
        # 对于训练集：使用随机增强来人为增加数据多样性，防止过拟合
        transforms.RandomResizedCrop(224),  # 随机裁剪并缩放至224x224
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转（50%概率）。对于对称性不强的文字，可以移除或降低概率。
        transforms.ToTensor(),  # 将PIL图像或numpy数组转换为Tensor，并自动归一化到[0,1]
        # 使用ImageNet的均值和标准差进行归一化，这是通用预训练模型的标准处理方式
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        # 对于验证集：不进行增强，只进行 resize、中心裁剪和归一化，保证评估的公平性
        transforms.Resize(256),  # 将图像短边缩放至256像素
        transforms.CenterCrop(224),  # 从中心裁剪出224x224的区域
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("正在加载数据集...")

# --- 2. 创建数据集和数据加载器 (DataLoader) ---
# 使用ImageFolder自动根据文件夹结构创建数据集，它会自动将子文件夹名作为类别标签
image_datasets = {
    x: datasets.ImageFolder(
        root=os.path.join(data_dir, x),
        transform=data_transforms[x]
    )
    for x in ['train', 'val']
}

# 创建数据加载器，它负责在训练时按批次提供数据
dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x],
        batch_size=batch_size,
        shuffle=True if x == 'train' else False,  # 只有训练集需要打乱顺序
        num_workers=0,  # 用于数据加载的子进程数，在Windows上设为0可避免问题
        pin_memory=True if torch.cuda.is_available() else False  # 如果使用GPU，可以加速数据传输
    )
    for x in ['train', 'val']
}

# 获取数据集大小和类别名称
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes  # 这将自动获取 ['E', 'H', 'S', 'V']

print(f"训练集大小: {dataset_sizes['train']}")
print(f"验证集大小: {dataset_sizes['val']}")
print(f"检测到的类别: {class_names}")

# --- 3. 检查设备并定义模型 ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 加载在ImageNet上预训练的ResNet18模型
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# 修改模型的最后一层（全连接层），使其输出数量等于我们的类别数（4类）
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))

# 将模型移动到GPU或CPU
model = model.to(device)

# --- 4. 定义损失函数和优化器 ---
criterion = nn.CrossEntropyLoss()  # 多分类任务的标准损失函数
# 使用随机梯度下降优化器，并设置权重衰减（L2正则化）来防止过拟合
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
# 定义学习率调度器，每7个epoch将学习率乘以0.1
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


# --- 5. 训练与验证循环函数 ---
def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()
    # 用于保存最佳模型权重
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    # 用于记录训练历史，方便后续绘图
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 60)

        # 每个epoch都有训练和验证两个阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式 (启用Dropout等)
            else:
                model.eval()  # 设置模型为评估模式 (禁用Dropout等)

            running_loss = 0.0
            running_corrects = 0

            # 迭代数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)  # 取输出中最大值的索引作为预测类别
                    loss = criterion(outputs, labels)

                    # 只在训练阶段进行反向传播和优化
                    if phase == 'train':
                        optimizer.zero_grad()  # 清空过往梯度
                        loss.backward()  # 反向传播，计算当前梯度
                        optimizer.step()  # 根据梯度更新模型参数

                # 统计本批次的损失和正确数
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # 一个epoch结束后，计算平均损失和准确率
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # 记录到历史中
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            print(f'{phase:5} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')

            # 如果是验证阶段，并且当前模型是最好的，则深拷贝模型权重
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # 可以在这里保存最佳模型，但我们在整个训练结束后再保存

        if phase == 'train':
            scheduler.step()  # 更新学习率
        print()

    time_elapsed = time.time() - since
    print(f'训练完成于 {time_elapsed // 60:.0f}分 {time_elapsed % 60:.0f}秒')
    print(f'最佳验证集准确率: {best_acc:.4f}')

    # 训练结束后，加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model, history


# --- 6. 开始训练！---
print("开始训练结构分类器...")
model, history = train_model(model, criterion, optimizer, scheduler, num_epochs)

# --- 7. 保存训练好的模型 ---
torch.save(model.state_dict(), model_save_path)
print(f"模型已保存至: {model_save_path}")

# --- 8. (可选) 绘制训练过程曲线 ---
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Acc')
plt.plot(history['val_acc'], label='Val Acc')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.savefig('training_history.png')  # 保存图表
plt.show()

print("全部流程结束！")
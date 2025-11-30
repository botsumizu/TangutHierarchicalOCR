import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os
import time
import copy
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


data_dir = 'TangutDataset'
model_save_path = 'tangut_structure_classifier_balanced.pth'
num_epochs = 30
batch_size = 16
learning_rate = 0.001


print("预处理")
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(100, scale=(0.8, 1.0)),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(110),
        transforms.CenterCrop(100),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("加载数据集")


image_datasets = {
    x: datasets.ImageFolder(
        root=os.path.join(data_dir, x),
        transform=data_transforms[x]
    )
    for x in ['train', 'val']
}


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"\n使用设备: {device}")



def get_class_weights(dataset):
    class_counts = []
    for class_name in dataset.classes:
        class_path = os.path.join(dataset.root, class_name)
        count = len(os.listdir(class_path))
        class_counts.append(count)


    total_samples = sum(class_counts)
    num_classes = len(class_counts)
    weights = [total_samples / (num_classes * count) for count in class_counts]

    print("样本数量、权重")
    for i, class_name in enumerate(dataset.classes):
        print(f"  {class_name}: {class_counts[i]}个样本, 权重: {weights[i]:.2f}")

    return torch.FloatTensor(weights)



train_weights = get_class_weights(image_datasets['train']).to(device)


train_samples_weight = []
for _, label in image_datasets['train'].samples:
    train_samples_weight.append(train_weights[label].item()) 

train_sampler = torch.utils.data.WeightedRandomSampler(
    train_samples_weight,
    len(train_samples_weight),
    replacement=True
)


dataloaders = {
    'train': torch.utils.data.DataLoader(
        image_datasets['train'],
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    ),
    'val': torch.utils.data.DataLoader(
        image_datasets['val'],
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
}


dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

print(f"\n数据集统计:")
print(f"训练集大小: {dataset_sizes['train']}")
print(f"验证集大小: {dataset_sizes['val']}")
print(f"类别 ({len(class_names)}类): {class_names}")


model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)


num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
model = model.to(device)

print(f" 最后一层调整为 {len(class_names)} 类输出")


criterion = nn.CrossEntropyLoss(weight=train_weights)


optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)


scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.1,
    patience=7,
    verbose=True,
    min_lr=1e-7
)



def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0


    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'learning_rate': [],
        'class_accuracy': {class_name: [] for class_name in class_names}  
    }

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print('-' * 60)


        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            class_correct = {class_name: 0 for class_name in class_names}
            class_total = {class_name: 0 for class_name in class_names}

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)


                for i in range(len(labels)):
                    label = labels[i].item()
                    pred = preds[i].item()
                    class_name = class_names[label]
                    class_total[class_name] += 1
                    if label == pred:
                        class_correct[class_name] += 1

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            if phase == 'train':
                current_lr = optimizer.param_groups[0]['lr']
                history['learning_rate'].append(current_lr)

            print(f'{phase:5} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}', end='')
            if phase == 'train':
                print(f' | LR: {current_lr:.2e}')
            else:
                print()


            if phase == 'val':
                print("验证准确率")
                for class_name in class_names:
                    if class_total[class_name] > 0:
                        class_acc = class_correct[class_name] / class_total[class_name]
                        history['class_accuracy'][class_name].append(class_acc)
                        print(
                            f"  {class_name}: {class_acc:.4f} ({class_correct[class_name]}/{class_total[class_name]})")


            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch + 1
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), f'best_{model_save_path}')


            if phase == 'val':
                scheduler.step(epoch_loss)

    time_elapsed = time.time() - since
    print(f'\n训练完成于 {time_elapsed // 60:.0f}分 {time_elapsed % 60:.0f}秒')
    print(f'最佳验证集准确率: {best_acc:.4f} (第 {best_epoch} epoch)')

    model.load_state_dict(best_model_wts)
    return model, history



print("\n开始训练")
print("=" * 60)
model, history = train_model(model, criterion, optimizer, scheduler, num_epochs)


torch.save(model.state_dict(), model_save_path)
print(f"\n最终模型已保存至: {model_save_path}")
print(f"最佳模型已保存至: best_{model_save_path}")


plt.figure(figsize=(18, 12))


plt.subplot(2, 2, 1)
plt.plot(history['train_loss'], 'b-', label='Train Loss', linewidth=2)
plt.plot(history['val_loss'], 'r-', label='Val Loss', linewidth=2)
plt.legend()
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)


plt.subplot(2, 2, 2)
plt.plot(history['train_acc'], 'b-', label='Train Acc', linewidth=2)
plt.plot(history['val_acc'], 'r-', label='Val Acc', linewidth=2)
plt.legend()
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True, alpha=0.3)


plt.subplot(2, 2, 3)
plt.plot(history['learning_rate'], 'g-', label='Learning Rate', linewidth=2)
plt.legend()
plt.title('Learning Rate Schedule')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.yscale('log')
plt.grid(True, alpha=0.3)


plt.subplot(2, 2, 4)
for class_name in class_names:
    if history['class_accuracy'][class_name]:
        plt.plot(history['class_accuracy'][class_name], label=class_name, linewidth=2)
plt.legend()
plt.title('Validation Accuracy by Class')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history_balanced.png', dpi=300, bbox_inches='tight')
plt.show()


print("\n" + "=" * 60)
print(f"数据集: {dataset_sizes['train']} 训练样本, {dataset_sizes['val']} 验证样本")
print(f"最佳验证准确率: {max(history['val_acc']):.4f}")
print(f"最终训练准确率: {history['train_acc'][-1]:.4f}")
print("各类别最终验证准确率:")
for class_name in class_names:
    if history['class_accuracy'][class_name]:
        final_acc = history['class_accuracy'][class_name][-1]
        print(f"  {class_name}: {final_acc:.4f}")
print("=" * 60)

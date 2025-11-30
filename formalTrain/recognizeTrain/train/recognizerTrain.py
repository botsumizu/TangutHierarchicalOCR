import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os
import time
import copy
import matplotlib.pyplot as plt


structure_type = 'S'  
data_dir = 'TangutRecognitionDataset_final'  
model_save_path = f'tangut_recognizer_{structure_type}_v2.pth'
num_epochs = 15  
batch_size = 16
learning_rate = 0.001



data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(100, scale=(0.85, 1.0)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(100),
        transforms.CenterCrop(100),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}




train_dir = os.path.join(data_dir, 'train', structure_type)
val_dir = os.path.join(data_dir, 'val', structure_type)


try:
    image_datasets = {
        'train': datasets.ImageFolder(root=train_dir, transform=data_transforms['train']),
        'val': datasets.ImageFolder(root=val_dir, transform=data_transforms['val'])
    }


    if hasattr(image_datasets['train'], 'classes') and hasattr(image_datasets['val'], 'classes'):
        if image_datasets['train'].classes != image_datasets['val'].classes:
            image_datasets['val'].classes = image_datasets['train'].classes
            image_datasets['val'].class_to_idx = image_datasets['train'].class_to_idx

except Exception as e:
    print(f"数据集加载错误: {e}")
    exit()

dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x],
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    for x in ['train', 'val']
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

print(f"训练集大小: {dataset_sizes['train']}")
print(f"验证集大小: {dataset_sizes['val']}")
print(f"字符类别数: {len(class_names)}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")

try:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
except AttributeError:
    try:
        model = models.resnet18(pretrained=True)
    except:
        model = models.resnet18(pretrained=True)

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
model = model.to(device)



criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)


scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)



class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


early_stopping = EarlyStopping(patience=5, min_delta=0.001)



def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'learning_rate': []
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

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            if phase == 'train':
                current_lr = optimizer.param_groups[0]['lr']
                history['learning_rate'].append(current_lr)
                scheduler.step()  

            print(f'{phase:5} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}', end='')
            if phase == 'train':
                print(f' | LR: {current_lr:.2e}')
            else:
                print()

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), f'best_{model_save_path}')

            # 早停检查
            if phase == 'val':
                early_stopping(epoch_loss)
                if early_stopping.early_stop:
                    print("早停: 验证损失不再改善")
                    break

        if early_stopping.early_stop:
            break

    time_elapsed = time.time() - since
    print(f'\n训练完成于 {time_elapsed // 60:.0f}分 {time_elapsed % 60:.0f}秒')
    print(f'最佳验证准确率: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model, history



print(f"\n开始训练 {structure_type} 结构文字识别器...")
print("=" * 60)
model, history = train_model(model, criterion, optimizer, scheduler, num_epochs)


torch.save(model.state_dict(), model_save_path)
print(f"\n模型已保存至: {model_save_path}")


plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], 'b-', label='Train Loss')
plt.plot(history['val_loss'], 'r-', label='Val Loss')
plt.legend()
plt.title('Loss')
plt.xlabel('Epoch')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], 'b-', label='Train Acc')
plt.plot(history['val_acc'], 'r-', label='Val Acc')
plt.legend()
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'training_recognizer_{structure_type}_v2.png', dpi=300)
plt.show()

print(f"\n训练完成！最佳验证准确率: {max(history['val_acc']):.4f}")
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据路径
train_path = '/kaggle/input/neu3123/fer_data/fer_data/train'
test_path = '/kaggle/input/neu3123/fer_data/fer_data/test'

# 情感标签映射
emotion_labels = {
    'Angry': 0,
    'Fear': 1,
    'Happy': 2,
    'Sad': 3,
    'Surprise': 4,
    'Neutral': 5
}
reverse_labels = {v: k for k, v in emotion_labels.items()}
label_names = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


# 自定义数据集类
class FERDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.is_train = labels is not None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L').convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.is_train:
            return image, self.labels[idx]
        else:
            return image, img_path


# 更强的数据增强
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 提高分辨率
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),  # 增加旋转角度
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 加载训练数据
def load_train_data():
    image_paths = []
    labels = []

    for emotion, label in emotion_labels.items():
        emotion_path = os.path.join(train_path, emotion)
        if os.path.exists(emotion_path):
            for img_file in os.listdir(emotion_path):
                if img_file.endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(emotion_path, img_file)
                    image_paths.append(img_path)
                    labels.append(label)

    return image_paths, labels


# 加载数据
image_paths, labels = load_train_data()
train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42, stratify=labels
)

# 创建数据加载器
train_dataset = FERDataset(train_paths, train_labels, train_transform)
val_dataset = FERDataset(val_paths, val_labels, val_transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)


# 改进的模型架构
class EnhancedResNetFER(nn.Module):
    def __init__(self, num_classes=6, dropout_rate=0.5):
        super(EnhancedResNetFER, self).__init__()

        # 使用预训练的ResNet18
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        # 替换第一个卷积层以接受单通道输入（灰度图）
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 复制预训练权重（取RGB三个通道的平均值）
        with torch.no_grad():
            backbone.conv1.weight[:, 0] = backbone.conv1.weight[:, :3].mean(dim=1)

        # 获取特征维度
        num_features = backbone.fc.in_features

        # 更深的分类头
        self.features = nn.Sequential(*list(backbone.children())[:-1])

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

        # 初始化分类器权重
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# Focal Loss用于处理类别不平衡
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# 训练函数
def train_epoch(model, dataloader, criterion, optimizer, device, scheduler=None):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    if scheduler is not None:
        scheduler.step()

    epoch_loss = running_loss / len(dataloader)
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted')

    return epoch_loss, epoch_f1


# 验证函数
def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader)
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
    f1_per_class = f1_score(all_labels, all_preds, average=None)

    return epoch_loss, epoch_f1, f1_per_class, all_preds, all_labels


# 早停类
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0
        return self.early_stop


# 训练配置
model = EnhancedResNetFER(num_classes=6, dropout_rate=0.6).to(device)

# 组合损失函数
criterion1 = nn.CrossEntropyLoss(label_smoothing=0.1)  # 标签平滑
criterion2 = FocalLoss(gamma=2, alpha=0.25)


def combined_loss(outputs, targets, alpha=0.7):
    return alpha * criterion1(outputs, targets) + (1 - alpha) * criterion2(outputs, targets)


# 优化器和学习率调度
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-6
)

# 早停
early_stopping = EarlyStopping(patience=10, min_delta=0.001)

# 训练循环
num_epochs = 50
best_val_f1 = 0.0
history = {'train_loss': [], 'train_f1': [], 'val_loss': [], 'val_f1': []}

print("Starting training...\n")
print(f"Training samples: {len(train_paths)}")
print(f"Validation samples: {len(val_paths)}")
print("-" * 50)

for epoch in range(num_epochs):
    # 训练阶段
    train_loss, train_f1 = train_epoch(
        model, train_loader,
        lambda outputs, targets: combined_loss(outputs, targets),
        optimizer, device, scheduler
    )

    # 验证阶段
    val_loss, val_f1, f1_per_class, val_preds, val_labels = validate_epoch(
        model, val_loader,
        lambda outputs, targets: combined_loss(outputs, targets),
        device
    )

    # 记录历史
    history['train_loss'].append(train_loss)
    history['train_f1'].append(train_f1)
    history['val_loss'].append(val_loss)
    history['val_f1'].append(val_f1)

    # 保存最佳模型
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_f1': val_f1,
            'val_loss': val_loss,
            'f1_per_class': f1_per_class,
        }, 'best_model.pth')

    # 早停检查
    if early_stopping(val_f1):
        print(f"\nEarly stopping triggered at epoch {epoch + 1}")
        break

    # 每5个epoch输出一次
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch + 1:3d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")

print(f"\nBest validation F1 score: {best_val_f1:.4f}")

# 加载最佳模型进行最终评估
checkpoint = torch.load('best_model.pth', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])

print("\n" + "=" * 50)
print("Final Evaluation on Validation Set")
print("=" * 50)

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print("\nClassification Report:")
print(classification_report(all_labels, all_preds,
                            target_names=label_names, digits=4))

print(f"\nWeighted F1 Score: {f1_score(all_labels, all_preds, average='weighted'):.4f}")
print(f"Macro F1 Score: {f1_score(all_labels, all_preds, average='macro'):.4f}")

# 生成测试集预测
print("\n" + "=" * 50)
print("Generating Test Predictions")
print("=" * 50)

# 加载提交文件
submission_path = '/kaggle/input/neu3123/submission.csv'
original_submission = pd.read_csv(submission_path)

# 创建测试集数据加载器
test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_files = sorted([f for f in os.listdir(test_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
test_predictions = {}

model.eval()
with torch.no_grad():
    for filename in test_files:
        img_path = os.path.join(test_path, filename)
        image = Image.open(img_path).convert('L').convert('RGB')
        image = test_transform(image).unsqueeze(0).to(device)

        output = model(image)
        _, pred = torch.max(output, 1)
        img_id = os.path.splitext(filename)[0]
        test_predictions[img_id] = int(pred.cpu().numpy())

# 更新提交文件
new_submission = original_submission.copy()
id_column = 'ID'
emotion_column = 'Emotion'

if emotion_column in new_submission.columns:
    updated_count = 0
    for idx, row in new_submission.iterrows():
        img_id_with_ext = row[id_column]
        if isinstance(img_id_with_ext, str):
            if img_id_with_ext.endswith('.jpg'):
                img_id = img_id_with_ext[:-4]
            else:
                img_id = img_id_with_ext

            if img_id in test_predictions:
                new_submission.at[idx, emotion_column] = test_predictions[img_id]
                updated_count += 1

# 保存新的提交文件
new_submission_path = 'submission.csv'
new_submission.to_csv(new_submission_path, index=False)

print(f"\nUpdated {updated_count} predictions in submission file")
print(f"New submission saved as: {new_submission_path}")

# 输出关键结果摘要
print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"Best Validation F1 Score: {best_val_f1:.4f}")
print(f"Final Epoch: {checkpoint['epoch'] + 1}")
print(f"Training Samples: {len(train_paths)}")
print(f"Validation Samples: {len(val_paths)}")
print(f"Test Samples: {len(test_files)}")
print(f"Model Architecture: Enhanced ResNet18")
print(f"Image Size: 64x64")
print(f"Data Augmentation: Strong")
print(f"Loss Function: CrossEntropy + Focal Loss")
print(f"Optimizer: AdamW with Cosine Annealing")
print("-" * 50)

# 输出验证集F1分数历史（只输出最佳和最后5个epoch）
print("\nValidation F1 Scores (Best and Recent):")
print("-" * 30)
for i, f1 in enumerate(history['val_f1']):
    if i == history['val_f1'].index(max(history['val_f1'])) or i >= len(history['val_f1']) - 5:
        marker = "★" if i == history['val_f1'].index(max(history['val_f1'])) else " "
        print(f"Epoch {i + 1:3d}{marker}: {f1:.4f}")
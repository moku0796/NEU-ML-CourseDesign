# simple_version.py
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

# 设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 路径
train_path = '/kaggle/input/neu2123/dataset-for-task2/dataset-for-task2/train'
test_path = '/kaggle/input/neu2123/dataset-for-task2/dataset-for-task2/test'

# 类别
class_names = ['Black-grass', 'Common wheat', 'Loose Silky-bent',
               'Scentless Mayweed', 'Sugar beet']
class_to_idx = {name: idx for idx, name in enumerate(class_names)}


# 简化数据集
class SimplePlantDataset(Dataset):
    def __init__(self, image_paths, labels=None, is_train=True):
        self.image_paths = image_paths
        self.labels = labels
        self.is_train = is_train

        # 简化的数据增强
        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(30),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        img = self.transform(img)

        if self.is_train:
            return img, self.labels[idx]
        return img


# 加载数据
def load_data():
    all_images = []
    all_labels = []

    for class_name in class_names:
        class_path = os.path.join(train_path, class_name)
        if not os.path.exists(class_path):
            print(f"Warning: {class_path} does not exist!")
            continue

        images = []
        for f in os.listdir(class_path):
            if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                images.append(os.path.join(class_path, f))

        print(f"{class_name}: {len(images)} images")
        all_images.extend(images)
        all_labels.extend([class_to_idx[class_name]] * len(images))

    return np.array(all_images), np.array(all_labels)


# 训练单个模型
def train_single_model():
    print("Loading data...")
    all_images, all_labels = load_data()

    if len(all_images) == 0:
        print("Error: No training images found!")
        return None

    print(f"\nTotal images: {len(all_images)}")

    # 使用全部数据训练一个模型（不交叉验证）
    train_dataset = SimplePlantDataset(all_images, all_labels, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)

    # 使用EfficientNet
    model = models.efficientnet_b0(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
    model = model.to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 训练
    print("\nTraining model...")
    for epoch in range(20):
        model.train()
        total_loss = 0
        all_preds = []
        all_labels_batch = []

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(output, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels_batch.extend(target.cpu().numpy())

        scheduler.step()
        f1 = f1_score(all_labels_batch, all_preds, average='macro')

        print(f"Epoch {epoch + 1:02d}: Loss: {total_loss / len(train_loader):.4f}, F1: {f1:.4f}")

    return model


# 预测
def predict(model):
    # 加载测试数据
    test_images = []
    test_ids = []

    for f in os.listdir(test_path):
        if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
            test_images.append(os.path.join(test_path, f))
            test_ids.append(f)

    print(f"\nFound {len(test_images)} test images")

    if len(test_images) == 0:
        # 尝试其他方式
        for root, dirs, files in os.walk(test_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                    test_images.append(os.path.join(root, file))
                    test_ids.append(file)

    test_dataset = SimplePlantDataset(test_images, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

    # 预测
    model.eval()
    predictions = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data)
            preds = torch.argmax(output, dim=1)
            predictions.extend(preds.cpu().numpy())

    # 保存结果
    submission = pd.DataFrame({
        'file': test_ids,
        'species': [class_names[p] for p in predictions]
    })

    submission.to_csv('submission.csv', index=False)
    print("\nSubmission saved to 'submission.csv'")
    print("\nPrediction distribution:")
    print(submission['species'].value_counts())


# 主函数
if __name__ == "__main__":
    model = train_single_model()
    if model is not None:
        predict(model)
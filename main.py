import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime
import csv
import matplotlib.pyplot as plt
from torchvision.models import mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large

# ==================== 日志与CSV记录模块 ====================
def setup_logger(args):
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join('LOG',
                                f'{args.model}_lr{args.learning_rate}_bs{args.batch_size}_ep{args.epochs}_run_{now}.txt')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # 文件输出
    file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger, log_filename, now


def create_csv_file(args, now):
    csv_filename = os.path.join('CSV',
                                f'{args.model}_lr{args.learning_rate}_bs{args.batch_size}_ep{args.epochs}_run_{now}.csv')
    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
    return csv_filename


def save_epoch_to_csv(csv_file, epoch, train_loss, train_acc, val_loss, val_acc):
    """保存每个epoch的训练与验证结果"""
    with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch + 1,
            f'{train_loss:.4f}',
            f'{train_acc:.2f}',
            f'{val_loss:.4f}',
            f'{val_acc:.2f}'
        ])


def plot_metrics(train_losses, val_losses, train_accs, val_accs, args, now):
    """生成并保存accuracy和loss图表"""
    epochs = range(1, len(train_losses) + 1)

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 生成Loss图
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=6)
    plt.plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2, markersize=6)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    loss_filename = os.path.join('Plot',
                                 f'{args.model}_lr{args.learning_rate}_bs{args.batch_size}_ep{args.epochs}_run_{now}_loss.png')
    plt.savefig(loss_filename, dpi=300, bbox_inches='tight')
    plt.close()

    # 生成Accuracy图
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accs, 'b-o', label='Train Accuracy', linewidth=2, markersize=6)
    plt.plot(epochs, val_accs, 'r-s', label='Validation Accuracy', linewidth=2, markersize=6)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    acc_filename = os.path.join('Plot',
                                f'{args.model}_lr{args.learning_rate}_bs{args.batch_size}_ep{args.epochs}_run_{now}_acc.png')
    plt.savefig(acc_filename, dpi=300, bbox_inches='tight')
    plt.close()

    return acc_filename, loss_filename

class CenterLoss(nn.Module):
    """
    Center Loss 用于增强类内紧凑性
    """
    def __init__(self, num_classes=4, feat_dim=512, device='cuda'):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        # 每个类别一个中心向量
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))

    def forward(self, features, labels):
        """
        features: [batch_size, feat_dim]
        labels: [batch_size]
        """
        batch_size = features.size(0)
        # 获取对应类别的中心
        centers_batch = self.centers[labels]
        # 计算欧式距离
        loss = ((features - centers_batch).pow(2).sum()) / (2.0 * batch_size)
        return loss

class FocalLoss(nn.Module):
    """
    Focal Loss 用于处理类别不平衡问题
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, labels):
        ce_loss = nn.functional.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)  # 正确分类概率
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ==================== 自定义数据集类 ====================
class GarbageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# ==================== ResNet实现 ====================
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=4):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        logits = self.fc(features)
        return logits, features


def ResNet18(num_classes=4):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def ResNet50(num_classes=4):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


# ==================== VGG实现 ====================
class VGG(nn.Module):
    def __init__(self, features, num_classes=4):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        logits = self.classifier(features)
        return logits, features


def make_vgg_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def VGG16(num_classes=4):
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
           512, 512, 512, 'M', 512, 512, 512, 'M']
    return VGG(make_vgg_layers(cfg, batch_norm=True), num_classes)

# ==================== 线性注意力模块 ====================
class LinearAttentionBlock(nn.Module):
    """轻量线性注意力模块 (用于ResNet18改进版)"""
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_q = nn.Linear(dim, hidden_dim, bias=False)
        self.to_k = nn.Linear(dim, hidden_dim, bias=False)
        self.to_v = nn.Linear(dim, hidden_dim, bias=False)
        self.proj = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # 分头
        B, N, _ = q.shape
        q = q.view(B, N, self.heads, -1)
        k = k.view(B, N, self.heads, -1)
        v = v.view(B, N, self.heads, -1)

        # 线性注意力计算 (无需softmax)
        k_softmax = torch.softmax(k, dim=1)
        context = torch.einsum('bnhd,bnhv->bhdv', k_softmax, v)
        out = torch.einsum('bnhd,bhdv->bnhv', q, context).reshape(B, N, -1)

        out = self.proj(out)
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return out

# ==================== 改进版 ResNet18-LA ====================
class ResNet18_LA(ResNet):
    """在ResNet18基础上增加线性注意力模块"""

    def __init__(self, num_classes=4):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes)
        self.attn = LinearAttentionBlock(dim=512 * BasicBlock.expansion, heads=4)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 在线性注意力前加入
        x = self.attn(x)

        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        logits = self.classifier(features)
        return logits, features

class MobileNetV2Custom(nn.Module):
    def __init__(self, num_classes=4, feature_dim=512):  # 新增feature_dim参数
        super(MobileNetV2Custom, self).__init__()

        # 第一层：卷积层，输入3个通道，输出32个通道，大小为3x3，stride为2
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        # 倒残差模块
        self.layer1 = self._make_layer(32, 16, stride=1)
        self.layer2 = self._make_layer(16, 32, stride=2)
        self.layer3 = self._make_layer(32, 64, stride=2)
        self.layer4 = self._make_layer(64, 64, stride=2)
        self.layer5 = self._make_layer(64, 128, stride=2)
        self.layer6 = self._make_layer(128, 256, stride=2)
        self.layer7 = self._make_layer(256, 512, stride=1)

        # 最后卷积层，升维输出
        self.conv2 = nn.Conv2d(512, feature_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(feature_dim)

        # 全局池化
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # 分类层
        self.fc = nn.Linear(feature_dim, num_classes)

    def _make_layer(self, in_channels, out_channels, stride=1):
        # 倒残差模块：逐点卷积 + 深度可分离卷积 + 逐点卷积
        layers = [
            # 第一层逐点卷积：用于扩大通道数
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

            # 第二层深度可分离卷积
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

            # 第三层逐点卷积：将通道数降回去
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        # 第一层卷积 + ReLU + 批量归一化
        x = self.relu(self.bn1(self.conv1(x)))

        # 通过倒残差模块
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)

        # 最后卷积 + 批量归一化 + ReLU
        x = self.relu(self.bn2(self.conv2(x)))

        # 全局池化
        x = self.avgpool(x)

        # 扁平化输出
        features = torch.flatten(x, 1)

        # 分类输出
        logits = self.fc(features)
        return logits, features



class MobileNetV3_SmallCustom(nn.Module):
    def __init__(self, num_classes=4):
        super(MobileNetV3_SmallCustom, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # 添加倒残差块和瓶颈层
        self.layer1 = self._make_layer(16, 24, stride=2)
        self.layer2 = self._make_layer(24, 40, stride=2)
        self.layer3 = self._make_layer(40, 80, stride=2)
        self.layer4 = self._make_layer(80, 112, stride=1)
        self.layer5 = self._make_layer(112, 160, stride=2)

        # 最终卷积层
        self.conv2 = nn.Conv2d(160, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1024)

        # 平均池化
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # 分类层
        self.fc = nn.Linear(1024, num_classes)

    def _make_layer(self, in_channels, out_channels, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.avgpool(x)

        features = torch.flatten(x, 1)
        logits = self.fc(features)
        return logits, features


class MobileNetV3_LargeCustom(nn.Module):
    def __init__(self, num_classes=4):
        super(MobileNetV3_LargeCustom, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # 添加倒残差块和瓶颈层
        self.layer1 = self._make_layer(16, 24, stride=2)
        self.layer2 = self._make_layer(24, 40, stride=2)
        self.layer3 = self._make_layer(40, 80, stride=2)
        self.layer4 = self._make_layer(80, 112, stride=1)
        self.layer5 = self._make_layer(112, 160, stride=2)
        self.layer6 = self._make_layer(160, 320, stride=1)

        # 最终卷积层
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)

        # 平均池化
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # 分类层
        self.fc = nn.Linear(1280, num_classes)

    def _make_layer(self, in_channels, out_channels, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.avgpool(x)

        features = torch.flatten(x, 1)
        logits = self.fc(features)
        return logits, features


# ==================== 数据加载 ====================
def load_dataset(data_dir):
    class_mapping = {
        '厨余垃圾': 0,
        '可回收物': 1,
        '其他垃圾': 2,
        '有害垃圾': 3
    }

    image_paths, labels = [], []
    for folder_name in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        label = next((idx for cls, idx in class_mapping.items() if folder_name.startswith(cls)), None)
        if label is None:
            print(f"警告: 文件夹 {folder_name} 不属于任何已知类别，跳过")
            continue

        for img_name in os.listdir(folder_path):
            if img_name.lower().endswith('.jpg'):
                image_paths.append(os.path.join(folder_path, img_name))
                labels.append(label)

    print(f"共加载 {len(image_paths)} 张图片")
    print(f"类别分布: {np.bincount(labels)}")
    return image_paths, labels, class_mapping


# ==================== 模型获取与训练 ====================
def get_model(model_name, num_classes=4):
    if model_name == 'resnet18':
        return ResNet18(num_classes)
    elif model_name == 'resnet18_la':
        return ResNet18_LA(num_classes)
    elif model_name == 'resnet50':
        return ResNet50(num_classes)
    elif model_name == 'vgg16':
        return VGG16(num_classes)
    elif model_name == 'mobilenetv2':
        return MobileNetV2Custom(num_classes)
    elif model_name == 'mobilenetv3_small':
        return MobileNetV3_SmallCustom(num_classes)
    elif model_name == 'mobilenetv3_large':
        return MobileNetV3_LargeCustom(num_classes)
    else:
        raise ValueError(f"不支持的模型: {model_name}")



def train_epoch(model, dataloader, ce_loss_fn, center_loss_fn, focal_loss_fn, optimizer, device, eta, lam):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(dataloader, desc='训练中')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs, features = model(images)

        # 分别计算三种loss
        loss_ce = ce_loss_fn(outputs, labels)
        loss_center = center_loss_fn(features, labels)
        loss_focal = focal_loss_fn(outputs, labels)

        # 融合总损失
        total_loss = (1 - eta - lam) * loss_ce + eta * loss_center + lam * loss_focal
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        pbar.set_postfix({'loss': f'{running_loss / len(dataloader):.4f}', 'acc': f'{100. * correct / total:.2f}%'})
    return running_loss / len(dataloader), 100. * correct / total


def validate(model, dataloader, ce_loss_fn, center_loss_fn, focal_loss_fn, device, eta, lam):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='验证中'):
            images, labels = images.to(device), labels.to(device)
            outputs, features = model(images)
            loss_ce = ce_loss_fn(outputs, labels)
            loss_center = center_loss_fn(features, labels)
            loss_focal = focal_loss_fn(outputs, labels)
            total_loss = (1 - eta - lam) * loss_ce + eta * loss_center + lam * loss_focal

            running_loss += total_loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return running_loss / len(dataloader), 100. * correct / total



# ==================== 主函数 ====================
def main():
    os.makedirs('CSV', exist_ok=True)
    os.makedirs('LOG', exist_ok=True)
    os.makedirs('PT', exist_ok=True)
    os.makedirs('Plot', exist_ok=True)

    parser = argparse.ArgumentParser(description='垃圾分类图像分类')
    parser.add_argument('--data_dir', type=str, default='Dataset')
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet50', 'vgg16', 'resnet18_la','mobilenetv2', 'mobilenetv3_small'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--test_split', type=float, default=0.2)
    parser.add_argument('--save_path', type=str, default='best_model.pth')
    parser.add_argument('--eta', type=float, default=0.2, help='Center loss 权重 (0 < η < 1)')
    parser.add_argument('--lam', type=float, default=0.2, help='Focal loss 权重 (0 < λ < 1)')


    args = parser.parse_args()

    # 设置日志
    logger, log_file, now = setup_logger(args)
    logger.info(f'日志文件: {log_file}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'使用设备: {device}')

    image_paths, labels, class_mapping = load_dataset(args.data_dir)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=args.test_split, random_state=42, stratify=labels
    )

    # 数据增强
    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        # transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        # transforms.RandomVerticalFlip(p=0.2),  # 随机垂直翻转
        transforms.RandomRotation(15),  # 随机旋转 ±15°
        # transforms.ColorJitter(  # 随机颜色扰动
        #     brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05
        # ),
        # transforms.RandomResizedCrop(  # 随机裁剪后缩放回原尺寸
        #     args.image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)
        # ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # DataLoader
    train_dataset = GarbageDataset(train_paths, train_labels, train_transform)
    val_dataset = GarbageDataset(val_paths, val_labels, val_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    logger.info(f'训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}')

    # 模型
    model = get_model(args.model, num_classes=4).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'总参数量: {total_params:,}, 可训练参数量: {trainable_params:,}')

    # 损失函数与优化器
    ce_loss_fn = nn.CrossEntropyLoss()
    center_loss_fn = CenterLoss(num_classes=4, feat_dim=512, device=device)
    focal_loss_fn = FocalLoss(alpha=1.0, gamma=2.0)

    eta, lam = args.eta, args.lam

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_acc = 0.0
    csv_file = create_csv_file(args, now)

    # 用于记录训练历史
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    # 训练过程
    for epoch in range(args.epochs):
        logger.info(f'\nEpoch {epoch + 1}/{args.epochs}')
        logger.info('-' * 50)

        train_loss, train_acc = train_epoch(model, train_loader, ce_loss_fn, center_loss_fn, focal_loss_fn, optimizer,
                                            device, eta, lam)
        val_loss, val_acc = validate(model, val_loader, ce_loss_fn, center_loss_fn, focal_loss_fn, device, eta, lam)

        # 记录指标
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        logger.info(f'训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%')
        logger.info(f'验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%')

        save_epoch_to_csv(csv_file, epoch, train_loss, train_acc, val_loss, val_acc)

        # 调整学习率
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_acc)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            logger.info(f'学习率调整: {old_lr:.6f} -> {new_lr:.6f}')

        if val_acc > best_acc:
            save_path = os.path.join('PT', f'{args.model}_best.pth')
            torch.save(model.state_dict(), save_path)
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'class_mapping': class_mapping,
                'model_name': args.model
            }, args.save_path)
            logger.info(f'保存最佳模型, 准确率: {best_acc:.2f}%')

    # 生成图表
    logger.info('\n生成训练曲线图...')
    acc_plot, loss_plot = plot_metrics(train_losses, val_losses, train_accs, val_accs, args, now)
    logger.info(f'准确率图表已保存: {acc_plot}')
    logger.info(f'损失图表已保存: {loss_plot}')

    logger.info(f'\n训练完成! 最佳验证准确率: {best_acc:.2f}%')
    logger.info(f'模型已保存至: {args.save_path}')
    logger.info(f'日志文件: {log_file}')
    logger.info(f'训练记录CSV: {csv_file}')


if __name__ == '__main__':
    main()
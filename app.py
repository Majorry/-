import os
from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import base64
import json
from datetime import datetime
import sqlite3
from pathlib import Path


# ==================== 数据库初始化 ====================
def init_database():
    """初始化SQLite数据库"""
    conn = sqlite3.connect('garbage_classifier.db')
    cursor = conn.cursor()

    # 创建历史记录表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS classification_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            image_path TEXT NOT NULL,
            predicted_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            all_probabilities TEXT NOT NULL,
            user_ip TEXT
        )
    ''')

    # 创建反馈表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            classification_id INTEGER,
            timestamp TEXT NOT NULL,
            feedback_type TEXT NOT NULL,
            correct_class TEXT,
            comment TEXT,
            user_ip TEXT,
            FOREIGN KEY (classification_id) REFERENCES classification_history(id)
        )
    ''')

    conn.commit()
    conn.close()
    print("数据库初始化完成")


def save_classification_record(image, predicted_class, confidence, all_probs, user_ip):
    """保存分类记录到数据库"""
    try:
        # 保存图像文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        image_dir = Path('static/history_images')
        image_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / f'{timestamp}.jpg'
        image.save(image_path, 'JPEG')

        # 保存到数据库
        conn = sqlite3.connect('garbage_classifier.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO classification_history 
            (timestamp, image_path, predicted_class, confidence, all_probabilities, user_ip)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            str(image_path),
            predicted_class,
            confidence,
            json.dumps(all_probs),
            user_ip
        ))
        record_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return record_id, str(image_path)
    except Exception as e:
        print(f"保存记录失败: {e}")
        return None, None


def save_feedback(classification_id, feedback_type, correct_class, comment, user_ip):
    """保存用户反馈"""
    try:
        conn = sqlite3.connect('garbage_classifier.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO feedback 
            (classification_id, timestamp, feedback_type, correct_class, comment, user_ip)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            classification_id,
            datetime.now().isoformat(),
            feedback_type,
            correct_class,
            comment,
            user_ip
        ))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"保存反馈失败: {e}")
        return False


def get_history_records(limit=50):
    """获取历史记录"""
    try:
        conn = sqlite3.connect('garbage_classifier.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, timestamp, image_path, predicted_class, confidence, all_probabilities
            FROM classification_history
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))
        records = cursor.fetchall()
        conn.close()

        result = []
        for record in records:
            result.append({
                'id': record[0],
                'timestamp': record[1],
                'image_path': record[2],
                'predicted_class': record[3],
                'confidence': record[4],
                'all_probabilities': json.loads(record[5])
            })
        return result
    except Exception as e:
        print(f"获取历史记录失败: {e}")
        return []


def get_statistics():
    """获取统计数据"""
    try:
        conn = sqlite3.connect('garbage_classifier.db')
        cursor = conn.cursor()

        # 总分类次数
        cursor.execute('SELECT COUNT(*) FROM classification_history')
        total_count = cursor.fetchone()[0]

        # 各类别统计
        cursor.execute('''
            SELECT predicted_class, COUNT(*) as count
            FROM classification_history
            GROUP BY predicted_class
        ''')
        class_stats = {row[0]: row[1] for row in cursor.fetchall()}

        # 反馈统计
        cursor.execute('SELECT COUNT(*) FROM feedback WHERE feedback_type = "correct"')
        correct_feedback = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM feedback WHERE feedback_type = "incorrect"')
        incorrect_feedback = cursor.fetchone()[0]

        conn.close()

        return {
            'total_classifications': total_count,
            'class_distribution': class_stats,
            'correct_feedback': correct_feedback,
            'incorrect_feedback': incorrect_feedback
        }
    except Exception as e:
        print(f"获取统计数据失败: {e}")
        return None


# ==================== 模型定义（与训练代码相同）====================
# ---------- BasicBlock ----------
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


# ---------- Bottleneck ----------
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


# ---------- ResNet ----------
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=4):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], 1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], 2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], 2)
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


# ---------- VGG ----------
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
            conv2d = nn.Conv2d(in_channels, v, 3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def VGG16(num_classes=4):
    cfg = [64, 64, 'M', 128, 128, 'M',
           256, 256, 256, 'M',
           512, 512, 512, 'M',
           512, 512, 512, 'M']
    return VGG(make_vgg_layers(cfg, batch_norm=True), num_classes)


# ---------- MobileNetV2 ---------
class MobileNetV2Custom(nn.Module):
    def __init__(self, num_classes=4, feature_dim=512):
        super(MobileNetV2Custom, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(32, 16, 1)
        self.layer2 = self._make_layer(16, 32, 2)
        self.layer3 = self._make_layer(32, 64, 2)
        self.layer4 = self._make_layer(64, 64, 2)
        self.layer5 = self._make_layer(64, 128, 2)
        self.layer6 = self._make_layer(128, 256, 2)
        self.layer7 = self._make_layer(256, 512, 1)
        self.conv2 = nn.Conv2d(512, feature_dim, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(feature_dim)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(feature_dim, num_classes)

    def _make_layer(self, in_c, out_c, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_c, in_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_c, in_c, 3, stride, 1, groups=in_c, bias=False),
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_c)
        )

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, self.layer6, self.layer7]:
            x = layer(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        logits = self.fc(features)
        return logits, features


# ---------- get_model ----------
def get_model(model_name, num_classes=4):
    if model_name == 'resnet18':
        return ResNet18(num_classes)
    elif model_name == 'resnet50':
        return ResNet50(num_classes)
    elif model_name == 'vgg16':
        return VGG16(num_classes)
    elif model_name == 'mobilenetv2':
        return MobileNetV2Custom(num_classes)
    else:
        raise ValueError(f"不支持的模型: {model_name}")


# ==================== Flask应用初始化 ====================
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 最大16MB

# 类别映射
CLASS_NAMES = {
    0: '厨余垃圾',
    1: '可回收物',
    2: '其他垃圾',
    3: '有害垃圾'
}

# 类别描述和颜色
CLASS_INFO = {
    '厨余垃圾': {
        'color': '#8BC34A',
        'description': '易腐烂的生物质废弃物',
        'examples': '剩菜剩饭、瓜皮果核、花卉绿植等'
    },
    '可回收物': {
        'color': '#2196F3',
        'description': '适宜回收和资源利用的废弃物',
        'examples': '纸类、塑料、玻璃、金属、织物等'
    },
    '其他垃圾': {
        'color': '#9E9E9E',
        'description': '除有害垃圾、可回收物、厨余垃圾以外的其他生活废弃物',
        'examples': '污损纸张、破旧陶瓷、尘土、一次性餐具等'
    },
    '有害垃圾': {
        'color': '#F44336',
        'description': '对人体健康或自然环境造成危害的废弃物',
        'examples': '电池、灯管、药品、油漆、杀虫剂等'
    }
}

# 全局变量
model = None
device = None
transform = None


def load_model(model_path='best_model.pth'):
    """加载训练好的模型"""
    global model, device, transform

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 加载checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # 判断文件内容类型
    if 'model_state_dict' in checkpoint:
        print("检测到完整checkpoint文件")
        model_name = checkpoint.get('model_name', 'mobilenetv2')
        model = get_model(model_name, num_classes=4)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_acc = checkpoint.get('best_acc', None)
    else:
        print("检测到纯模型参数文件")
        model_name = 'mobilenetv2'
        model = get_model(model_name, num_classes=4)
        model.load_state_dict(checkpoint)
        best_acc = None

    model.to(device)
    model.eval()

    print(f'模型加载成功: {model_name}')
    if best_acc is not None:
        print(f'最佳准确率: {best_acc:.2f}%')

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def predict_image(image):
    """预测图像类别"""
    if model is None:
        raise Exception("模型未加载")

    # 预处理
    image = image.convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # 预测
    with torch.no_grad():
        logits, features = model(image_tensor)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    predicted_class = predicted.item()
    confidence_score = confidence.item() * 100

    # 获取所有类别的概率
    all_probs = probabilities[0].cpu().numpy()
    class_probs = {CLASS_NAMES[i]: float(all_probs[i] * 100) for i in range(4)}

    return {
        'class_id': predicted_class,
        'class_name': CLASS_NAMES[predicted_class],
        'confidence': confidence_score,
        'all_probabilities': class_probs,
        'class_info': CLASS_INFO[CLASS_NAMES[predicted_class]]
    }


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """预测接口"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': '未上传文件'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '未选择文件'}), 400

        # 读取图像
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # 预测
        result = predict_image(image)

        # 获取用户IP
        user_ip = request.remote_addr

        # 保存记录到数据库
        record_id, image_path = save_classification_record(
            image,
            result['class_name'],
            result['confidence'],
            result['all_probabilities'],
            user_ip
        )

        # 将图像转换为base64用于显示
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        result['image'] = f"data:image/jpeg;base64,{img_str}"
        result['record_id'] = record_id

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """提交反馈接口"""
    try:
        data = request.get_json()
        classification_id = data.get('classification_id')
        feedback_type = data.get('feedback_type')  # 'correct' or 'incorrect'
        correct_class = data.get('correct_class')  # 如果incorrect，用户选择的正确类别
        comment = data.get('comment', '')

        user_ip = request.remote_addr

        success = save_feedback(
            classification_id,
            feedback_type,
            correct_class,
            comment,
            user_ip
        )

        if success:
            return jsonify({'status': 'success', 'message': '感谢您的反馈！'})
        else:
            return jsonify({'status': 'error', 'message': '反馈提交失败'}), 500

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/history')
def history():
    """历史记录页面"""
    return render_template('history.html')


@app.route('/api/history', methods=['GET'])
def get_history():
    """获取历史记录API"""
    try:
        limit = request.args.get('limit', 50, type=int)
        records = get_history_records(limit)
        return jsonify({'status': 'success', 'data': records})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/statistics', methods=['GET'])
def get_stats():
    """获取统计数据API"""
    try:
        stats = get_statistics()
        if stats:
            return jsonify({'status': 'success', 'data': stats})
        else:
            return jsonify({'status': 'error', 'message': '获取统计数据失败'}), 500
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    # 创建必要文件夹
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/history_images', exist_ok=True)

    # 初始化数据库
    init_database()

    # 加载模型
    model_path = 'best_model.pth'
    if not os.path.exists(model_path):
        print(f'错误: 找不到模型文件 {model_path}')
        print('请确保模型文件存在，或修改model_path变量')
        exit(1)

    load_model(model_path)

    # 启动Flask应用
    print('\n' + '=' * 50)
    print('垃圾分类识别系统已启动!')
    print('请在浏览器中访问: http://localhost:5000')
    print('历史记录页面: http://localhost:5000/history')
    print('=' * 50 + '\n')

    app.run(debug=True, host='0.0.0.0', port=5000)
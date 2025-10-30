明白了，我给你把整个 README 全部内容整理成完整 Markdown 格式，包括**功能说明、项目结构、依赖、使用方法、API、数据库结构、模型说明和注意事项**，可以直接用作项目文档。下面是完整版本：

```markdown
# 垃圾分类识别系统

基于 **PyTorch + Flask** 的垃圾分类识别 Web 应用，支持图片上传、实时分类、历史记录查看及用户反馈收集。系统集成了多种 CNN 模型（ResNet、VGG、MobileNetV2）并使用 SQLite 进行数据存储。

---

## 功能

1. **垃圾图片上传分类**
   - 支持用户上传图片，系统返回预测类别及置信度。
   - 显示类别描述、示例及颜色标识。

2. **历史记录管理**
   - 自动保存每次分类结果，包括图片路径、预测类别、置信度及所有类别概率。
   - 提供历史记录查询接口及网页显示。

3. **用户反馈收集**
   - 用户可以对分类结果进行反馈（正确或错误）。
   - 保存反馈信息用于后续模型优化。

4. **统计信息**
   - 提供总分类次数、各类别分布、反馈统计等数据。

---

## 项目结构

```

garbage_classification/
│
├─ app.py                  # Flask主应用文件
├─ best_model.pth          # 训练好的PyTorch模型
├─ garbage_classifier.db   # SQLite数据库（运行时生成）
├─ templates/
│   ├─ index.html          # 主页面模板
│   └─ history.html        # 历史记录页面模板
├─ static/
│   └─ history_images/     # 保存历史上传图片
└─ README.md



---

## 使用方法

1. **准备模型**

将训练好的模型文件 `best_model.pth` 放置在项目根目录。支持的模型：

* ResNet18
* ResNet50
* VGG16
* MobileNetV2（默认）

2. **启动 Flask 应用**

```bash
python app.py
```

系统将自动：

* 创建数据库 `garbage_classifier.db` 并初始化表结构
* 创建历史图片文件夹 `static/history_images/`
* 加载训练好的模型
* 启动 Web 服务（默认端口 5000）

3. **访问网页**

* 分类主页面: [http://localhost:5000](http://localhost:5000)
* 历史记录页面: [http://localhost:5000/history](http://localhost:5000)

---

## API接口说明

### 1. 图片预测

* **URL**: `/predict`
* **方法**: POST
* **参数**: 文件参数 `file`（图片）
* **返回**:

```json
{
  "class_id": 1,
  "class_name": "可回收物",
  "confidence": 95.23,
  "all_probabilities": {
    "厨余垃圾": 2.3,
    "可回收物": 95.23,
    "其他垃圾": 1.2,
    "有害垃圾": 1.27
  },
  "class_info": {
    "color": "#2196F3",
    "description": "适宜回收和资源利用的废弃物",
    "examples": "纸类、塑料、玻璃、金属、织物等"
  },
  "image": "data:image/jpeg;base64,...",
  "record_id": 12
}
```

---

### 2. 提交反馈

* **URL**: `/feedback`
* **方法**: POST
* **参数**（JSON）:

```json
{
  "classification_id": 12,
  "feedback_type": "correct",
  "correct_class": "其他垃圾",
  "comment": "分类错误示例"
}
```

* **返回**:

```json
{
  "status": "success",
  "message": "感谢您的反馈！"
}
```

---

### 3. 获取历史记录

* **URL**: `/api/history`
* **方法**: GET
* **参数**: `limit`（可选，默认 50）
* **返回**:

```json
{
  "status": "success",
  "data": [
    {
      "id": 12,
      "timestamp": "2025-10-30T10:00:00",
      "image_path": "static/history_images/20251030_100000_123456.jpg",
      "predicted_class": "可回收物",
      "confidence": 95.23,
      "all_probabilities": { ... }
    }
  ]
}
```

---

### 4. 获取统计信息

* **URL**: `/api/statistics`
* **方法**: GET
* **返回**:

```json
{
  "status": "success",
  "data": {
    "total_classifications": 150,
    "class_distribution": {
      "厨余垃圾": 50,
      "可回收物": 70,
      "其他垃圾": 20,
      "有害垃圾": 10
    },
    "correct_feedback": 120,
    "incorrect_feedback": 5
  }
}
```

---

## 数据库结构

### `classification_history`

| 字段                | 类型      | 描述              |
| ----------------- | ------- | --------------- |
| id                | INTEGER | 主键              |
| timestamp         | TEXT    | 分类时间            |
| image_path        | TEXT    | 图像路径            |
| predicted_class   | TEXT    | 模型预测类别          |
| confidence        | REAL    | 分类置信度           |
| all_probabilities | TEXT    | 所有类别概率（JSON字符串） |
| user_ip           | TEXT    | 用户IP            |

### `feedback`

| 字段                | 类型      | 描述                      |
| ----------------- | ------- | ----------------------- |
| id                | INTEGER | 主键                      |
| classification_id | INTEGER | 对应的分类记录ID               |
| timestamp         | TEXT    | 反馈时间                    |
| feedback_type     | TEXT    | 'correct' 或 'incorrect' |
| correct_class     | TEXT    | 用户提供的正确类别               |
| comment           | TEXT    | 用户评论                    |
| user_ip           | TEXT    | 用户IP                    |

---

## 模型与图像预处理

* 输入图像大小：128×128
* 归一化参数：mean=[0.485,0.456,0.406]，std=[0.229,0.224,0.225]
* 输出类别：

| 类别编号 | 类别名称 |
| ---- | ---- |
| 0    | 厨余垃圾 |
| 1    | 可回收物 |
| 2    | 其他垃圾 |
| 3    | 有害垃圾 |

* 可替换模型：`resnet18` / `resnet50` / `vgg16` / `mobilenetv2`

---

## 注意事项

1. **模型文件**必须存在，否则应用无法启动。
2. 支持最大上传文件大小 **16MB**。
3. 图片将自动保存到 `static/history_images/`，用于历史记录展示。
4. 推荐在 GPU 环境下运行以加速推理。

```

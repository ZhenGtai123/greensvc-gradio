# 城市绿地空间视觉分析系统 - 完整使用指南

## 📋 目录

1. [系统概述](#系统概述)
2. [快速部署指南](#快速部署指南)
3. [AutoDL部署说明](#autodl部署说明)
4. [详细功能说明](#详细功能说明)
5. [指标开发指南](#指标开发指南)
6. [常见问题解答](#常见问题解答)

## 系统概述

本系统通过三个主要组件协同工作：

1. **Gradio主应用**：用户界面和工作流管理
2. **视觉模型API**（AutoDL部署）：深度学习模型处理
3. **指标推荐API**：基于OpenAI的智能推荐

### 系统特点

- ✅ 模块化设计，易于扩展
- ✅ 支持批量图片处理
- ✅ 自动GPS信息提取
- ✅ 可视化报告生成
- ✅ AI驱动的指标推荐

## 快速部署指南

### 方式1：使用启动器（推荐）

```bash
# 1. 初始化项目
python launch.py --setup

# 2. 配置环境变量
cp .env.template .env
# 编辑 .env 文件，填入API密钥

# 3. 启动系统
python launch.py

# 或者使用公共分享链接
python launch.py --share
```

### 方式2：手动启动

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 初始化
python setup.py

# 3. 启动各个组件
python metrics_recommender.py  # 终端1
python app.py                  # 终端2
```

### 方式3：Docker部署

```bash
# 1. 构建镜像
docker-compose build

# 2. 启动服务
docker-compose up
```

## AutoDL部署说明

### 1. 准备AutoDL环境

在AutoDL上创建实例，选择：
- GPU: RTX 3090 或更高
- 镜像: PyTorch 1.13 + CUDA 11.7
- 系统盘: 50GB以上

### 2. 上传代码和模型

```bash
# SSH连接到AutoDL
ssh root@<your-autodl-ip>

# 创建项目目录
mkdir /root/vision-model
cd /root/vision-model

# 上传文件
# - vision_model_api.py
# - lang_sam.py (如果是独立文件)
# - depth_anything_v2/ (深度估计模块)
```

### 3. 下载模型权重

```bash
# 创建模型目录
mkdir -p models/lang_sam
mkdir -p models/depth_anything_v2

# 下载SAM模型
wget -O models/lang_sam/sam_vit_h_4b8939.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# 下载深度估计模型
wget -O models/depth_anything_v2/depth_anything_v2_vitb.pth \
  https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth
```

### 4. 安装依赖并启动

```bash
# 安装依赖
pip install fastapi uvicorn opencv-python torch torchvision \
  transformers groundingdino-py segment-anything scipy

# 启动API
python vision_model_api.py
```

### 5. 配置端口转发

在AutoDL控制台设置端口转发：
- 内部端口: 8000
- 获取公网地址

## 详细功能说明

### 1. 指标推荐与选择

#### AI推荐
- 输入分析需求的自然语言描述
- 系统基于指标库智能推荐相关指标
- 支持中英文输入

#### 手动选择
- 浏览完整指标库
- 查看指标详细说明
- 多选支持

### 2. 图片处理

#### 自动处理流程
1. **尺寸标准化**：调整为800×600像素
2. **GPS提取**：读取EXIF信息
3. **格式转换**：统一为PNG格式

#### GPS功能
- 自动检测所有图片GPS信息
- 支持生成空间热力图
- 显示拍摄位置分布

### 3. 视觉分析配置

#### 语义类别设置
```
# 每行一个类别，支持同义词
Sky
Trees, Tree, Forest
Building, Buildings, Architecture
People, Person, Human, Pedestrian
Water, Lake, River, Pond
```

#### 参数说明
- **可数性**：1=可数（建筑、人），0=不可数（天空、水面）
- **开放度**：1=开放空间，0=封闭空间

### 4. 报告生成

#### 报告内容
- 📊 数据统计表格
- 📈 多维度可视化图表
- 🗺️ 空间分布热力图（需GPS）
- 🤖 AI智能分析文本
- 📄 PDF/HTML双格式

#### 图表类型
- 指标分布箱线图
- 相关性热力图
- 雷达图对比
- 时序变化图

## 指标开发指南

### 1. 添加新指标定义

编辑 `data/library_metrics.xlsx`，添加行：

| 字段 | 说明 | 示例 |
|------|------|------|
| metric name | 指标名称 | Green Space Ratio |
| Primary Category | 主类别 | Composition |
| Calculation Method | 计算方法 | 绿色像素/总像素 |
| Professional Interpretation | 专业解释 | 反映绿化覆盖率 |

### 2. 编写计算代码

创建 `data/metrics_code/Green_Space_Ratio.py`：

```python
def calculate(vision_result):
    """计算绿地率"""
    import cv2
    import numpy as np
    
    # 获取语义分割图
    semantic_map = vision_result['images']['semantic_map']
    
    # 解码图像
    nparr = np.frombuffer(semantic_map, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 识别绿色类别（假设类别2是草地）
    green_mask = (img[:,:,1] > 200)  # 简化示例
    
    # 计算比例
    green_ratio = np.sum(green_mask) / (img.shape[0] * img.shape[1])
    
    return float(green_ratio)
```

### 3. 测试新指标

```python
# 在Python环境中测试
from modules.metrics_calculator import MetricsCalculator

calc = MetricsCalculator('data/metrics_code')
test_result = calc.validate_calculation('Green Space Ratio')
print(test_result)
```

## 常见问题解答

### Q1: AutoDL连接失败怎么办？

**解决方案**：
1. 检查端口转发设置
2. 确认防火墙规则
3. 验证API地址格式：`http://公网IP:端口`

### Q2: GPU内存不足？

**解决方案**：
1. 减少批处理数量
2. 使用更小的模型（vits而非vitb）
3. 升级AutoDL实例

### Q3: 指标计算错误？

**调试步骤**：
1. 检查输入图像格式
2. 验证代码语法
3. 查看日志文件
4. 使用测试数据验证

### Q4: 如何批量处理大量图片？

**建议**：
1. 分批处理（每批10-20张）
2. 使用脚本自动化
3. 考虑并行处理

### Q5: 报告生成失败？

**检查项**：
1. 是否有计算结果
2. 磁盘空间是否充足
3. 依赖库是否完整

## 进阶技巧

### 1. 自定义颜色方案

修改 `modules/api_clients.py` 中的颜色配置：

```python
'semantic_colors': {
    "1": [0, 255, 0],    # 绿色
    "2": [0, 0, 255],    # 蓝色
    # ...
}
```

### 2. 批处理脚本

```python
import os
from pathlib import Path

# 批量处理目录下所有图片
image_dir = Path("./images")
for img_file in image_dir.glob("*.jpg"):
    # 处理逻辑
    pass
```

### 3. 结果导出

所有分析结果保存在 `outputs/` 目录：
- `metrics_results.csv` - 原始数据
- `analysis_report.pdf` - 完整报告
- 各类图表文件

## 技术支持

- 📧 邮箱：[您的邮箱]
- 💬 微信：[您的微信]
- 🐛 问题反馈：[GitHub Issues]

---

祝您使用愉快！如有任何问题，请随时联系技术支持。
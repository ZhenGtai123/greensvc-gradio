# 🌳 Greenspace Vision - Frontend

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

基于 Gradio 的城市绿地空间视觉分析前端应用，通过调用 Vision API 实现语义分割、深度估计和前中背景分析，为城市绿地规划与设计提供数据驱动的决策支持。

## 🔗 相关项目

| 项目 | 说明 | 链接 |
|------|------|------|
| **Greenspace Vision API** | 后端 API 服务（必需） | [GitHub](https://github.com/ZhenGtai123/greensvc_vision) |

> ⚠️ **重要**：本前端项目需要配合后端 Vision API 使用。请先部署后端服务。

## ✨ 功能特性

### 🎯 视觉分析
- **语义分割**：支持 41 类园林要素识别（天空、草地、树木、建筑、水体等）
- **深度估计**：基于 Depth Anything V2 的单目深度估计
- **前中背景分割**：智能 FMB（Foreground-Middleground-Background）分层算法
- **GPS 信息提取**：自动从图片 EXIF 中提取地理位置
- **自定义配置**：支持自定义语义类别、颜色映射

### 📊 指标系统
- **AI 指标推荐**：基于需求描述智能推荐相关指标（需 OpenAI API）
- **指标库管理**：Excel 格式的指标库，支持自定义扩展
- **自定义计算代码**：为每个指标上传 Python 计算代码
- **批量计算**：支持多张图片批量分析

### 📄 报告生成
- 自动生成分析报告
- 支持 AI 辅助的专业解读
- 空间热力图（需要 GPS 数据）

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│              Gradio Frontend (本项目)                        │
├─────────────────────────────────────────────────────────────┤
│  API配置 │ 指标推荐 │ 指标管理 │ 视觉分析 │ 报告生成        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼ HTTP API
┌─────────────────────────────────────────────────────────────┐
│              Vision API (后端项目)                           │
│         支持 Google Colab / 本地 GPU / 云服务器              │
│                                                             │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │  Lang-SAM   │ │DepthAnything│ │  FMB 算法   │           │
│  │ (语义分割)  │ │ (深度估计)  │ │(前中背景)   │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

## 📁 项目结构

```
greenspace-frontend/
├── app.py                    # 主应用入口
├── launch.py                 # 系统启动器
├── requirements.txt          # Python 依赖
├── .env.example              # 环境变量模板
├── .gitignore                # Git 忽略规则
├── README.md
│
├── ui/                       # UI 模块
│   ├── __init__.py
│   ├── base.py              # 主界面布局
│   ├── state.py             # 应用状态管理
│   └── tabs/                # 标签页模块
│       ├── __init__.py
│       ├── api_config.py           # API 配置页
│       ├── metrics_recommendation.py # 指标推荐页
│       ├── metrics_management.py    # 指标管理页
│       ├── vision_analysis.py       # 视觉分析页
│       └── metrics_report.py        # 报告生成页
│
├── modules/                  # 业务模块
│   ├── api_clients.py       # API 客户端
│   ├── metrics_manager.py   # 指标库管理
│   ├── metrics_calculator.py # 指标计算
│   ├── metrics_recommender.py # 指标推荐
│   └── report_generator.py  # 报告生成
│
├── data/                     # 数据目录
│   ├── library_metrics.xlsx # 指标库文件
│   └── metrics_code/        # 指标计算代码
│
├── outputs/                  # 输出目录（报告等）
└── temp/                     # 临时文件目录
```

## 🚀 快速开始

### 前置条件

- Python 3.8+
- 运行中的 Vision API 后端

### 1. 部署后端 API（必需）

首先需要部署 Vision API 后端服务，支持以下方式：

| 部署方式 | 适用场景 | 说明 |
|---------|---------|------|
| **Google Colab** | 快速体验、无本地 GPU | 免费 T4 GPU，通过 ngrok 暴露服务 |
| **本地 GPU** | 开发调试、稳定使用 | 需要 NVIDIA GPU (≥8GB 显存) |
| **云服务器** | 生产部署 | AWS/GCP/Azure 或国内云平台 |

👉 详见 [Greenspace Vision API 文档](https://github.com/ZhenGtai123/greensvc_vision.git)

### 2. 克隆前端项目

```bash
git clone https://github.com/ZhenGtai123/greensvc-gradio.git
```

### 3. 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 4. 配置环境变量

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件
nano .env  # 或使用其他编辑器
```

关键配置项：

```env
# Vision API 地址（必填）
# - Colab 部署：填入 ngrok URL，如 https://xxxx.ngrok-free.app
# - 本地部署：填入 http://localhost:8000
VISION_API_URL=https://xxxx.ngrok-free.app

# OpenAI API Key（可选，用于 AI 指标推荐功能）
OPENAI_API_KEY=sk-your-api-key
```

> ⚠️ **安全提示**：`.env` 文件包含敏感信息，已在 `.gitignore` 中排除，不会被提交到 Git。

### 5. 启动前端应用

```bash
# 方式一：使用启动器（推荐）
python launch.py

# 方式二：启动并创建公共分享链接
python launch.py --share

# 方式三：直接运行
python app.py
```

### 6. 访问应用

打开浏览器访问：`http://localhost:7860`

## 📖 使用指南

### Tab 1：⚙️ API 配置

1. 输入 Vision API 的 URL（从后端部署获取）
2. 点击"🔌 连接"测试连接
3. 确认状态显示 ✅ API 正常

### Tab 2：指标推荐与选择

1. 输入分析需求描述
   - 例如："分析公园的开放度和视觉层次"
2. 输入 OpenAI API Key（可选）
3. 点击"获取 AI 推荐"
4. 或直接从指标库中勾选需要的指标
5. 点击"确认选择"

### Tab 3：指标库管理

1. **更新指标库**：上传新的 Excel 文件
2. **上传指标代码**：
   - 选择指标名称
   - 上传对应的 Python 计算代码
3. **查看状态**：确认各指标的代码上传状态

### Tab 4：视觉分析

1. **上传图片**
   - 支持多选上传
   - 自动提取并显示 GPS 信息

2. **选择配置模式**
   - **默认模式**：使用后端 41 类园林配置（推荐）
   - **自定义模式**：勾选"使用自定义配置"后可配置：
     - 预设配置选择
     - 语义类别编辑
     - 可数性/开放度参数
     - 颜色配置

3. **设置分析参数**
   - 模型大小：标准(vitb) / 轻量(vits)
   - 智能空洞填充：推荐 ✅
   - 中值滤波平滑：推荐 ✅

4. 点击"🚀 开始分析"

### Tab 5：指标计算与报告

1. 点击"Calculate Metrics"计算已选指标
2. 查看计算结果表格
3. 勾选是否包含空间热力图（需要 GPS 数据）
4. 点击"Generate Report"生成报告

## 🎨 自定义配置

### 预设配置

| 配置名称 | 类别数 | 说明 |
|---------|-------|------|
| 默认配置（41类园林） | 41 | 完整的园林景观要素分类 |
| 简单配置（8类） | 8 | 基础分类（天空、草地、树木等） |

### 自定义语义类别

1. 在"📋 预设与自定义配置"中编辑类别列表
2. 每行一个类别名称
3. 设置对应的可数性和开放度参数（逗号分隔的 0/1）

### 自定义颜色

1. 点击"生成颜色配置"
2. 编辑颜色（格式：`类别名=#RRGGBB`）
3. 点击"应用颜色"

## 📝 添加自定义指标

### 1. 创建计算代码

创建 Python 文件，必须包含 `calculate(vision_result)` 函数：

```python
"""
指标名称: 前景面积占比
"""
import cv2
import numpy as np

def calculate(vision_result):
    """
    计算前景面积占比
    
    Args:
        vision_result: dict, 包含:
            - status: 'success' 或 'error'
            - images: dict, 各类型图像数据
            - statistics: dict, 统计信息
    
    Returns:
        float: 指标值，失败返回 None
    """
    # 验证输入
    if vision_result.get('status') != 'success':
        return None
    
    images = vision_result.get('images', {})
    
    # 获取前中背景图
    if 'fmb_map' not in images:
        return None
    
    # 解码图像
    img_data = images['fmb_map']
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        return None
    
    # 计算前景占比（fmb_map 中前景值为 0）
    foreground_ratio = np.sum(img == 0) / img.size
    
    return float(foreground_ratio)
```

### 2. 上传代码

在"指标库管理"页面：
1. 从下拉列表选择对应指标
2. 上传 Python 文件
3. 确认状态显示 ✓

### 可用图像类型

| 图像键名 | 说明 | 数据格式 |
|---------|------|---------|
| `semantic_map` | 语义分割彩色图 | BGR |
| `depth_map` | 深度估计图 | 灰度 |
| `fmb_map` | 前中背景图 | 灰度 (0/1/2) |
| `openness_map` | 开放度图 | BGR |
| `foreground_map` | 前景掩码 | 二值 |
| `middleground_map` | 中景掩码 | 二值 |
| `background_map` | 背景掩码 | 二值 |
| `original` | 原图 | BGR |
| `semantic_foreground` | 语义图-前景 | BGR |
| `semantic_middleground` | 语义图-中景 | BGR |
| `semantic_background` | 语义图-背景 | BGR |
| `depth_foreground` | 深度图-前景 | 灰度 |
| `depth_middleground` | 深度图-中景 | 灰度 |
| `depth_background` | 深度图-背景 | 灰度 |

## 🐛 常见问题

### Q: API 连接失败？

**检查步骤：**
1. 确认后端 Vision API 正在运行
2. 检查 URL 格式是否正确（需包含 `https://` 或 `http://`）
3. 如使用 ngrok，确认隧道仍然有效
4. 尝试在浏览器直接访问 `{API_URL}/health`

**常见原因：**
- Colab 运行时断开
- ngrok 隧道过期
- 防火墙阻止连接

### Q: 分析速度很慢？

**优化建议：**
- 使用"轻量"模型 (vits) 代替"标准" (vitb)
- 确保后端使用 GPU 运行
- 减少单次分析的图片数量
- 检查网络连接质量

### Q: 指标计算返回 N/A？

**排查方法：**
1. 确认指标代码已正确上传（状态显示 ✓）
2. 检查代码中访问的图像类型是否存在
3. 查看浏览器控制台或终端日志
4. 确认视觉分析已成功完成

### Q: GPS 信息未提取？

**可能原因：**
- 图片不包含 EXIF GPS 数据
- 拍摄时未开启位置服务
- 图片经过编辑软件处理后 EXIF 被清除

### Q: 如何更新后端 API 地址？

在"⚙️ API 配置"页面重新输入新地址并点击连接，或修改 `.env` 文件后重启应用。

## 🔒 安全说明

- ✅ `.env` 文件已在 `.gitignore` 中排除
- ✅ 不要在代码中硬编码 API Key
- ✅ OpenAI API Key 仅用于指标推荐功能（可选）
- ✅ 建议定期更换敏感凭证

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request

## 📧 联系方式

如有问题或建议，请：
- 提交 [GitHub Issue](https://github.com/ZhenGtai123/greensvc-gradio/issues)
- 联系：1755061678@qq.com
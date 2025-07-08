# 城市绿地空间视觉分析系统

基于AI与空间视觉指标相结合的专业分析工具，为城市绿地等空间的科学决策与设计优化提供数据驱动支持。

## 系统架构

```
城市绿地空间视觉分析系统
├── app.py                    # 主应用程序（Gradio界面）
├── modules/                  # 功能模块
│   ├── api_clients.py       # API客户端（视觉模型、指标推荐）
│   ├── image_processor.py   # 图像处理（GPS提取、尺寸调整）
│   ├── metrics_manager.py   # 指标库管理
│   ├── metrics_calculator.py # 指标计算引擎
│   └── report_generator.py  # 报告生成
├── data/                    # 数据目录
│   ├── library_metrics.xlsx # 指标库
│   └── metrics_code/       # 指标计算代码
├── outputs/                # 输出目录
├── requirements.txt        # 依赖列表
└── setup.py               # 设置脚本
```

## 快速开始

### 1. 环境准备

```bash
# 克隆或下载项目
git clone <repository-url>
cd spatial-analysis-system

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 初始化项目

```bash
# 运行设置脚本
python setup.py
```

这将创建必要的目录结构和示例文件。

### 3. 配置环境

复制环境变量模板并配置：

```bash
cp .env.template .env
```

编辑 `.env` 文件，填入：
- `OPENAI_API_KEY`: 您的OpenAI API密钥
- `VISION_API_URL`: AutoDL上部署的视觉模型地址
- `METRICS_API_URL`: 指标推荐API地址（通常为本地）

### 4. 部署视觉模型（AutoDL）

在AutoDL上部署 `vision_model_api.py`：

```bash
# 在AutoDL环境中
cd /your/project/path
python vision_model_api.py
```

确保模型文件已下载：
- `models/lang_sam/sam_vit_h_4b8939.pth`
- `models/depth_anything_v2/depth_anything_v2_vitb.pth`

### 5. 启动系统

```bash
# 启动指标推荐API（可选，如果需要AI推荐功能）
python metrics_recommender.py

# 启动主应用
python app.py
```

访问 `http://localhost:7860` 即可使用系统。

## 使用流程

### 步骤1：指标推荐与选择
1. 输入您的分析需求描述
2. 获取AI推荐的指标
3. 查看完整指标库
4. 选择要使用的指标

### 步骤2：图片上传与处理
1. 上传待分析的图片（支持批量）
2. 系统自动：
   - 调整图片尺寸为800×600像素
   - 提取GPS信息（如有）
   - 显示处理状态

### 步骤3：视觉分析
1. 配置语义分割参数：
   - 语义类别（支持多个同义词）
   - 可数性设置
   - 开放度设置
2. 运行视觉分析，获取：
   - 语义分割图
   - 深度图
   - 前中后景图
   - 开放度图

### 步骤4：指标计算
1. 基于视觉分析结果计算选定指标
2. 查看计算结果表格

### 步骤5：报告生成
1. 选择是否包含空间热力图（需要GPS信息）
2. 生成综合报告，包含：
   - 数据统计
   - 可视化图表
   - AI智能分析
   - PDF和HTML格式

## 指标管理

### 添加新指标

1. 在 `data/library_metrics.xlsx` 中添加指标定义
2. 创建对应的计算代码：

```python
# data/metrics_code/Your_Metric_Name.py
def calculate(vision_result):
    """计算您的指标"""
    # 从vision_result中获取需要的图像
    # 进行计算
    # 返回结果
    return metric_value
```

### 指标代码规范

- 函数名必须是 `calculate`、`calculate_metric`、`calc` 或 `main` 之一
- 参数为 `vision_result` 字典
- 返回数值结果

## 配置说明

### 语义类别配置

支持逗号分隔的同义词：
```
Trees, Tree, Forest
People, Person, Human, Pedestrian
```

### 可数性配置

- `1`: 可数物体（如建筑、人、车）
- `0`: 不可数物体（如天空、草地、水面）

### 开放度配置

- `1`: 开放空间
- `0`: 封闭空间

## API接口

### 视觉模型API

- 端点：`POST /analyze`
- 输入：图片文件 + 配置参数
- 输出：各类分析图像

### 指标推荐API

- 端点：`POST /recommend/simple`
- 输入：用户需求描述
- 输出：推荐的指标列表

## 常见问题

### Q: 如何处理没有GPS信息的图片？
A: 系统会正常处理，但无法生成空间热力图。

### Q: 支持哪些图片格式？
A: 支持常见格式：JPG、PNG、BMP等。

### Q: 如何自定义颜色方案？
A: 在 `modules/api_clients.py` 中修改 `default_colors` 配置。

### Q: 计算结果如何导出？
A: 报告包含：
- PDF格式的完整报告
- Excel格式的原始数据
- 所有生成的图表

## 故障排除

### 视觉模型连接失败
- 检查AutoDL服务是否运行
- 验证API地址是否正确
- 确认防火墙设置

### 内存不足
- 减少批处理图片数量
- 降低图片分辨率
- 增加系统内存

### GPU相关错误
- 确认AutoDL环境有GPU
- 检查CUDA版本兼容性

## 开发指南

### 添加新的分析模块

1. 在 `modules/` 下创建新模块
2. 实现标准接口
3. 在主应用中集成

### 扩展可视化功能

1. 在 `report_generator.py` 中添加新图表类型
2. 更新报告模板

## 许可证

[您的许可证信息]

## 联系方式

如有问题或建议，请联系：[您的联系方式]

## 更新日志

### v1.0.0
- 初始版本发布
- 支持基础的空间视觉分析
- 集成AI指标推荐
- 完整的报告生成功能
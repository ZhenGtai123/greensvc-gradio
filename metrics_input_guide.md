# 指标输入配置指南

## 概述

为了确保每个指标使用正确的输入图像，系统支持在指标库中配置 `required_images` 字段。这解决了指标计算可能使用错误图像的问题。

## 当前的工作原理

### 1. 指标计算流程

```
用户选择指标
    ↓
视觉分析生成多种图像
    ↓
指标代码从vision_result中选择图像  ← 问题：可能选错
    ↓
计算指标值
```

### 2. Vision Result 包含的图像

| 图像类型 | 说明 | 用途示例 |
|---------|------|---------|
| semantic_map | 语义分割图 | 识别不同类别的区域 |
| depth_map | 深度图 | 计算空间层次、距离 |
| fmb_map | 前中后景图 | 分析空间层次关系 |
| foreground_map | 前景掩码 | 计算前景占比、形状 |
| middleground_map | 中景掩码 | 中景分析 |
| background_map | 背景掩码 | 背景分析 |
| openness_map | 开放度图 | 空间开放性分析 |
| original | 原始图片 | 参考用 |

## 改进方案：添加 required_images

### 1. 在指标库中添加配置

更新 `library_metrics.xlsx`，添加 `required_images` 列：

| metric name | ... | required_images |
|------------|-----|-----------------|
| Shape Edge Regularity Index (S_ERI) | ... | fmb_map, foreground_map |
| Shape Edge Contrast Index (S_ECI) | ... | semantic_map |
| Size View Field Ratio (S_VFR) | ... | foreground_map |

### 2. 配置格式

#### 基本格式
```
# 单个图像
semantic_map

# 多个图像（逗号分隔）
semantic_map, depth_map

# 可选图像（竖线分隔，使用第一个可用的）
fmb_map|foreground_map
```

#### 示例配置

| 指标类型 | required_images | 说明 |
|---------|-----------------|------|
| 形状分析 | fmb_map, foreground_map | 需要前景轮廓 |
| 纹理分析 | semantic_map | 需要语义信息 |
| 深度分析 | depth_map | 需要深度信息 |
| 综合分析 | semantic_map, depth_map, fmb_map | 需要多种信息 |

### 3. 系统如何使用配置

改进后的 `MetricsCalculator` 会：

1. **读取配置**
   ```python
   required_images = metric_info['required_images']
   # 例如: ['semantic_map', 'depth_map']
   ```

2. **验证图像**
   ```python
   # 检查所需图像是否都存在
   validation = validate_required_images(vision_result, required_images)
   if not validation['valid']:
       print(f"缺少图像: {validation['missing']}")
       return None
   ```

3. **准备输入**
   ```python
   # 只传递所需的图像给指标代码
   simplified_input = {
       'images': {
           'semantic_map': vision_result['images']['semantic_map'],
           'depth_map': vision_result['images']['depth_map']
       }
   }
   ```

## 实施步骤

### 步骤1：更新指标库

运行更新脚本：
```bash
python update_metrics_library.py
```

这会：
- 添加 `required_images` 列
- 根据指标类型自动填充建议值
- 创建备份文件

### 步骤2：验证配置

检查更新后的指标库：
1. 打开 `data/library_metrics.xlsx`
2. 查看 `required_images` 列
3. 根据需要调整配置

### 步骤3：更新指标代码（可选）

在代码中明确声明所需图像：
```python
"""
指标名称: My Metric
required_images: ['semantic_map', 'depth_map']
"""

def calculate(vision_result):
    # 现在可以确信需要的图像存在
    semantic_map = vision_result['images']['semantic_map']
    depth_map = vision_result['images']['depth_map']
    # ...
```

## 优势

### 1. 验证机制
- 计算前检查所需图像是否存在
- 提供清晰的错误信息
- 避免运行时错误

### 2. 文档化
- 明确每个指标的输入需求
- 便于理解和维护
- 新用户容易上手

### 3. 性能优化
- 只传递必需的图像数据
- 减少内存使用
- 加快计算速度

### 4. 灵活性
- 支持多种配置格式
- 可以指定替代图像
- 向后兼容（不影响现有代码）

## 兼容性

### 现有代码仍能工作
如果没有配置 `required_images`，系统会：
1. 传递完整的 `vision_result`
2. 让指标代码自己选择图像
3. 保持向后兼容

### 逐步迁移
1. 先更新指标库配置
2. 测试确保正常工作
3. 逐步更新指标代码

## 最佳实践

### 1. 明确需求
```yaml
# 好的配置
required_images: semantic_map, depth_map

# 不好的配置
required_images: all  # 太模糊
```

### 2. 使用替代选项
```yaml
# 当多个图像都可以时
required_images: fmb_map|foreground_map
```

### 3. 最小化依赖
只声明真正需要的图像，不要"以防万一"地添加。

### 4. 测试验证
```python
# 测试指标是否能处理配置的图像
python test_metric_config.py "Shape Edge Regularity Index (S_ERI)"
```

## 故障排除

### 问题：指标计算返回None
检查：
1. required_images配置是否正确
2. 视觉分析是否生成了所需图像
3. 图像名称是否拼写正确

### 问题：性能下降
可能原因：
1. required_images包含了不必要的图像
2. 考虑优化配置，只保留必需的

### 问题：兼容性问题
解决方案：
1. 暂时清空required_images
2. 逐个指标测试和配置
3. 确保代码和配置匹配

---

通过这种配置方式，系统变得更加健壮和易于维护，同时保持了灵活性和向后兼容性。
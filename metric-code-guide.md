# 指标代码编写指南

## 概述

每个指标都需要一个对应的Python代码文件来计算其值。代码文件必须包含一个名为 `calculate` 的函数（或 `calculate_metric`、`calc`、`main` 之一）。

## 基本结构

```python
def calculate(vision_result):
    """
    计算指标的主函数
    
    参数:
        vision_result: 包含视觉分析结果的字典
    
    返回:
        float: 指标值
        或 None: 如果计算失败
    """
    # 您的计算逻辑
    return metric_value
```

## 重要：函数命名

系统会按以下顺序查找计算函数：
1. `calculate` （推荐）
2. `calculate_metric`
3. `calc`
4. `main`

**必须使用其中一个函数名，否则代码无法运行！**

## vision_result 数据结构

```python
vision_result = {
    'status': 'success',  # 或 'error'
    'images': {
        'semantic_map': bytes,      # 语义分割图
        'depth_map': bytes,         # 深度图
        'fmb_map': bytes,          # 前中后景图（0=前景, 1=中景, 2=背景）
        'foreground_map': bytes,    # 前景掩码（255=前景）
        'middleground_map': bytes,  # 中景掩码
        'background_map': bytes,    # 背景掩码
        'openness_map': bytes,      # 开放度图
        'original': bytes,          # 调整尺寸后的原图
        # 可能还有其他图像...
    },
    'image_path': str  # 原始图片路径
}
```

## 图像数据处理

所有图像数据都以字节形式存储，需要转换为OpenCV格式：

```python
import cv2
import numpy as np

# 从字节转换为图像
img_bytes = vision_result['images']['semantic_map']
nparr = np.frombuffer(img_bytes, np.uint8)
img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

# 转换为灰度图（如需要）
if len(img.shape) == 3:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
else:
    gray = img
```

## 常用图像说明

### 1. 语义分割图 (semantic_map)
- 彩色图像，每种颜色代表一个语义类别
- 使用配置的颜色方案

### 2. 深度图 (depth_map)
- 灰度图，值越大表示距离越远
- 值范围：0-255

### 3. 前中后景图 (fmb_map)
- 灰度图，像素值含义：
  - 0 = 前景
  - 1 = 中景
  - 2 = 背景

### 4. 前景掩码 (foreground_map)
- 二值图，255 = 前景，0 = 非前景

### 5. 开放度图 (openness_map)
- 彩色图，显示空间开放度

## 示例代码

### 示例1：计算前景面积比例

```python
def calculate(vision_result):
    """计算前景占视野的比例"""
    import cv2
    import numpy as np
    
    if 'images' not in vision_result:
        return None
    
    # 获取前景掩码
    foreground_data = vision_result['images'].get('foreground_map')
    if foreground_data is None:
        return None
    
    # 转换为图像
    nparr = np.frombuffer(foreground_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 计算比例
    foreground_pixels = np.sum(gray > 127)
    total_pixels = gray.shape[0] * gray.shape[1]
    
    ratio = foreground_pixels / total_pixels if total_pixels > 0 else 0
    return float(ratio)
```

### 示例2：计算形状复杂度

```python
def calculate(vision_result):
    """计算前景形状的复杂度"""
    import cv2
    import numpy as np
    
    if 'images' not in vision_result:
        return None
    
    # 获取前中后景图
    fmb_data = vision_result['images'].get('fmb_map')
    if fmb_data is None:
        return None
    
    # 转换并提取前景
    nparr = np.frombuffer(fmb_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 前景为值0的像素
    foreground = (gray == 0).astype(np.uint8) * 255
    
    # 查找轮廓
    contours, _ = cv2.findContours(
        foreground, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        return 0.0
    
    # 计算最大轮廓的复杂度
    max_contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(max_contour, True)
    area = cv2.contourArea(max_contour)
    
    if area == 0:
        return 0.0
    
    # 形状复杂度指数
    complexity = perimeter / (2 * np.sqrt(np.pi * area))
    return float(complexity)
```

### 示例3：计算深度变化率

```python
def calculate(vision_result):
    """计算场景深度变化率"""
    import cv2
    import numpy as np
    
    if 'images' not in vision_result:
        return None
    
    # 获取深度图
    depth_data = vision_result['images'].get('depth_map')
    if depth_data is None:
        return None
    
    # 转换为图像
    nparr = np.frombuffer(depth_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    # 计算深度梯度
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    # 计算梯度幅值
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # 平均变化率
    avg_change_rate = np.mean(gradient_magnitude)
    
    return float(avg_change_rate)
```

## 最佳实践

1. **错误处理**：始终检查输入数据是否存在
2. **类型转换**：确保返回值是float类型
3. **异常捕获**：使用try-except处理可能的错误
4. **性能优化**：避免不必要的计算
5. **文档说明**：添加清晰的注释

## 调试技巧

1. **打印中间结果**
```python
print(f"图像形状: {img.shape}")
print(f"前景像素数: {np.sum(foreground > 0)}")
```

2. **保存中间图像**
```python
cv2.imwrite('debug_foreground.png', foreground)
```

3. **测试代码**
```python
# 在代码文件末尾添加测试
if __name__ == "__main__":
    # 创建模拟数据
    test_result = {
        'status': 'success',
        'images': {
            'foreground_map': cv2.imencode('.png', test_img)[1].tobytes()
        }
    }
    result = calculate(test_result)
    print(f"测试结果: {result}")
```

## 常见问题

### Q: 如何处理多个前景对象？
A: 使用cv2.findContours找到所有轮廓，然后分别处理或统计。

### Q: 如何获取特定语义类别的区域？
A: 需要知道颜色映射，然后使用颜色掩码提取特定区域。

### Q: 计算结果为None怎么办？
A: 检查输入数据是否完整，添加更多调试信息。

### Q: 如何使用其他库？
A: 可以导入标准库和已安装的包（numpy, scipy, sklearn等）。

## 上传代码

1. 将代码保存为 `.py` 文件
2. 文件名建议与指标名称对应
3. 在界面的"指标库管理"标签页上传
4. 上传后会自动更新指标状态

---

需要更多帮助？查看 `example_metric_code.py` 获取完整示例。
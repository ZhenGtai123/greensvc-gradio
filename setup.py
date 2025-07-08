"""
项目设置脚本
创建必要的目录结构和初始文件
"""

import os
import json
import pandas as pd
import shutil

def create_directory_structure():
    """创建项目目录结构"""
    directories = [
        'data',
        'data/metrics_code',
        'outputs',
        'temp',
        'modules'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ 创建目录: {directory}")

def create_sample_metrics_library():
    """创建示例指标库"""
    # 示例指标数据
    metrics_data = [
        {
            "metric name": "Shape Edge Regularity Index (S_ERI)",
            "Primary Category": "Composition/Configuration",
            "Secondary Attribute": "Shape",
            "Classification Rationale": "Evaluates the regularity of the foreground element's boundary",
            "Standard Range": "[π/2, +∞)",
            "Unit": "Dimensionless",
            "Parameter Definition": "Based on isoperimetric inequality",
            "Data Input": "Segmentation Image",
            "Calculation Method": "S_ERI = 0.25 * (P / A)",
            "Professional Interpretation": "High regularity aids recognition and sense of order"
        },
        {
            "metric name": "Shape Edge Contrast Index (S_ECI)",
            "Primary Category": "Composition/Configuration",
            "Secondary Attribute": "Shape",
            "Classification Rationale": "Reflects the difference between the foreground boundary and surrounding",
            "Standard Range": "[0,1]",
            "Unit": "Dimensionless",
            "Parameter Definition": "Measures label consistency of pixels",
            "Data Input": "Segmentation Image",
            "Calculation Method": "S_ECI = Py / (Py + Pb)",
            "Professional Interpretation": "High contrast enhances spatial independence"
        },
        {
            "metric name": "Size View Field Ratio (S_VFR)",
            "Primary Category": "Composition/Configuration",
            "Secondary Attribute": "Size",
            "Classification Rationale": "Ratio of foreground area in the field of view",
            "Standard Range": "[0,1]",
            "Unit": "Dimensionless",
            "Parameter Definition": "Ratio of foreground pixels to total pixels",
            "Data Input": "Segmentation Image",
            "Calculation Method": "S_VFR = Ai/Aj",
            "Professional Interpretation": "High foreground ratio enhances enclosure"
        }
    ]
    
    # 保存为Excel文件
    df = pd.DataFrame(metrics_data)
    df.to_excel('data/library_metrics.xlsx', index=False)
    print("✓ 创建示例指标库: data/library_metrics.xlsx")

def create_sample_metric_code():
    """创建示例指标计算代码"""
    
    # S_ERI计算代码
    seri_code = '''"""
指标计算代码: Shape Edge Regularity Index (S_ERI)
"""

def calculate(vision_result):
    """计算S_ERI指标"""
    import cv2
    import numpy as np
    
    # 获取前景掩码
    if 'images' not in vision_result:
        return None
    
    fmb_map_data = vision_result['images'].get('fmb_map')
    if fmb_map_data is None:
        return None
    
    # 将字节数据转换为图像
    nparr = np.frombuffer(fmb_map_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 提取前景
    foreground = (gray == 0).astype(np.uint8) * 255
    
    # 查找轮廓
    contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0.0
    
    # 获取最大轮廓
    max_contour = max(contours, key=cv2.contourArea)
    
    # 计算周长和面积
    perimeter = cv2.arcLength(max_contour, True)
    area = cv2.contourArea(max_contour)
    
    if area == 0:
        return 0.0
    
    # 计算S_ERI
    seri = 0.25 * perimeter / np.sqrt(area)
    
    return float(seri)
'''
    
    with open('data/metrics_code/Shape_Edge_Regularity_Index_(S_ERI).py', 'w', encoding='utf-8') as f:
        f.write(seri_code)
    
    # S_VFR计算代码
    svfr_code = '''"""
指标计算代码: Size View Field Ratio (S_VFR)
"""

def calculate(vision_result):
    """计算S_VFR指标"""
    import cv2
    import numpy as np
    
    # 获取前景掩码
    if 'images' not in vision_result:
        return None
    
    foreground_data = vision_result['images'].get('foreground_map')
    if foreground_data is None:
        return None
    
    # 将字节数据转换为图像
    nparr = np.frombuffer(foreground_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 转换为灰度图
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # 计算前景像素比例
    foreground_pixels = np.sum(gray > 127)
    total_pixels = gray.shape[0] * gray.shape[1]
    
    if total_pixels == 0:
        return 0.0
    
    svfr = foreground_pixels / total_pixels
    
    return float(svfr)
'''
    
    with open('data/metrics_code/Size_View_Field_Ratio_(S_VFR).py', 'w', encoding='utf-8') as f:
        f.write(svfr_code)
    
    # 创建示例代码模板
    example_template = '''"""
指标计算代码模板
请根据您的指标修改此模板

指标名称: [您的指标名称]
功能: [指标功能描述]
"""

import cv2
import numpy as np

def calculate(vision_result):
    """
    计算指标的主函数
    
    参数:
        vision_result: 包含视觉分析结果的字典
            - images: 各种分析图像
                - semantic_map: 语义分割图
                - depth_map: 深度图
                - fmb_map: 前中后景图
                - foreground_map: 前景掩码
                - openness_map: 开放度图
    
    返回:
        float: 指标值，或 None（如果失败）
    """
    
    try:
        # 1. 检查输入
        if 'images' not in vision_result:
            return None
        
        # 2. 获取需要的图像
        img_data = vision_result['images'].get('semantic_map')  # 根据需要修改
        if img_data is None:
            return None
        
        # 3. 转换为OpenCV图像
        if isinstance(img_data, bytes):
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            img = img_data
        
        # 4. 在这里实现您的指标计算逻辑
        # ...
        
        # 5. 返回计算结果
        metric_value = 0.0  # 替换为实际计算值
        return float(metric_value)
        
    except Exception as e:
        print(f"计算指标时出错: {str(e)}")
        return None
'''
    
    with open('data/metrics_code/metric_template.py', 'w', encoding='utf-8') as f:
        f.write(example_template)
    
    print("✓ 创建示例指标计算代码")
    print("  - Shape_Edge_Regularity_Index_(S_ERI).py")
    print("  - Size_View_Field_Ratio_(S_VFR).py")
    print("  - metric_template.py (代码模板)")
    
    # 复制完整示例到项目根目录供参考
    import shutil
    shutil.copy('data/metrics_code/metric_template.py', 'example_metric_code.py')
    print("✓ 复制代码模板到: example_metric_code.py")

def create_config_file():
    """创建配置文件"""
    config = {
        "vision_api_url": "http://localhost:8000",  # 需要替换为实际的AutoDL URL
        "metrics_api_url": "http://localhost:8001",
        "default_semantic_classes": [
            "Sky",
            "Lawn, Grass, Grassland",
            "Trees, Tree",
            "Building, Buildings",
            "Water, River, Lake",
            "People, Person, Human",
            "Roads, Street",
            "Cars, Vehicles"
        ],
        "default_countability": "1,0,0,1,0,1,0,1",
        "default_openness": "1,1,0,0,1,0,1,0"
    }
    
    with open('config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print("✓ 创建配置文件: config.json")

def create_env_template():
    """创建环境变量模板"""
    env_content = """# OpenAI API配置
OPENAI_API_KEY=your-openai-api-key

# AutoDL视觉模型API地址
VISION_API_URL=http://your-autodl-url:8000

# 指标推荐API地址（本地运行）
METRICS_API_URL=http://localhost:8001
"""
    
    with open('.env.template', 'w') as f:
        f.write(env_content)
    
    print("✓ 创建环境变量模板: .env.template")

def main():
    """运行设置脚本"""
    print("=== 城市绿地空间视觉分析系统 - 项目设置 ===\n")
    
    # 创建目录结构
    print("1. 创建目录结构...")
    create_directory_structure()
    
    # 创建示例文件
    print("\n2. 创建示例文件...")
    create_sample_metrics_library()
    create_sample_metric_code()
    create_config_file()
    create_env_template()
    
    # 最终说明
    print("\n=== 设置完成 ===")
    print("\n接下来的步骤：")
    print("1. 安装依赖: pip install -r requirements.txt")
    print("2. 复制 .env.template 为 .env 并填写API密钥")
    print("3. 部署AutoDL视觉模型并更新配置中的URL")
    print("4. 运行指标推荐API: python metrics_recommender.py")
    print("5. 启动主应用: python app.py")
    print("\n提示：")
    print("- 请确保AutoDL上的视觉模型API已经运行")
    print("- 可以在 data/library_metrics.xlsx 中查看和编辑指标库")
    print("- 可以在 data/metrics_code/ 目录中添加更多指标计算代码")

if __name__ == "__main__":
    main()
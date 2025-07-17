"""
指标名称: Size View Field Ratio (S_VFR)
功能: 计算视野中前景所占的比例
required_images: ['foreground_map']
作者: 系统
更新日期: 2024
"""

import cv2
import numpy as np

def calculate(vision_result):
    """
    计算S_VFR指标 - 视野中前景区域的比例
    
    Args:
        vision_result: 视觉分析结果字典，包含images字段
        
    Returns:
        float: S_VFR值，范围[0, 1]，表示前景占总面积的比例
        None: 如果计算失败
    """
    try:
        # 验证输入
        if not isinstance(vision_result, dict) or 'images' not in vision_result:
            print("错误：无效的输入格式")
            return None
        
        # 获取前景掩码数据
        foreground_data = vision_result['images'].get('foreground_map')
        if foreground_data is None:
            print("错误：缺少 foreground_map 图像")
            return None
        
        # 将字节数据转换为图像
        if isinstance(foreground_data, bytes):
            nparr = np.frombuffer(foreground_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)  # 直接读取为灰度图
        elif isinstance(foreground_data, np.ndarray):
            img = foreground_data
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            print(f"错误：不支持的数据类型 {type(foreground_data)}")
            return None
        
        # 验证图像
        if img is None or img.size == 0:
            print("错误：无法解码图像")
            return None
        
        # 计算前景像素比例
        # 前景图中，白色(255)表示前景，黑色(0)表示背景
        foreground_pixels = np.sum(img > 127)  # 使用阈值127来区分前景和背景
        total_pixels = img.shape[0] * img.shape[1]
        
        if total_pixels == 0:
            return 0.0
        
        # 计算S_VFR（前景视野比）
        svfr = foreground_pixels / total_pixels
        
        # 确保结果在合理范围内
        svfr = max(0.0, min(1.0, svfr))
        
        return float(svfr)
        
    except Exception as e:
        print(f"计算S_VFR时发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# 用于测试的辅助函数
def validate_input(vision_result):
    """验证输入数据的完整性"""
    if not isinstance(vision_result, dict):
        return False, "输入必须是字典类型"
    
    if 'images' not in vision_result:
        return False, "缺少'images'字段"
    
    if 'foreground_map' not in vision_result['images']:
        return False, "缺少'foreground_map'图像"
    
    return True, "输入验证通过"

# 指标元数据
METRIC_INFO = {
    'name': 'Size View Field Ratio (S_VFR)',
    'category': 'Spatial Metrics',
    'required_images': ['foreground_map'],
    'output_range': [0.0, 1.0],
    'unit': 'ratio',
    'description': '衡量前景区域在整个视野中所占的比例'
}
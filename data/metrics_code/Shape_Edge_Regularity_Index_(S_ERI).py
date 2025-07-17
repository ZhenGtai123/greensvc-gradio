"""
指标名称: Shape Edge Regularity Index (S_ERI)
功能: 计算形状边缘规则性指数
required_images: ['fmb_map|foreground_map']
作者: 系统
更新日期: 2024
"""

import cv2
import numpy as np

def calculate(vision_result):
    """
    计算S_ERI指标 - 形状边缘规则性指数
    
    该指标通过计算形状的紧凑度来衡量边缘的规则性。
    值越接近1，形状越接近圆形（最规则）；值越大，形状越不规则。
    
    Args:
        vision_result: 视觉分析结果字典，包含images字段
        
    Returns:
        float: S_ERI值，通常在[1, ∞)范围内
        None: 如果计算失败
    """
    try:
        # 验证输入
        if not isinstance(vision_result, dict) or 'images' not in vision_result:
            print("错误：无效的输入格式")
            return None
        
        # 优先使用fmb_map，如果没有则使用foreground_map
        img_data = None
        use_fmb = False
        
        if 'fmb_map' in vision_result['images']:
            img_data = vision_result['images']['fmb_map']
            use_fmb = True
        elif 'foreground_map' in vision_result['images']:
            img_data = vision_result['images']['foreground_map']
            use_fmb = False
        else:
            print("错误：缺少 fmb_map 或 foreground_map 图像")
            return None
        
        # 将字节数据转换为图像
        if isinstance(img_data, bytes):
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif isinstance(img_data, np.ndarray):
            img = img_data.copy()
        else:
            print(f"错误：不支持的数据类型 {type(img_data)}")
            return None
        
        # 验证图像
        if img is None or img.size == 0:
            print("错误：无法解码图像")
            return None
        
        # 转换为灰度图
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # 提取前景
        if use_fmb:
            # FMB map中，前景值为0（深红色对应的值）
            # 需要先检查实际的值分布
            unique_values = np.unique(gray)
            if len(unique_values) > 0:
                # 假设最小值是前景
                foreground_value = unique_values[0]
                foreground = (gray == foreground_value).astype(np.uint8) * 255
            else:
                return 0.0
        else:
            # foreground_map中，白色(>127)表示前景
            foreground = (gray > 127).astype(np.uint8) * 255
        
        # 形态学操作，去除噪点
        kernel = np.ones((3, 3), np.uint8)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("警告：未找到前景轮廓")
            return 0.0
        
        # 计算所有轮廓的S_ERI，然后取加权平均（按面积加权）
        total_area = 0
        weighted_seri = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 10:  # 忽略太小的轮廓（可能是噪点）
                continue
            
            perimeter = cv2.arcLength(contour, True)
            
            if area > 0 and perimeter > 0:
                # 计算单个轮廓的S_ERI
                # 使用标准的形状规则性公式：周长²/(4π×面积)
                # 对于圆形，这个值为1；形状越不规则，值越大
                seri_single = (perimeter * perimeter) / (4 * np.pi * area)
                
                weighted_seri += seri_single * area
                total_area += area
        
        if total_area == 0:
            return 0.0
        
        # 计算加权平均S_ERI
        seri = weighted_seri / total_area
        
        # 确保结果在合理范围内（理论最小值为1）
        seri = max(1.0, seri)
        
        return float(seri)
        
    except Exception as e:
        print(f"计算S_ERI时发生错误: {str(e)}")
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
    
    if 'fmb_map' not in vision_result['images'] and 'foreground_map' not in vision_result['images']:
        return False, "缺少'fmb_map'或'foreground_map'图像"
    
    return True, "输入验证通过"

# 指标元数据
METRIC_INFO = {
    'name': 'Shape Edge Regularity Index (S_ERI)',
    'category': 'Shape Metrics',
    'required_images': ['fmb_map', 'foreground_map'],  # 可以使用任一
    'output_range': [1.0, float('inf')],
    'unit': 'index',
    'description': '衡量形状边缘的规则性，值越接近1表示形状越规则（接近圆形）'
}
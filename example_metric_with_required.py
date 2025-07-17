"""
指标名称: Green Space Coverage Index (GSCI)
功能: 计算绿色空间覆盖率
required_images: ['semantic_map']
"""

import cv2
import numpy as np

def calculate(vision_result):
    """
    计算绿色空间覆盖率
    
    Required: semantic_map
    """
    try:
        # 检查输入
        if 'images' not in vision_result:
            return None
        
        # 获取语义分割图
        semantic_data = vision_result['images'].get('semantic_map')
        if semantic_data is None:
            return None
        
        # 转换为图像
        nparr = np.frombuffer(semantic_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 假设绿色类别的颜色是 [4, 250, 7]（草地）
        green_color = np.array([4, 250, 7])
        
        # 创建掩码
        mask = cv2.inRange(img, green_color - 10, green_color + 10)
        
        # 计算绿色像素比例
        green_pixels = np.sum(mask > 0)
        total_pixels = mask.shape[0] * mask.shape[1]
        
        if total_pixels == 0:
            return 0.0
        
        coverage = green_pixels / total_pixels
        
        return float(coverage)
        
    except Exception as e:
        print(f"计算GSCI时出错: {e}")
        return None

"""
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

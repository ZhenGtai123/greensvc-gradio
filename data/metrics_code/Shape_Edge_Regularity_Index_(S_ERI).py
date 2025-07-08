"""
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

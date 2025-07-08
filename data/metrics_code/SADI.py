"""
指标计算代码: Spatial Area Dispersion Index (SADI)
"""

def calculate(vision_result):
    """计算 SADI 指标"""
    import cv2
    import numpy as np
    from skimage.measure import label

    # 获取前景掩码
    if 'images' not in vision_result:
        return None

    foreground_data = vision_result['images'].get('foreground_map')
    if foreground_data is None:
        return None

    # 字节数据转换为图像
    nparr = np.frombuffer(foreground_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)  # 直接读为灰度图

    if img is None or img.size == 0:
        return 0.0

    # 构建二值图（前景为0，背景为1）
    binary = (img == 0).astype(np.uint8)

    # 连通域标记
    labeled = label(binary, connectivity=2)

    # 获取各连通区域面积（排除背景0）
    region_sizes = np.bincount(labeled.ravel())[1:]

    if len(region_sizes) == 0:
        return 0.0

    # SADI = 面积方差 / 平均面积
    mean_area = np.mean(region_sizes)
    variance = np.var(region_sizes)

    sadi = variance / (mean_area + 1e-8)  # 避免除以0

    return float(sadi)

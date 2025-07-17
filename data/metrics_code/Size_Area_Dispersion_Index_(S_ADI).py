"""
指标名称: Spatial Area Dispersion Index (SADI)
功能: 计算空间区域分散度指数
required_images: ['foreground_map']
作者: 系统
更新日期: 2024
"""

import cv2
import numpy as np
from skimage.measure import label, regionprops

def calculate(vision_result):
    """
    计算 SADI 指标 - 空间区域分散度指数
    
    该指标衡量前景区域的分散程度。
    值越大，表示区域大小差异越大，空间分布越不均匀。
    
    Args:
        vision_result: 视觉分析结果字典，包含images字段
        
    Returns:
        float: SADI值，非负数
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
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)  # 直接读为灰度图
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
            return 0.0
        
        # 构建二值图（前景为1，背景为0）
        # 在foreground_map中，白色(>127)表示前景
        binary = (img > 127).astype(np.uint8)
        
        # 形态学操作，去除小噪点
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 连通域标记
        labeled = label(binary, connectivity=2)
        
        # 获取区域属性
        regions = regionprops(labeled)
        
        if len(regions) == 0:
            print("警告：未找到前景区域")
            return 0.0
        
        # 获取各连通区域面积（过滤掉太小的区域）
        min_area_threshold = 10  # 最小面积阈值
        region_areas = []
        
        for region in regions:
            if region.area >= min_area_threshold:
                region_areas.append(region.area)
        
        if len(region_areas) == 0:
            return 0.0
        
        region_areas = np.array(region_areas)
        
        # 计算SADI
        if len(region_areas) == 1:
            # 只有一个区域时，分散度为0
            sadi = 0.0
        else:
            # SADI = 标准差 / 平均面积
            # 这个比值反映了区域大小的相对分散程度
            mean_area = np.mean(region_areas)
            std_area = np.std(region_areas)
            
            sadi = std_area / (mean_area + 1e-8)  # 避免除以0
        
        # 确保结果非负
        sadi = max(0.0, sadi)
        
        # 可选：输出额外信息
        if len(region_areas) > 0:
            print(f"SADI计算完成: 找到{len(region_areas)}个区域, " + 
                  f"平均面积={np.mean(region_areas):.1f}, " +
                  f"标准差={np.std(region_areas):.1f}")
        
        return float(sadi)
        
    except Exception as e:
        print(f"计算SADI时发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# 扩展功能：计算更详细的分散度指标
def calculate_extended(vision_result):
    """
    计算扩展的分散度指标，包括：
    - SADI: 面积分散度
    - 区域数量
    - 最大/最小面积比
    - 面积基尼系数
    """
    try:
        # 首先计算基本SADI
        sadi = calculate(vision_result)
        if sadi is None:
            return None
        
        # 重新获取区域信息以计算扩展指标
        foreground_data = vision_result['images'].get('foreground_map')
        if isinstance(foreground_data, bytes):
            nparr = np.frombuffer(foreground_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        else:
            img = foreground_data
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        binary = (img > 127).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        labeled = label(binary, connectivity=2)
        regions = regionprops(labeled)
        
        region_areas = [r.area for r in regions if r.area >= 10]
        
        # 计算扩展指标
        extended_metrics = {
            'sadi': sadi,
            'num_regions': len(region_areas),
            'total_area': sum(region_areas) if region_areas else 0,
            'mean_area': np.mean(region_areas) if region_areas else 0,
            'max_min_ratio': max(region_areas) / min(region_areas) if len(region_areas) > 1 else 1,
            'gini_coefficient': calculate_gini(region_areas) if region_areas else 0
        }
        
        return extended_metrics
        
    except Exception as e:
        print(f"计算扩展指标时出错: {str(e)}")
        return None

def calculate_gini(areas):
    """计算基尼系数"""
    if len(areas) == 0:
        return 0
    
    areas = np.array(sorted(areas))
    n = len(areas)
    index = np.arange(1, n + 1)
    
    return (2 * np.sum(index * areas)) / (n * np.sum(areas)) - (n + 1) / n

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
    'name': 'Spatial Area Dispersion Index (SADI)',
    'category': 'Spatial Metrics',
    'required_images': ['foreground_map'],
    'output_range': [0.0, float('inf')],
    'unit': 'index',
    'description': '衡量前景区域的空间分散程度，值越大表示区域大小差异越大'
}
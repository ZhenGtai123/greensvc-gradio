"""
改进的指标计算代码示例
展示如何更好地处理输入图像选择

指标名称: Shape Edge Regularity Index (S_ERI)
功能: 计算前景形状的边缘规则度
required_images: ['fmb_map', 'foreground_map']  # 文档用途
"""

import cv2
import numpy as np
import logging

# 设置日志
logger = logging.getLogger(__name__)

def calculate(vision_result):
    """
    计算S_ERI指标 - 改进版本
    
    这个版本展示了更健壮的图像选择和错误处理
    """
    try:
        # 1. 验证输入
        if not vision_result or vision_result.get('status') != 'success':
            logger.warning("视觉分析结果无效或失败")
            return None
        
        if 'images' not in vision_result:
            logger.error("视觉分析结果中没有图像数据")
            return None
        
        # 2. 灵活的图像选择
        # 定义可用的图像选项（按优先级排序）
        image_options = [
            ('fmb_map', lambda g: g == 0),  # 前中后景图，前景值为0
            ('foreground_map', lambda g: g > 127),  # 前景掩码，前景值>127
        ]
        
        # 尝试获取可用的图像
        img = None
        foreground_extractor = None
        used_image_type = None
        
        for img_type, extractor in image_options:
            if img_type in vision_result['images']:
                img_data = vision_result['images'][img_type]
                
                # 解码图像
                try:
                    nparr = np.frombuffer(img_data, np.uint8)
                    decoded_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if decoded_img is not None:
                        img = decoded_img
                        foreground_extractor = extractor
                        used_image_type = img_type
                        logger.info(f"使用 {img_type} 计算S_ERI")
                        break
                except Exception as e:
                    logger.warning(f"无法解码 {img_type}: {e}")
                    continue
        
        if img is None:
            logger.error("没有找到可用的图像来计算S_ERI")
            return None
        
        # 3. 转换为灰度图
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # 4. 提取前景
        foreground = foreground_extractor(gray).astype(np.uint8) * 255
        
        # 5. 形态学处理（可选，提高轮廓质量）
        kernel = np.ones((3, 3), np.uint8)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
        
        # 6. 查找轮廓
        contours, _ = cv2.findContours(
            foreground, 
            cv2.RETR_EXTERNAL,  # 只获取外部轮廓
            cv2.CHAIN_APPROX_SIMPLE  # 简化轮廓
        )
        
        if not contours:
            logger.warning("未找到前景轮廓")
            return 0.0
        
        # 7. 获取最大轮廓（主要前景对象）
        contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
        max_contour = contours_sorted[0]
        
        # 检查轮廓是否有效
        area = cv2.contourArea(max_contour)
        if area < 100:  # 最小面积阈值
            logger.warning(f"前景面积过小: {area}")
            return 0.0
        
        # 8. 计算周长
        perimeter = cv2.arcLength(max_contour, True)
        
        # 9. 计算S_ERI
        # 基于等周不等式：圆形的值最小（π/2 ≈ 1.57）
        # 正方形的值为1，形状越不规则值越大
        seri = 0.25 * perimeter / np.sqrt(area)
        
        # 10. 结果验证
        if not np.isfinite(seri) or seri < 0:
            logger.error(f"计算结果无效: {seri}")
            return None
        
        # 11. 可选：记录额外信息
        logger.debug(f"S_ERI计算详情: 周长={perimeter:.2f}, 面积={area:.2f}, SERI={seri:.3f}")
        
        # 如果需要，可以返回更详细的结果
        # return {
        #     'value': float(seri),
        #     'details': {
        #         'perimeter': perimeter,
        #         'area': area,
        #         'image_used': used_image_type,
        #         'num_contours': len(contours)
        #     }
        # }
        
        return float(seri)
        
    except Exception as e:
        logger.error(f"计算S_ERI时发生错误: {str(e)}", exc_info=True)
        return None


def calculate_batch(vision_results_list):
    """
    批量计算S_ERI（可选的扩展功能）
    
    Args:
        vision_results_list: 视觉分析结果列表
        
    Returns:
        结果列表
    """
    results = []
    for i, vision_result in enumerate(vision_results_list):
        result = calculate(vision_result)
        results.append({
            'index': i,
            'value': result,
            'success': result is not None
        })
    return results


def get_metric_info():
    """
    返回指标的元信息（可选的扩展功能）
    """
    return {
        'name': 'Shape Edge Regularity Index (S_ERI)',
        'category': 'Shape',
        'required_images': ['fmb_map', 'foreground_map'],
        'preferred_image': 'fmb_map',
        'value_range': '[π/2, +∞)',
        'unit': 'Dimensionless',
        'description': '评估前景元素边界的规则程度，值越小表示形状越规则'
    }


# 测试代码
if __name__ == "__main__":
    # 创建测试数据
    print("测试S_ERI指标计算...")
    
    # 创建一个简单的测试图像
    test_img = np.zeros((600, 800), dtype=np.uint8)
    
    # 添加一个矩形作为前景（在fmb_map中，前景值为0）
    cv2.rectangle(test_img, (200, 200), (400, 400), 0, -1)
    
    # 背景设为2
    test_img[test_img != 0] = 2
    
    # 编码为彩色图像（模拟实际的fmb_map）
    test_img_color = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR)
    _, buffer = cv2.imencode('.png', test_img_color)
    
    # 创建测试vision_result
    test_vision_result = {
        'status': 'success',
        'images': {
            'fmb_map': buffer.tobytes()
        }
    }
    
    # 测试计算
    result = calculate(test_vision_result)
    
    print(f"测试结果: S_ERI = {result}")
    print(f"预期值约为 1.0（正方形）")
    
    # 测试错误情况
    print("\n测试错误处理...")
    
    # 空输入
    result_empty = calculate({})
    print(f"空输入结果: {result_empty}")
    
    # 缺少图像
    result_no_img = calculate({'status': 'success', 'images': {}})
    print(f"缺少图像结果: {result_no_img}")
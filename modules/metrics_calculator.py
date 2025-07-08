"""
指标计算模块
动态加载和执行指标计算代码
"""

import os
import sys
import importlib.util
import cv2
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union
import tempfile
import io
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class MetricsCalculator:
    """指标计算器"""
    
    def __init__(self, metrics_code_dir: str):
        self.metrics_code_dir = metrics_code_dir
        self.loaded_modules = {}
        self.temp_dir = tempfile.mkdtemp(prefix='metrics_calc_')
        
        # 确保代码目录存在
        os.makedirs(self.metrics_code_dir, exist_ok=True)
        
        # 初始化默认指标计算函数
        self._init_default_calculators()
    
    def _init_default_calculators(self):
        """初始化内置的指标计算函数"""
        self.default_calculators = {
            'Shape Edge Regularity Index (S_ERI)': self._calculate_seri,
            'Shape Edge Contrast Index (S_ECI)': self._calculate_seci,
            'Shape Patch Similarity Index (S_PSI)': self._calculate_spsi,
            'Size View Field Ratio (S_VFR)': self._calculate_svfr,
            # 可以继续添加更多默认指标
        }
    
    def calculate_metric(self, metric_name: str, vision_result: Dict) -> Union[float, Dict]:
        """
        计算指标值
        
        Args:
            metric_name: 指标名称
            vision_result: 视觉分析结果
            
        Returns:
            指标值或结果字典
        """
        try:
            # 首先尝试使用默认计算器
            if metric_name in self.default_calculators:
                return self.default_calculators[metric_name](vision_result)
            
            # 尝试加载自定义代码
            module = self._load_metric_module(metric_name)
            if module:
                # 查找计算函数
                calc_func = None
                for attr_name in ['calculate', 'calculate_metric', 'calc', 'main']:
                    if hasattr(module, attr_name):
                        calc_func = getattr(module, attr_name)
                        break
                
                if calc_func:
                    return calc_func(vision_result)
                else:
                    logger.warning(f"未在模块中找到计算函数: {metric_name}")
                    return None
            
            logger.warning(f"未找到指标计算方法: {metric_name}")
            return None
            
        except Exception as e:
            logger.error(f"计算指标 {metric_name} 时出错: {e}")
            return None
    
    def _load_metric_module(self, metric_name: str) -> Optional[Any]:
        """加载指标计算模块"""
        try:
            # 检查是否已加载
            if metric_name in self.loaded_modules:
                return self.loaded_modules[metric_name]
            
            # 构建文件路径
            safe_name = metric_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
            code_path = os.path.join(self.metrics_code_dir, f"{safe_name}.py")
            
            if not os.path.exists(code_path):
                return None
            
            # 动态加载模块
            spec = importlib.util.spec_from_file_location(safe_name, code_path)
            module = importlib.util.module_from_spec(spec)
            
            # 添加必要的导入到模块命名空间
            module.cv2 = cv2
            module.np = np
            module.pd = pd
            module.Image = Image
            
            spec.loader.exec_module(module)
            
            # 缓存模块
            self.loaded_modules[metric_name] = module
            
            logger.info(f"成功加载指标模块: {metric_name}")
            return module
            
        except Exception as e:
            logger.error(f"加载指标模块失败 {metric_name}: {e}")
            return None
    
    def batch_calculate(self, metric_names: List[str], vision_results: List[Dict]) -> pd.DataFrame:
        """批量计算多个指标"""
        results = []
        
        for idx, vision_result in enumerate(vision_results):
            row = {'image_index': idx}
            
            for metric_name in metric_names:
                value = self.calculate_metric(metric_name, vision_result)
                row[metric_name] = value
            
            results.append(row)
        
        return pd.DataFrame(results)
    
    # === 默认指标计算函数 ===
    
    def _get_image_from_result(self, vision_result: Dict, image_type: str) -> Optional[np.ndarray]:
        """从视觉分析结果中获取图像"""
        try:
            if 'images' not in vision_result:
                return None
            
            img_data = vision_result['images'].get(image_type)
            if img_data is None:
                return None
            
            # 如果是字节数据，转换为numpy数组
            if isinstance(img_data, bytes):
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                return img
            elif isinstance(img_data, np.ndarray):
                return img_data
            else:
                return None
                
        except Exception as e:
            logger.error(f"获取图像失败 {image_type}: {e}")
            return None
    
    def _calculate_seri(self, vision_result: Dict) -> Optional[float]:
        """计算Shape Edge Regularity Index (S_ERI)"""
        try:
            # 获取前景掩码图
            fmb_map = self._get_image_from_result(vision_result, 'fmb_map')
            if fmb_map is None:
                return None
            
            # 转换为灰度图
            if len(fmb_map.shape) == 3:
                gray = cv2.cvtColor(fmb_map, cv2.COLOR_BGR2GRAY)
            else:
                gray = fmb_map
            
            # 提取前景（假设前景值为0）
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
            
        except Exception as e:
            logger.error(f"计算S_ERI失败: {e}")
            return None
    
    def _calculate_seci(self, vision_result: Dict) -> Optional[float]:
        """计算Shape Edge Contrast Index (S_ECI)"""
        try:
            # 获取语义分割图
            semantic_map = self._get_image_from_result(vision_result, 'semantic_map')
            if semantic_map is None:
                return None
            
            # 转换为灰度图（如果需要）
            if len(semantic_map.shape) == 3:
                gray = cv2.cvtColor(semantic_map, cv2.COLOR_BGR2GRAY)
            else:
                gray = semantic_map
            
            # 检测边缘
            edges = cv2.Canny(gray, 50, 150)
            
            # 计算边缘对比度
            # 这里使用简化的方法：计算边缘像素的比例
            edge_pixels = np.sum(edges > 0)
            total_pixels = edges.shape[0] * edges.shape[1]
            
            if total_pixels == 0:
                return 0.0
            
            seci = edge_pixels / total_pixels
            
            return float(seci)
            
        except Exception as e:
            logger.error(f"计算S_ECI失败: {e}")
            return None
    
    def _calculate_spsi(self, vision_result: Dict) -> Optional[float]:
        """计算Shape Patch Similarity Index (S_PSI)"""
        try:
            # 获取前景掩码
            foreground_map = self._get_image_from_result(vision_result, 'foreground_map')
            if foreground_map is None:
                return None
            
            # 转换为二值图
            if len(foreground_map.shape) == 3:
                gray = cv2.cvtColor(foreground_map, cv2.COLOR_BGR2GRAY)
            else:
                gray = foreground_map
            
            binary = (gray > 127).astype(np.uint8)
            
            # 查找所有孔洞（内部轮廓）
            contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if hierarchy is None:
                return 0.0
            
            # 提取孔洞
            holes = []
            for i, h in enumerate(hierarchy[0]):
                if h[3] != -1:  # 有父轮廓，说明是孔洞
                    holes.append(contours[i])
            
            if not holes:
                return 0.0
            
            # 计算每个孔洞的形状规则度
            regularities = []
            for hole in holes:
                perimeter = cv2.arcLength(hole, True)
                area = cv2.contourArea(hole)
                if area > 0:
                    reg = 0.25 * perimeter / np.sqrt(area)
                    regularities.append(reg)
            
            if not regularities:
                return 0.0
            
            # 计算变异系数
            mean_reg = np.mean(regularities)
            std_reg = np.std(regularities)
            
            if mean_reg == 0:
                return 0.0
            
            cv = std_reg / mean_reg
            
            return float(cv)
            
        except Exception as e:
            logger.error(f"计算S_PSI失败: {e}")
            return None
    
    def _calculate_svfr(self, vision_result: Dict) -> Optional[float]:
        """计算Size View Field Ratio (S_VFR)"""
        try:
            # 获取前景掩码
            foreground_map = self._get_image_from_result(vision_result, 'foreground_map')
            if foreground_map is None:
                return None
            
            # 转换为二值图
            if len(foreground_map.shape) == 3:
                gray = cv2.cvtColor(foreground_map, cv2.COLOR_BGR2GRAY)
            else:
                gray = foreground_map
            
            # 计算前景像素比例
            foreground_pixels = np.sum(gray > 127)
            total_pixels = gray.shape[0] * gray.shape[1]
            
            if total_pixels == 0:
                return 0.0
            
            svfr = foreground_pixels / total_pixels
            
            return float(svfr)
            
        except Exception as e:
            logger.error(f"计算S_VFR失败: {e}")
            return None
    
    def validate_calculation(self, metric_name: str, test_data: Optional[Dict] = None) -> Dict[str, Any]:
        """验证指标计算是否正常工作"""
        try:
            # 使用测试数据或创建模拟数据
            if test_data is None:
                # 创建模拟的视觉分析结果
                test_data = self._create_test_data()
            
            # 尝试计算
            result = self.calculate_metric(metric_name, test_data)
            
            return {
                'success': result is not None,
                'result': result,
                'error': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'result': None,
                'error': str(e)
            }
    
    def _create_test_data(self) -> Dict:
        """创建测试数据"""
        # 创建模拟图像
        test_img = np.zeros((600, 800, 3), dtype=np.uint8)
        
        # 添加一些形状
        cv2.rectangle(test_img, (100, 100), (300, 300), (255, 255, 255), -1)
        cv2.circle(test_img, (500, 300), 100, (128, 128, 128), -1)
        
        # 编码为字节
        _, buffer = cv2.imencode('.png', test_img)
        img_bytes = buffer.tobytes()
        
        return {
            'status': 'success',
            'images': {
                'semantic_map': img_bytes,
                'depth_map': img_bytes,
                'fmb_map': img_bytes,
                'foreground_map': img_bytes,
                'openness_map': img_bytes
            }
        }
    
    def cleanup(self):
        """清理临时文件"""
        try:
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info(f"清理临时目录: {self.temp_dir}")
        except Exception as e:
            logger.error(f"清理失败: {e}")

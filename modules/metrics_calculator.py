"""
Improved Metrics Calculation Module
Supports both legacy format and Stage 2.5 calculator_layer format
"""

import os
import sys
import importlib.util
import cv2
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union, Tuple
import tempfile
import io
from PIL import Image
import logging
import json

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Metrics Calculator
    Supports:
    - Legacy format: calculate(vision_result) functions
    - Stage 2.5 format: calculate_indicator(image_path) functions with INDICATOR dict
    """
    
    def __init__(self, metrics_code_dir: str, metrics_manager=None):
        self.metrics_code_dir = metrics_code_dir
        self.metrics_manager = metrics_manager
        self.loaded_modules = {}
        self.temp_dir = tempfile.mkdtemp(prefix='metrics_calc_')
        
        # Semantic colors for calculator_layer format
        self.semantic_colors = {}
        
        # Ensure code directory exists
        os.makedirs(self.metrics_code_dir, exist_ok=True)
        
        # Initialize default calculators
        self._init_default_calculators()
    
    def load_semantic_colors(self, config_path: str) -> bool:
        """
        Load semantic color configuration for calculator_layer format
        
        Args:
            config_path: Path to Semantic_configuration.json
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            self.semantic_colors = {}
            for item in config:
                name = item.get('name', '')
                hex_color = item.get('color', '')
                if name and hex_color:
                    h = hex_color.lstrip('#')
                    rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
                    self.semantic_colors[name] = rgb
            
            logger.info(f"Loaded {len(self.semantic_colors)} semantic classes")
            return True
        except Exception as e:
            logger.error(f"Failed to load semantic colors: {e}")
            return False
    
    def load_calculator_module(self, indicator_id: str) -> Optional[Any]:
        """
        Load a calculator_layer module by indicator ID
        
        Returns the module with semantic_colors injected
        """
        try:
            # Check cache first
            cache_key = f"calc_{indicator_id}"
            if cache_key in self.loaded_modules:
                return self.loaded_modules[cache_key]
            
            # Find calculator file
            calc_path = os.path.join(self.metrics_code_dir, f"calculator_layer_{indicator_id}.py")
            
            if not os.path.exists(calc_path):
                logger.error(f"Calculator not found: {calc_path}")
                return None
            
            # Load module
            spec = importlib.util.spec_from_file_location(f"calculator_{indicator_id}", calc_path)
            module = importlib.util.module_from_spec(spec)
            
            # Inject semantic_colors before execution
            module.semantic_colors = self.semantic_colors
            
            # Execute module
            spec.loader.exec_module(module)
            
            # Validate module has required components
            if not hasattr(module, 'INDICATOR'):
                logger.error(f"Calculator missing INDICATOR dict: {indicator_id}")
                return None
            
            if not hasattr(module, 'calculate_indicator'):
                logger.error(f"Calculator missing calculate_indicator function: {indicator_id}")
                return None
            
            # Cache and return
            self.loaded_modules[cache_key] = module
            return module
            
        except Exception as e:
            logger.error(f"Failed to load calculator module {indicator_id}: {e}")
            return None
    
    def calculate_from_calculator_layer(self, indicator_id: str, image_path: str) -> Dict:
        """
        Calculate indicator using calculator_layer format
        
        Args:
            indicator_id: Indicator ID (e.g., "IND_ASV")
            image_path: Path to semantic segmentation mask image
            
        Returns:
            Result dict with 'success', 'value', and other fields
        """
        try:
            module = self.load_calculator_module(indicator_id)
            if not module:
                return {'success': False, 'error': f'Failed to load calculator: {indicator_id}'}
            
            # Call calculate_indicator function
            result = module.calculate_indicator(image_path)
            
            # Add indicator info to result
            result['indicator_id'] = indicator_id
            result['indicator_name'] = module.INDICATOR.get('name', '')
            result['unit'] = module.INDICATOR.get('unit', '')
            
            return result
            
        except Exception as e:
            logger.error(f"Calculator error {indicator_id}: {e}")
            return {'success': False, 'error': str(e)}
    
    def calculate_batch(self, indicator_ids: List[str], image_paths: List[str]) -> pd.DataFrame:
        """
        Calculate multiple indicators for multiple images
        
        Returns DataFrame with results
        """
        results = []
        
        for image_path in image_paths:
            for indicator_id in indicator_ids:
                result = self.calculate_from_calculator_layer(indicator_id, image_path)
                
                results.append({
                    'Image': os.path.basename(image_path),
                    'Indicator': indicator_id,
                    'Name': result.get('indicator_name', ''),
                    'Value': result.get('value'),
                    'Unit': result.get('unit', ''),
                    'Success': result.get('success', False),
                    'Error': result.get('error', ''),
                    'Target Pixels': result.get('target_pixels', ''),
                    'Total Pixels': result.get('total_pixels', '')
                })
        
        return pd.DataFrame(results)
    
    def get_calculator_info(self, indicator_id: str) -> Optional[Dict]:
        """Get INDICATOR dict from a calculator module"""
        module = self.load_calculator_module(indicator_id)
        if module and hasattr(module, 'INDICATOR'):
            return module.INDICATOR.copy()
        return None
    
    def _init_default_calculators(self):
        """初始化内置的指标计算函数"""
        self.default_calculators = {
            'Shape Edge Regularity Index (S_ERI)': {
                'function': self._calculate_seri,
                'required_images': ['fmb_map', 'foreground_map']  # 可以使用任一
            },
            'Shape Edge Contrast Index (S_ECI)': {
                'function': self._calculate_seci,
                'required_images': ['semantic_map']
            },
            'Shape Patch Similarity Index (S_PSI)': {
                'function': self._calculate_spsi,
                'required_images': ['foreground_map']
            },
            'Size View Field Ratio (S_VFR)': {
                'function': self._calculate_svfr,
                'required_images': ['foreground_map']
            },
        }
    
    def calculate_metric(self, metric_name: str, vision_result: Dict, 
                        metric_info: Optional[Dict] = None) -> Union[float, Dict]:
        """
        计算指标值
        
        Args:
            metric_name: 指标名称
            vision_result: 视觉分析结果
            metric_info: 指标信息（包含required_images等）
            
        Returns:
            指标值或结果字典
        """
        try:
            # 获取指标的required_images配置
            required_images = self._get_required_images(metric_name, metric_info)
            
            # 验证所需图像是否存在
            validation_result = self._validate_required_images(vision_result, required_images)
            if not validation_result['valid']:
                logger.error(f"指标 {metric_name} 缺少必需的图像: {validation_result['missing']}")
                return None
            
            # 准备简化的输入（只包含所需图像）
            simplified_input = self._prepare_metric_input(vision_result, required_images)
            
            # 首先尝试使用默认计算器
            if metric_name in self.default_calculators:
                calc_func = self.default_calculators[metric_name]['function']
                return calc_func(simplified_input)
            
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
                    # 调用函数，传入简化的输入
                    return calc_func(simplified_input)
                else:
                    logger.warning(f"未在模块中找到计算函数: {metric_name}")
                    return None
            
            logger.warning(f"未找到指标计算方法: {metric_name}")
            return None
            
        except Exception as e:
            logger.error(f"计算指标 {metric_name} 时出错: {e}")
            return None
    
    def _get_required_images(self, metric_name: str, metric_info: Optional[Dict] = None) -> List[str]:
        """获取指标所需的图像列表"""
        # 1. 优先使用metric_info中的配置
        if metric_info and 'required_images' in metric_info:
            if isinstance(metric_info['required_images'], str):
                # 如果是字符串，按逗号分割
                return [img.strip() for img in metric_info['required_images'].split(',')]
            elif isinstance(metric_info['required_images'], list):
                return metric_info['required_images']
        
        # 2. 检查默认计算器配置
        if metric_name in self.default_calculators:
            return self.default_calculators[metric_name].get('required_images', [])
        
        # 3. 从指标库获取（如果有metrics_manager）
        if self.metrics_manager:
            metric_data = self.metrics_manager.get_metric_by_name(metric_name)
            if metric_data and 'required_images' in metric_data:
                if isinstance(metric_data['required_images'], str):
                    return [img.strip() for img in metric_data['required_images'].split(',')]
        
        # 4. 尝试从代码文件的文档字符串中提取
        required = self._extract_required_from_code(metric_name)
        if required:
            return required
        
        # 5. 默认返回所有可能的图像（兼容旧代码）
        logger.warning(f"未找到指标 {metric_name} 的required_images配置，返回所有图像")
        return []
    
    def _extract_required_from_code(self, metric_name: str) -> Optional[List[str]]:
        """从代码文件的文档字符串中提取所需图像"""
        try:
            safe_name = metric_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
            code_path = os.path.join(self.metrics_code_dir, f"{safe_name}.py")
            
            if os.path.exists(code_path):
                with open(code_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 查找 required_images 注释
                import re
                match = re.search(r'required_images:\s*\[(.*?)\]', content, re.IGNORECASE)
                if match:
                    images_str = match.group(1)
                    # 提取图像名称
                    images = re.findall(r"'([^']+)'|\"([^\"]+)\"", images_str)
                    return [img[0] or img[1] for img in images]
                
                # 查找 Required 或 需要 关键词
                match = re.search(r'(?:Required|需要):\s*([^\n]+)', content, re.IGNORECASE)
                if match:
                    text = match.group(1)
                    # 查找已知的图像类型
                    known_types = ['semantic_map', 'depth_map', 'fmb_map', 'foreground_map', 
                                  'openness_map', 'middleground_map', 'background_map']
                    found = []
                    for img_type in known_types:
                        if img_type in text:
                            found.append(img_type)
                    if found:
                        return found
        
        except Exception as e:
            logger.debug(f"无法从代码提取required_images: {e}")
        
        return None
    
    def _validate_required_images(self, vision_result: Dict, required_images: List[str]) -> Dict[str, Any]:
        """验证所需图像是否存在"""
        if not required_images:  # 如果没有指定，认为验证通过
            return {'valid': True, 'missing': []}
        
        if 'images' not in vision_result:
            return {'valid': False, 'missing': required_images}
        
        available_images = vision_result['images'].keys()
        missing = []
        
        for img in required_images:
            if img not in available_images:
                missing.append(img)
        
        return {
            'valid': len(missing) == 0,
            'missing': missing,
            'available': list(available_images)
        }
    
    def _prepare_metric_input(self, vision_result: Dict, required_images: List[str]) -> Dict:
        """准备指标计算的输入，只包含所需的图像"""
        if not required_images:
            # 如果没有指定，返回完整的vision_result（兼容旧代码）
            return vision_result
        
        # 创建简化的输入
        simplified = {
            'status': vision_result.get('status', 'success'),
            'images': {}
        }
        
        # 只复制所需的图像
        if 'images' in vision_result:
            for img_name in required_images:
                if img_name in vision_result['images']:
                    simplified['images'][img_name] = vision_result['images'][img_name]
        
        # 如果required_images中包含替代选项（用|分隔），选择第一个可用的
        for img_spec in required_images:
            if '|' in img_spec:
                alternatives = [alt.strip() for alt in img_spec.split('|')]
                for alt in alternatives:
                    if alt in vision_result.get('images', {}):
                        simplified['images'][img_spec] = vision_result['images'][alt]
                        break
        
        return simplified
    
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
    
    def get_metric_info(self, metric_name: str) -> Dict[str, Any]:
        """获取指标的详细信息，包括所需图像"""
        info = {
            'name': metric_name,
            'has_code': False,
            'required_images': [],
            'is_default': metric_name in self.default_calculators
        }
        
        # 获取所需图像
        info['required_images'] = self._get_required_images(metric_name)
        
        # 检查是否有代码
        if info['is_default']:
            info['has_code'] = True
        else:
            safe_name = metric_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
            code_path = os.path.join(self.metrics_code_dir, f"{safe_name}.py")
            info['has_code'] = os.path.exists(code_path)
        
        return info
    
    def batch_calculate(self, metric_names: List[str], vision_results: List[Dict],
                       metrics_info: Optional[Dict[str, Dict]] = None) -> pd.DataFrame:
        """批量计算多个指标"""
        results = []
        
        for idx, vision_result in enumerate(vision_results):
            row = {'image_index': idx}
            
            for metric_name in metric_names:
                metric_info = metrics_info.get(metric_name) if metrics_info else None
                value = self.calculate_metric(metric_name, vision_result, metric_info)
                row[metric_name] = value
            
            results.append(row)
        
        return pd.DataFrame(results)
    
    # === 默认指标计算函数（保持不变）===
    
    def _get_image_from_result(self, vision_result: Dict, image_type: str) -> Optional[np.ndarray]:
        """从视觉分析结果中获取图像"""
        try:
            if 'images' not in vision_result:
                return None
            
            # 支持多个备选图像（用|分隔）
            if '|' in image_type:
                for alt in image_type.split('|'):
                    img = self._get_single_image(vision_result, alt.strip())
                    if img is not None:
                        return img
                return None
            else:
                return self._get_single_image(vision_result, image_type)
                
        except Exception as e:
            logger.error(f"获取图像失败 {image_type}: {e}")
            return None
    
    def _get_single_image(self, vision_result: Dict, image_type: str) -> Optional[np.ndarray]:
        """获取单个图像"""
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
    
    def _calculate_seri(self, vision_result: Dict) -> Optional[float]:
        """计算Shape Edge Regularity Index (S_ERI)"""
        try:
            # 尝试使用fmb_map或foreground_map
            img = self._get_image_from_result(vision_result, 'fmb_map|foreground_map')
            if img is None:
                return None
            
            # 转换为灰度图
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            
            # 提取前景
            if 'fmb_map' in vision_result.get('images', {}):
                foreground = (gray == 0).astype(np.uint8) * 255
            else:  # foreground_map
                foreground = (gray > 127).astype(np.uint8) * 255
            
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
            
            # 获取指标信息
            metric_info = self.get_metric_info(metric_name)
            
            # 尝试计算
            result = self.calculate_metric(metric_name, test_data, metric_info)
            
            return {
                'success': result is not None,
                'result': result,
                'error': None,
                'metric_info': metric_info
            }
            
        except Exception as e:
            return {
                'success': False,
                'result': None,
                'error': str(e),
                'metric_info': None
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
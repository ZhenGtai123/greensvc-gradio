"""
API客户端模块
处理与视觉模型和指标推荐API的通信
"""

import requests
import json
import base64
import io
from typing import Dict, List, Optional, Any
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class VisionModelClient:
    """视觉模型API客户端"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.default_colors = self._get_default_colors()
    
    def _get_default_colors(self) -> Dict:
        """获取默认颜色配置"""
        # 生成一个包含足够多颜色的调色板
        base_colors = [
            [0, 0, 0],          # 0: 背景 - 黑色
            [6, 230, 230],      # 1: Sky - 青色
            [4, 250, 7],        # 2: Grass - 绿色
            [250, 127, 4],      # 3: Trees - 橙色
            [4, 200, 3],        # 4: Shrubs - 深绿
            [204, 255, 4],      # 5: Water - 亮黄绿
            [237, 117, 57],     # 6: Land - 橙红
            [220, 20, 60],      # 7: Building - 深红
            [255, 192, 203],    # 8: Rock - 粉红
            [255, 0, 255],      # 9: People - 品红
            [128, 128, 128],    # 10: Fences - 灰色
            [100, 100, 100],    # 11: Roads - 深灰
            [200, 200, 200],    # 12: Pavements - 浅灰
            [139, 69, 19],      # 13: Bridge - 棕色
            [0, 0, 255],        # 14: Cars - 蓝色
            [255, 255, 0],      # 15: Others - 黄色
            [204, 70, 3],       # 16: Chairs
            [255, 31, 0],       # 17: Bases
            [255, 224, 0],      # 18: Steps
            [255, 184, 6],      # 19: Fences2
            [255, 5, 153],      # 20: Signs
            [173, 0, 255],      # 21: Bins
            [255, 184, 184],    # 22: Towers
            [0, 255, 61],       # 23: Awnings
            [0, 71, 255],       # 24: Street Lights
            [255, 235, 0],      # 25: Boat
            [8, 184, 170],      # 26: Fountains
            [255, 245, 0],      # 27: Bicycles
            [255, 255, 0],      # 28: Sculptures
            [71, 0, 255],       # 29: Piers
            [78, 255, 0],       # 30: Aquatic plants
            [0, 255, 78],       # 31: Green buildings
            [130, 81, 62],      # 32: Couplets
            [226, 200, 160],    # 33: Riverbanks
            [143, 255, 140],    # 34: Hills
            [255, 113, 4],      # 35: Construction equipment
            [181, 166, 174],    # 36: Poles
            [110, 220, 167],    # 37: Animal
            [72, 72, 70],       # 38: Monuments
            [54, 40, 59],       # 39: Doors
            [55, 57, 58],       # 40: Sports equipment
            [39, 196, 196],     # 41: Waterfalls
            [255, 208, 0]       # 42: Pavilion
        ]
        
        # 生成额外的颜色以支持更多类别
        import random
        random.seed(42)  # 确保颜色生成的一致性
        while len(base_colors) < 100:  # 支持最多99个类别
            # 生成不重复的颜色
            new_color = [
                random.randint(50, 255),
                random.randint(50, 255),
                random.randint(50, 255)
            ]
            if new_color not in base_colors:
                base_colors.append(new_color)
        
        # 转换为字符串键的字典
        semantic_colors = {str(i): color for i, color in enumerate(base_colors)}
        
        return {
            'semantic_colors': semantic_colors,
            'openness_colors': {
                "0": [113, 6, 230],      # 封闭 - 紫色
                "1": [173, 255, 0]       # 开放 - 亮绿色
            },
            'fmb_colors': {
                "0": [220, 20, 60],      # 前景 - 深红色
                "1": [46, 125, 50],      # 中景 - 深绿色
                "2": [30, 144, 255]      # 背景 - 道奇蓝
            }
        }
    
    def _generate_colors_for_classes(self, num_classes: int) -> Dict[str, List[int]]:
        """为指定数量的类别生成颜色映射"""
        colors = self._get_default_colors()['semantic_colors']
        
        # 确保有足够的颜色（类别数 + 1个背景）
        needed_colors = num_classes + 1
        
        # 如果预定义的颜色足够，直接返回所需数量
        if len(colors) >= needed_colors:
            return {str(i): colors[str(i)] for i in range(needed_colors)}
        
        # 如果不够，生成额外的颜色
        result = {}
        for i in range(needed_colors):
            if str(i) in colors:
                result[str(i)] = colors[str(i)]
            else:
                # 生成新颜色
                import random
                random.seed(i)  # 确保相同索引生成相同颜色
                result[str(i)] = [
                    random.randint(50, 255),
                    random.randint(50, 255),
                    random.randint(50, 255)
                ]
        
        return result
    
    def analyze_image(self, image_path: str, semantic_classes: List[str], 
                     semantic_countability: List[int], openness_list: List[int]) -> Dict:
        """
        调用视觉模型API分析图片
        
        Args:
            image_path: 图片路径
            semantic_classes: 语义类别列表
            semantic_countability: 可数性列表
            openness_list: 开放度列表
            
        Returns:
            分析结果字典
        """
        try:
            # 生成正确数量的颜色
            semantic_colors = self._generate_colors_for_classes(len(semantic_classes))
            
            # 准备请求数据 - 更新为新API格式
            request_data = {
                "image_id": f"img_{hash(image_path)}",
                "semantic_classes": semantic_classes,
                "semantic_countability": semantic_countability,
                "openness_list": openness_list,
                "encoder": "vitb",
                "semantic_colors": semantic_colors,  # 使用动态生成的颜色
                "openness_colors": self.default_colors['openness_colors'],
                "fmb_colors": self.default_colors['fmb_colors'],
                "segmentation_mode": "single_label",  # 可选: "single_label" 或 "instance"
                "detection_threshold": 0.05,
                "min_object_area_ratio": 0.00005,
                "enable_hole_filling": False
            }
            
            # 读取图片文件
            with open(image_path, 'rb') as f:
                files = {'file': f}
                data = {'request_data': json.dumps(request_data)}
                
                # 发送请求
                response = requests.post(
                    f"{self.base_url}/analyze",
                    files=files,
                    data=data,
                    timeout=300  # 5分钟超时
                )
            
            if response.status_code == 200:
                # 解析响应 - 响应格式已改变
                result = response.json()
                
                # 处理新的响应格式
                if result.get('status') == 'success':
                    # 将hex字符串转换回图片数据
                    processed_images = {}
                    if 'images' in result:
                        for key, hex_data in result['images'].items():
                            if isinstance(hex_data, str):
                                # 将hex转换为字节
                                img_bytes = bytes.fromhex(hex_data)
                                processed_images[key] = img_bytes
                    
                    # 构建返回结果
                    return {
                        'status': 'success',
                        'images': processed_images,
                        'image_path': image_path,
                        'statistics': {
                            'detected_classes': result.get('detected_classes', 0),
                            'total_classes': result.get('total_classes', len(semantic_classes)),
                            'class_statistics': result.get('class_statistics', {}),
                            'fmb_statistics': result.get('fmb_statistics', {}),
                            'original_size': result.get('original_size', {}),
                            'processed_size': result.get('processed_size', {'width': 800, 'height': 600})
                        },
                        'segmentation_mode': result.get('segmentation_mode', 'single_label'),
                        'instances': result.get('instances', [])  # 如果是实例分割模式
                    }
                else:
                    logger.error(f"Vision API returned error status: {result}")
                    return {
                        'status': 'error',
                        'error': 'API返回错误状态'
                    }
            else:
                logger.error(f"Vision API error: {response.status_code} - {response.text}")
                return {
                    'status': 'error',
                    'error': f"API返回错误: {response.status_code}"
                }
                
        except Exception as e:
            logger.error(f"Vision API exception: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def analyze_image_advanced(self, image_path: str, semantic_classes: List[str], 
                             semantic_countability: List[int], openness_list: List[int],
                             segmentation_mode: str = "single_label",
                             detection_threshold: float = 0.05,
                             min_object_area_ratio: float = 0.00005,
                             enable_hole_filling: bool = False) -> Dict:
        """
        使用高级选项调用视觉模型API分析图片
        
        Args:
            image_path: 图片路径
            semantic_classes: 语义类别列表
            semantic_countability: 可数性列表
            openness_list: 开放度列表
            segmentation_mode: 分割模式 ("single_label" 或 "instance")
            detection_threshold: 检测阈值
            min_object_area_ratio: 最小对象面积比例
            enable_hole_filling: 是否启用空洞填充
            
        Returns:
            分析结果字典
        """
        try:
            # 生成正确数量的颜色
            semantic_colors = self._generate_colors_for_classes(len(semantic_classes))
            
            # 准备请求数据
            request_data = {
                "image_id": f"img_{hash(image_path)}",
                "semantic_classes": semantic_classes,
                "semantic_countability": semantic_countability,
                "openness_list": openness_list,
                "encoder": "vitb",
                "semantic_colors": semantic_colors,  # 使用动态生成的颜色
                "openness_colors": self.default_colors['openness_colors'],
                "fmb_colors": self.default_colors['fmb_colors'],
                "segmentation_mode": segmentation_mode,
                "detection_threshold": detection_threshold,
                "min_object_area_ratio": min_object_area_ratio,
                "enable_hole_filling": enable_hole_filling
            }
            
            # 读取图片文件
            with open(image_path, 'rb') as f:
                files = {'file': f}
                data = {'request_data': json.dumps(request_data)}
                
                # 发送请求
                response = requests.post(
                    f"{self.base_url}/analyze",
                    files=files,
                    data=data,
                    timeout=300
                )
            
            if response.status_code == 200:
                # 解析响应
                result = response.json()
                
                if result.get('status') == 'success':
                    # 处理图像数据
                    processed_images = {}
                    if 'images' in result:
                        for key, hex_data in result['images'].items():
                            if isinstance(hex_data, str):
                                img_bytes = bytes.fromhex(hex_data)
                                processed_images[key] = img_bytes
                    
                    return {
                        'status': 'success',
                        'images': processed_images,
                        'image_path': image_path,
                        'statistics': {
                            'detected_classes': result.get('detected_classes', 0),
                            'total_classes': result.get('total_classes', len(semantic_classes)),
                            'class_statistics': result.get('class_statistics', {}),
                            'fmb_statistics': result.get('fmb_statistics', {}),
                            'original_size': result.get('original_size', {}),
                            'processed_size': result.get('processed_size', {'width': 800, 'height': 600})
                        },
                        'segmentation_mode': result.get('segmentation_mode', segmentation_mode),
                        'instances': result.get('instances', []),
                        'hole_filling_enabled': result.get('hole_filling_enabled', enable_hole_filling)
                    }
                else:
                    return {
                        'status': 'error',
                        'error': 'API返回错误状态'
                    }
            else:
                logger.error(f"Vision API error: {response.status_code}")
                return {
                    'status': 'error',
                    'error': f"API返回错误: {response.status_code}"
                }
                
        except Exception as e:
            logger.error(f"Vision API exception: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def download_results_zip(self, image_path: str, semantic_classes: List[str],
                           semantic_countability: List[int], openness_list: List[int]) -> Optional[bytes]:
        """下载结果的ZIP文件"""
        try:
            # 生成正确数量的颜色
            semantic_colors = self._generate_colors_for_classes(len(semantic_classes))
            
            request_data = {
                "image_id": f"img_{hash(image_path)}",
                "semantic_classes": semantic_classes,
                "semantic_countability": semantic_countability,
                "openness_list": openness_list,
                "encoder": "vitb",
                "semantic_colors": semantic_colors,  # 使用动态生成的颜色
                "openness_colors": self.default_colors['openness_colors'],
                "fmb_colors": self.default_colors['fmb_colors'],
                "segmentation_mode": "single_label",
                "detection_threshold": 0.05,
                "min_object_area_ratio": 0.00005,
                "enable_hole_filling": False
            }
            
            with open(image_path, 'rb') as f:
                files = {'file': f}
                data = {'request_data': json.dumps(request_data)}
                
                response = requests.post(
                    f"{self.base_url}/download_zip",
                    files=files,
                    data=data,
                    timeout=300
                )
            
            if response.status_code == 200:
                return response.content
            else:
                logger.error(f"Download ZIP error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Download ZIP exception: {str(e)}")
            return None
    
    def get_config(self) -> Optional[Dict]:
        """获取API配置信息"""
        try:
            response = requests.get(f"{self.base_url}/config", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Get config error: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Get config exception: {str(e)}")
            return None
    
    def check_health(self) -> bool:
        """检查API健康状态"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                return health_data.get('status') == 'healthy'
            return False
        except:
            return False


class MetricsRecommenderClient:
    """指标推荐API客户端"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
    
    def recommend_metrics(self, user_input: str, openai_api_key: Optional[str] = None,
                         temperature: float = 0.3, max_tokens: int = 1000) -> Dict:
        """
        调用指标推荐API
        
        Args:
            user_input: 用户输入的需求描述
            openai_api_key: OpenAI API密钥
            temperature: 生成温度
            max_tokens: 最大令牌数
            
        Returns:
            推荐结果字典
        """
        try:
            # 准备请求数据
            request_data = {
                "user_input": user_input,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            if openai_api_key:
                request_data["openai_api_key"] = openai_api_key
            
            # 发送请求
            response = requests.post(
                f"{self.base_url}/recommend/simple",
                json=request_data,
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Metrics API error: {response.status_code} - {response.text}")
                return {
                    'error': f"API返回错误: {response.status_code}",
                    'recommendation': '[]'
                }
                
        except Exception as e:
            logger.error(f"Metrics API exception: {str(e)}")
            return {
                'error': str(e),
                'recommendation': '[]'
            }
    
    def check_health(self) -> bool:
        """检查API健康状态"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False


class LocalMetricsRecommender:
    """本地指标推荐器（当API不可用时的备选方案）"""
    
    def __init__(self, metrics_library_path: str):
        self.metrics_library_path = metrics_library_path
        self._load_metrics()
    
    def _load_metrics(self):
        """加载指标库"""
        try:
            import pandas as pd
            self.metrics_df = pd.read_excel(self.metrics_library_path)
        except Exception as e:
            logger.error(f"Failed to load metrics library: {e}")
            self.metrics_df = None
    
    def recommend_metrics(self, user_input: str) -> List[Dict]:
        """基于关键词的简单推荐"""
        if self.metrics_df is None:
            return []
        
        recommendations = []
        keywords = user_input.lower().split()
        
        for _, metric in self.metrics_df.iterrows():
            score = 0
            metric_text = f"{metric.get('metric name', '')} {metric.get('Professional Interpretation', '')}".lower()
            
            for keyword in keywords:
                if keyword in metric_text:
                    score += 1
            
            if score > 0:
                recommendations.append({
                    'name': metric['metric name'],
                    'reason': f"包含关键词: {', '.join([kw for kw in keywords if kw in metric_text])}",
                    'score': score
                })
        
        # 按分数排序并返回前5个
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:5]
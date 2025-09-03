"""
API客户端模块 - 增强版
处理与视觉模型和指标推荐API的通信
"""

import requests
import json
import base64
import io
import os
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from PIL import Image
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)

class VisionModelClient:
    """视觉模型API客户端 - 增强版"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')  # 确保是public属性
        self.default_colors = self._get_default_colors()
        self._last_health_check = 0
        self._health_cache = None
        self._config_cache = None
    
    def _get_default_colors(self) -> Dict:
        """获取默认颜色配置 - 支持更多类别"""
        # 使用新API中的默认颜色
        semantic_colors = {
            "0": [0, 0, 0],          # Background
            "1": [6, 230, 230],      # Sky
            "2": [4, 250, 7],        # Lawn
            "3": [250, 127, 4],      # Herbaceous
            "4": [4, 200, 3],        # Trees
            "5": [204, 255, 4],      # Shrubs
            "6": [9, 7, 230],        # Water
            "7": [120, 120, 70],     # Land
            "8": [180, 120, 120],    # Building
            "9": [255, 41, 10],      # Rock
            "10": [150, 5, 61],      # People
            "11": [120, 120, 120],   # Wall
            "12": [140, 140, 140],   # Roads
            "13": [235, 255, 7],     # Pavements
            "14": [255, 82, 0],      # Bridge
            "15": [0, 102, 200],     # Automobiles
            "16": [204, 70, 3],      # Chairs
            "17": [255, 31, 0],      # Bases
            "18": [255, 224, 0],     # Steps
            "19": [255, 184, 6],     # Fences
            "20": [255, 5, 153],     # Signs
            "21": [173, 0, 255],     # Bins
            "22": [255, 184, 184],   # Towers
            "23": [0, 255, 61],      # Awnings
            "24": [0, 71, 255],      # Street Lights
            "25": [255, 235, 0],     # Boat
            "26": [8, 184, 170],     # Fountains
            "27": [255, 245, 0],     # Bicycles
            "28": [255, 255, 0],     # Sculptures
            "29": [71, 0, 255],      # Piers
            "30": [78, 255, 0],      # Aquatic plants
            "31": [0, 255, 78],      # Green buildings
            "32": [130, 81, 62],     # Couplets
            "33": [226, 200, 160],   # Riverbanks
            "34": [143, 255, 140],   # Hills
            "35": [255, 113, 4],     # Construction equipment
            "36": [181, 166, 174],   # Poles
            "37": [110, 220, 167],   # Animal
            "38": [72, 72, 70],      # Monuments
            "39": [54, 40, 59],      # Doors
            "40": [55, 57, 58],      # Sports equipment
            "41": [39, 196, 196],    # Waterfalls
            "42": [255, 208, 0]      # Pavilion
        }
        
        # 生成额外的颜色以支持最多99个类别
        import random
        random.seed(42)
        color_set = set(tuple(color) for color in semantic_colors.values())
        
        for i in range(43, 100):
            while True:
                new_color = [
                    random.randint(30, 255),
                    random.randint(30, 255),
                    random.randint(30, 255)
                ]
                if tuple(new_color) not in color_set:
                    semantic_colors[str(i)] = new_color
                    color_set.add(tuple(new_color))
                    break
        
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
        
        if len(colors) >= needed_colors:
            return {str(i): colors[str(i)] for i in range(needed_colors)}
        
        # 理论上不应该到这里，因为我们已经生成了100个颜色
        logger.warning(f"Requested {needed_colors} colors, but only have {len(colors)}")
        return colors
    
    def analyze_image(self, image_path: str, semantic_classes: List[str], 
                     semantic_countability: List[int], openness_list: List[int],
                     encoder: str = "vitb") -> Dict:
        """
        调用视觉模型API分析图片 - 基础版本
        
        Args:
            image_path: 图片路径
            semantic_classes: 语义类别列表
            semantic_countability: 可数性列表
            openness_list: 开放度列表
            encoder: 深度模型编码器类型
            
        Returns:
            分析结果字典
        """
        return self.analyze_image_advanced(
            image_path=image_path,
            semantic_classes=semantic_classes,
            semantic_countability=semantic_countability,
            openness_list=openness_list,
            encoder=encoder,
            segmentation_mode="single_label",
            detection_threshold=0.3,
            min_object_area_ratio=0.0001,
            enable_hole_filling=False
        )
    
    def analyze_image_advanced(self, image_path: str, semantic_classes: List[str], 
                            semantic_countability: List[int], openness_list: List[int],
                            encoder: str = "vitb",
                            segmentation_mode: str = "single_label",
                            detection_threshold: float = 0.3,
                            min_object_area_ratio: float = 0.0001,
                            enable_hole_filling: bool = False) -> Dict:
        """
        使用高级选项调用视觉模型API分析图片
        
        Args:
            image_path: 图片路径
            semantic_classes: 语义类别列表
            semantic_countability: 可数性列表
            openness_list: 开放度列表
            encoder: 深度模型编码器类型 ("vitb", "vitl", "vits")
            segmentation_mode: 分割模式 ("single_label" 或 "instance")
            detection_threshold: 检测阈值 (必须 >= 0.1)
            min_object_area_ratio: 最小对象面积比例
            enable_hole_filling: 是否启用空洞填充
            
        Returns:
            分析结果字典
        """
        try:
            # 验证参数
            if not semantic_classes:
                raise ValueError("语义类别列表不能为空")
            
            if len(semantic_classes) != len(semantic_countability) or len(semantic_classes) != len(openness_list):
                raise ValueError("类别数量与参数数量不匹配")
            
            # 验证检测阈值
            if detection_threshold < 0.1:
                logger.warning(f"检测阈值 {detection_threshold} 小于最小值 0.1，自动调整为 0.1")
                detection_threshold = 0.1
            elif detection_threshold > 0.9:
                logger.warning(f"检测阈值 {detection_threshold} 大于最大值 0.9，自动调整为 0.9")
                detection_threshold = 0.9
            # 生成正确数量的颜色
            semantic_colors = self._generate_colors_for_classes(len(semantic_classes))
            
            # 准备请求数据
            request_data = {
                "image_id": f"img_{int(time.time() * 1000)}_{hash(image_path)}",
                "semantic_classes": semantic_classes,
                "semantic_countability": semantic_countability,
                "openness_list": openness_list,
                "encoder": encoder,
                "semantic_colors": semantic_colors,
                "openness_colors": self.default_colors['openness_colors'],
                "fmb_colors": self.default_colors['fmb_colors'],
                "segmentation_mode": segmentation_mode,
                "detection_threshold": detection_threshold,
                "min_object_area_ratio": min_object_area_ratio,
                "enable_hole_filling": enable_hole_filling
            }
            
            # 记录请求信息
            logger.info(f"Analyzing image with {len(semantic_classes)} classes, mode: {segmentation_mode}")
            
            # 读取图片文件
            with open(image_path, 'rb') as f:
                files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
                data = {'request_data': json.dumps(request_data)}
                
                # 发送请求
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/analyze",
                    files=files,
                    data=data,
                    timeout=600  # 10分钟超时，适应大图片和复杂分析
                )
                elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get('status') == 'success':
                    # 处理图像数据
                    processed_images = {}
                    if 'images' in result:
                        for key, hex_data in result['images'].items():
                            if isinstance(hex_data, str):
                                img_bytes = bytes.fromhex(hex_data)
                                processed_images[key] = img_bytes
                    
                    # 构建完整的返回结果
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
                        'hole_filling_enabled': result.get('hole_filling_enabled', enable_hole_filling),
                        'image_count': result.get('image_count', len(processed_images)),
                        'processing_time': elapsed_time,
                        'encoder': encoder
                    }
                else:
                    logger.error(f"API returned error status: {result}")
                    return {
                        'status': 'error',
                        'error': result.get('detail', 'API返回错误状态')
                    }
            else:
                error_msg = f"API返回错误: {response.status_code}"
                try:
                    error_detail = response.json()
                    error_msg += f" - {error_detail.get('detail', response.text)}"
                except:
                    error_msg += f" - {response.text[:200]}"
                
                logger.error(error_msg)
                return {
                    'status': 'error',
                    'error': error_msg
                }
                
        except Exception as e:
            logger.error(f"Vision API exception: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def download_results_zip(self, image_path: str, semantic_classes: List[str],
                           semantic_countability: List[int], openness_list: List[int],
                           encoder: str = "vitb",
                           segmentation_mode: str = "single_label",
                           detection_threshold: float = 0.3,
                           min_object_area_ratio: float = 0.0001,
                           enable_hole_filling: bool = False) -> Tuple[Optional[bytes], Optional[str]]:
        """
        下载结果的ZIP文件
        
        Returns:
            (zip_content, filename) 元组
        """
        try:
            semantic_colors = self._generate_colors_for_classes(len(semantic_classes))
            
            request_data = {
                "image_id": f"img_{int(time.time() * 1000)}_{hash(image_path)}",
                "semantic_classes": semantic_classes,
                "semantic_countability": semantic_countability,
                "openness_list": openness_list,
                "encoder": encoder,
                "semantic_colors": semantic_colors,
                "openness_colors": self.default_colors['openness_colors'],
                "fmb_colors": self.default_colors['fmb_colors'],
                "segmentation_mode": segmentation_mode,
                "detection_threshold": detection_threshold,
                "min_object_area_ratio": min_object_area_ratio,
                "enable_hole_filling": enable_hole_filling
            }
            
            with open(image_path, 'rb') as f:
                files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
                data = {'request_data': json.dumps(request_data)}
                
                response = requests.post(
                    f"{self.base_url}/download_zip",
                    files=files,
                    data=data,
                    timeout=600
                )
            
            if response.status_code == 200:
                # 从响应头获取文件名
                content_disposition = response.headers.get('Content-Disposition', '')
                if 'filename=' in content_disposition:
                    filename = content_disposition.split('filename=')[-1].strip('"')
                else:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"vision_analysis_{timestamp}.zip"
                
                return response.content, filename
            else:
                logger.error(f"Download ZIP error: {response.status_code}")
                return None, None
                
        except Exception as e:
            logger.error(f"Download ZIP exception: {str(e)}")
            return None, None
    
    def batch_analyze(self, image_paths: List[str], semantic_classes: List[str],
                     semantic_countability: List[int], openness_list: List[int],
                     **kwargs) -> List[Dict]:
        """
        批量分析多张图片
        
        Args:
            image_paths: 图片路径列表
            其他参数同 analyze_image_advanced
            
        Returns:
            结果列表
        """
        results = []
        for idx, image_path in enumerate(image_paths):
            logger.info(f"Batch analyzing image {idx + 1}/{len(image_paths)}")
            result = self.analyze_image_advanced(
                image_path,
                semantic_classes,
                semantic_countability,
                openness_list,
                **kwargs
            )
            results.append(result)
            
            # 避免过快请求
            if idx < len(image_paths) - 1:
                time.sleep(0.5)
        
        return results
    
    def get_config(self) -> Optional[Dict]:
        """获取API配置信息（带缓存）"""
        try:
            # 使用缓存避免频繁请求
            if self._config_cache is not None:
                return self._config_cache
            
            response = requests.get(f"{self.base_url}/config", timeout=5)
            if response.status_code == 200:
                self._config_cache = response.json()
                return self._config_cache
            else:
                logger.error(f"Get config error: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Get config exception: {str(e)}")
            return None
    
    def check_health(self) -> bool:
        """检查API健康状态（带缓存）"""
        try:
            # 缓存健康检查结果5秒
            current_time = time.time()
            if self._health_cache is not None and current_time - self._last_health_check < 5:
                return self._health_cache
            
            response = requests.get(f"{self.base_url}/health", timeout=3)
            if response.status_code == 200:
                health_data = response.json()
                self._health_cache = health_data.get('status') == 'healthy'
                self._last_health_check = current_time
                return self._health_cache
            
            self._health_cache = False
            return False
        except:
            self._health_cache = False
            return False
    
    def get_supported_encoders(self) -> List[str]:
        """获取支持的编码器类型"""
        config = self.get_config()
        if config:
            # 从配置中提取支持的编码器
            return ["vitb", "vitl", "vits"]  # 目前API支持的编码器
        return ["vitb"]  # 默认编码器
    
    def validate_parameters(self, semantic_classes: List[str], 
                          semantic_countability: List[int], 
                          openness_list: List[int]) -> Tuple[bool, str]:
        """
        验证参数的有效性
        
        Returns:
            (is_valid, error_message)
        """
        if not semantic_classes:
            return False, "语义类别列表不能为空"
        
        if len(semantic_classes) > 99:
            return False, f"类别数量({len(semantic_classes)})超过最大限制(99)"
        
        if len(semantic_countability) != len(semantic_classes):
            return False, f"可数性参数数量({len(semantic_countability)})与类别数量({len(semantic_classes)})不匹配"
        
        if len(openness_list) != len(semantic_classes):
            return False, f"开放度参数数量({len(openness_list)})与类别数量({len(semantic_classes)})不匹配"
        
        if not all(x in [0, 1] for x in semantic_countability):
            return False, "可数性参数只能是0或1"
        
        if not all(x in [0, 1] for x in openness_list):
            return False, "开放度参数只能是0或1"
        
        return True, ""


class MetricsRecommenderClient:
    """指标推荐API客户端"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self._health_cache = None
        self._last_health_check = 0
    
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
                timeout=300
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
        """检查API健康状态（带缓存）"""
        try:
            current_time = time.time()
            if self._health_cache is not None and current_time - self._last_health_check < 5:
                return self._health_cache
            
            response = requests.get(f"{self.base_url}/health", timeout=3)
            self._health_cache = response.status_code == 200
            self._last_health_check = current_time
            return self._health_cache
        except:
            self._health_cache = False
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
            logger.info(f"Loaded {len(self.metrics_df)} metrics from library")
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
            
            # 计算关键词匹配得分
            matched_keywords = []
            for keyword in keywords:
                if len(keyword) > 2 and keyword in metric_text:  # 忽略太短的词
                    score += 1
                    matched_keywords.append(keyword)
            
            if score > 0:
                recommendations.append({
                    'name': metric.get('metric name', 'Unknown'),
                    'reason': f"包含关键词: {', '.join(matched_keywords)}",
                    'score': score,
                    'description': metric.get('Professional Interpretation', '')[:100] + '...'
                })
        
        # 按分数排序并返回前10个
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:10]


# 工具函数
def create_vision_client(base_url: str) -> VisionModelClient:
    """创建视觉模型客户端"""
    client = VisionModelClient(base_url)
    if client.check_health():
        logger.info("Vision API is healthy")
    else:
        logger.warning("Vision API is not responding")
    return client


def create_metrics_client(base_url: str, fallback_library_path: Optional[str] = None) -> Any:
    """创建指标推荐客户端，支持降级到本地"""
    client = MetricsRecommenderClient(base_url)
    if client.check_health():
        logger.info("Metrics API is healthy")
        return client
    elif fallback_library_path:
        logger.warning("Metrics API not available, using local recommender")
        return LocalMetricsRecommender(fallback_library_path)
    else:
        logger.error("No metrics recommender available")
        return None
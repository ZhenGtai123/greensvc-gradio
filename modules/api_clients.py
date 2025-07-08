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
        return {
            'semantic_colors': {
                "0": [0, 0, 0],          # 背景 - 黑色
                "1": [6, 230, 230],      # Sky - 青色
                "2": [4, 250, 7],        # Grass - 绿色
                "3": [250, 127, 4],      # Trees - 橙色
                "4": [4, 200, 3],        # Shrubs - 深绿
                "5": [204, 255, 4],      # Water - 亮黄绿
                "6": [237, 117, 57],     # Land - 橙红
                "7": [220, 20, 60],      # Building - 深红
                "8": [255, 192, 203],    # Rock - 粉红
                "9": [255, 0, 255],      # People - 品红
                "10": [128, 128, 128],   # Fences - 灰色
                "11": [100, 100, 100],   # Roads - 深灰
                "12": [200, 200, 200],   # Pavements - 浅灰
                "13": [139, 69, 19],     # Bridge - 棕色
                "14": [0, 0, 255],       # Cars - 蓝色
                "15": [255, 255, 0],     # Others - 黄色
            },
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
            # 准备请求数据
            request_data = {
                "image_id": f"img_{hash(image_path)}",
                "semantic_classes": semantic_classes,
                "semantic_countability": semantic_countability,
                "openness_list": openness_list,
                "encoder": "vitb",
                **self.default_colors
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
                # 解析响应
                result = response.json()
                
                # 将hex字符串转换回图片数据
                processed_result = {}
                for key, hex_data in result.items():
                    if isinstance(hex_data, str):
                        # 将hex转换为字节
                        img_bytes = bytes.fromhex(hex_data)
                        processed_result[key] = img_bytes
                
                return {
                    'status': 'success',
                    'images': processed_result,
                    'image_path': image_path
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
    
    def download_results_zip(self, image_path: str, semantic_classes: List[str],
                           semantic_countability: List[int], openness_list: List[int]) -> Optional[bytes]:
        """下载结果的ZIP文件"""
        try:
            request_data = {
                "image_id": f"img_{hash(image_path)}",
                "semantic_classes": semantic_classes,
                "semantic_countability": semantic_countability,
                "openness_list": openness_list,
                "encoder": "vitb",
                **self.default_colors
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

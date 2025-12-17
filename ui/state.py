"""
应用状态管理模块
"""

from typing import Dict, List, Any, Optional
import pandas as pd

class AppState:
    """应用状态管理器"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置所有状态"""
        self.selected_metrics = []
        self.vision_results = {}
        self.metrics_results = pd.DataFrame()
        self.gps_data = {}
        self.components = None
        self.vision_api_url = None
    
    def set_components(self, components: Dict):
        """设置组件引用"""
        self.components = components
    
    def set_selected_metrics(self, metrics: List[Dict]):
        """设置选中的指标"""
        self.selected_metrics = metrics
    
    def add_vision_result(self, path: str, result: Dict):
        """添加视觉分析结果"""
        self.vision_results[path] = result
    
    def set_metrics_results(self, results: pd.DataFrame):
        """设置指标计算结果"""
        self.metrics_results = results
    
    def set_gps_data(self, data: Dict):
        """设置GPS数据"""
        self.gps_data = data
    
    def get_selected_metrics(self) -> List[Dict]:
        """获取选中的指标"""
        return self.selected_metrics
    
    def get_vision_results(self) -> Dict:
        """获取视觉分析结果"""
        return self.vision_results
    
    def get_metrics_results(self) -> pd.DataFrame:
        """获取指标计算结果"""
        return self.metrics_results
    
    def get_gps_data(self) -> Dict:
        """获取GPS数据"""
        return self.gps_data
    
    def has_vision_results(self) -> bool:
        """检查是否有视觉分析结果"""
        return len(self.vision_results) > 0
    
    def has_metrics_results(self) -> bool:
        """检查是否有指标计算结果"""
        return not self.metrics_results.empty
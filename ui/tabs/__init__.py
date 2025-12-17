"""
UI Tab模块初始化
"""

from .metrics_recommendation import create_metrics_recommendation_tab
from .metrics_management import create_metrics_management_tab
from .vision_analysis import create_vision_analysis_tab
from .metrics_report import create_metrics_report_tab
from .api_config import create_api_config_tab 

__all__ = [
    'create_api_config_tab',
    'create_metrics_recommendation_tab',
    'create_metrics_management_tab',
    'create_vision_analysis_tab',
    'create_metrics_report_tab'
]
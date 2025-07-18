"""
UI Tab模块初始化
"""

from .metrics_recommendation import create_metrics_recommendation_tab
from .metrics_management import create_metrics_management_tab
from .image_processing import create_image_processing_tab
from .vision_analysis import create_vision_analysis_tab
from .metrics_report import create_metrics_report_tab

__all__ = [
    'create_metrics_recommendation_tab',
    'create_metrics_management_tab',
    'create_image_processing_tab',
    'create_vision_analysis_tab',
    'create_metrics_report_tab'
]
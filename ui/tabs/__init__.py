"""UI Tabs"""

from .api_config import create_api_config_tab
from .project_questionnaire import create_project_questionnaire_tab
from .indicator_recommendation import create_indicator_recommendation_tab
from .metrics_management import create_metrics_management_tab
from .vision_analysis import create_vision_analysis_tab
from .metrics_calculation import create_metrics_calculation_tab
from .report_generation import create_report_generation_tab

__all__ = [
    'create_api_config_tab',
    'create_project_questionnaire_tab',
    'create_indicator_recommendation_tab',
    'create_metrics_management_tab',
    'create_vision_analysis_tab',
    'create_metrics_calculation_tab',
    'create_report_generation_tab'
]

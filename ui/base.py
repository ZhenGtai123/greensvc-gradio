"""
基础UI模块，创建主界面
"""

import gradio as gr
from .state import AppState
from .tabs import (
    create_api_config_tab,
    create_metrics_recommendation_tab,
    create_metrics_management_tab,
    create_vision_analysis_tab,
    create_metrics_report_tab
)

def create_main_interface(components: dict, config: dict, app_state: AppState = None):
    """
    创建主界面
    
    Args:
        components: 初始化的业务组件
        config: 配置信息
        app_state: 应用状态管理器
    """
    if app_state is None:
        app_state = AppState()
    
    app_state.set_components(components)
    
    with gr.Blocks(title="城市绿地空间视觉分析系统") as app:
        gr.Markdown("# 城市绿地空间视觉分析系统")
        gr.Markdown("通过AI与空间视觉指标相结合，为城市绿地等空间的专业分析、科学决策与设计优化提供数据驱动工具")
        
        with gr.Tabs():
            # Tab 0: API配置
            api_config_components = create_api_config_tab(components, app_state)
            
            # Tab 1: 指标推荐与选择
            metrics_rec_components = create_metrics_recommendation_tab(components, app_state)
            
            # Tab 2: 指标库管理
            metrics_mgmt_components = create_metrics_management_tab(components, app_state)
            
            # Tab 3: 视觉分析（合并了图片上传）
            vision_analysis_components = create_vision_analysis_tab(components, app_state, config)
            
            # Tab 4: 指标计算与报告
            report_components = create_metrics_report_tab(components, app_state)
        
        # 初始加载函数
        def initial_load():
            """初始加载数据"""
            from .tabs.metrics_recommendation import refresh_metrics
            from .tabs.metrics_management import update_metric_dropdown, get_metrics_code_status
            
            df_metrics, choices_update = refresh_metrics(components)
            dropdown_update = update_metric_dropdown(components)
            code_status = get_metrics_code_status(components)
            
            return df_metrics, choices_update, dropdown_update, code_status
        
        # 绑定初始加载
        app.load(
            fn=initial_load,
            outputs=[
                metrics_rec_components['metrics_library'],
                metrics_rec_components['selected_indices'],
                metrics_mgmt_components['metric_name_dropdown'],
                metrics_mgmt_components['metrics_code_status']
            ]
        )
    
    return app
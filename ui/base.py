"""
Main Interface Framework
"""

import gradio as gr
from .state import AppState
from .tabs import (
    create_api_config_tab,
    create_project_questionnaire_tab,
    create_indicator_recommendation_tab,
    create_metrics_management_tab,
    create_vision_analysis_tab,
    create_metrics_calculation_tab,
    create_report_generation_tab
)


# Custom CSS for larger UI
CUSTOM_CSS = """
/* Increase base font size */
.gradio-container {
    font-size: 16px !important;
}

/* Larger headings */
h1 { font-size: 2.2rem !important; }
h2 { font-size: 1.8rem !important; }
h3 { font-size: 1.4rem !important; }

/* Larger input fields */
input, textarea, select {
    font-size: 15px !important;
    padding: 10px !important;
}

/* Larger buttons */
button {
    font-size: 15px !important;
    padding: 12px 20px !important;
}

/* Primary button style */
.gr-button-primary, button.primary {
    background: #10b981 !important;
    font-weight: 600 !important;
}

/* Larger labels */
label {
    font-size: 14px !important;
    font-weight: 500 !important;
}

/* Tab labels */
.tab-nav button {
    font-size: 15px !important;
    padding: 12px 16px !important;
}

/* Dataframe cells */
.dataframe td, .dataframe th {
    font-size: 14px !important;
    padding: 8px !important;
}

/* Markdown text */
.prose {
    font-size: 15px !important;
    line-height: 1.6 !important;
}

/* Accordion headers */
.accordion-header {
    font-size: 15px !important;
}

/* Group spacing */
.gr-group {
    margin-bottom: 1.5rem !important;
    padding: 1rem !important;
}

/* Checkbox and radio labels */
.gr-checkbox label, .gr-radio label {
    font-size: 14px !important;
}
"""


def create_main_interface(components: dict, config, app_state: AppState = None):
    """
    Create main interface
    
    Tab Structure:
        0. API Configuration
        1. Project Questionnaire 
        2. Indicator Recommendation 
        3. Metrics Library Management
        4. Vision Analysis
        5. Metrics Calculation
        6. Report Generation
    """
    if app_state is None:
        app_state = AppState()
    
    app_state.set_components(components)
    
    with gr.Blocks(
        title="GreenSVC-AI",
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS
    ) as app:
        gr.Markdown("""
        # 🌿 GreenSVC-AI Urban Greenspace Visual Analysis System
        **Evidence-Based Landscape Design Support System v2.0**
        """)
        
        with gr.Tabs():
            create_api_config_tab(components, app_state, config)
            create_project_questionnaire_tab(components, app_state, config)
            create_indicator_recommendation_tab(components, app_state, config)
            create_metrics_management_tab(components, app_state, config)
            create_vision_analysis_tab(components, app_state, config)
            create_metrics_calculation_tab(components, app_state, config)
            create_report_generation_tab(components, app_state, config)
        
        gr.Markdown("""
        ---
        <center><small>GreenSVC-AI v2.0 | © 2024</small></center>
        """)
    
    return app

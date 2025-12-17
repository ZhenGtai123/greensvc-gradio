"""
Tab 5: 指标计算与报告
"""

import gradio as gr
import pandas as pd
import os
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

def create_metrics_report_tab(components: dict, app_state):
    """创建指标计算与报告Tab"""
    
    with gr.Tab("4. Metrics Calculation & Report"):
        gr.Markdown("### Calculate Selected Metrics")
        calc_btn = gr.Button("Calculate Metrics", variant="primary")
        calc_status = gr.Textbox(label="Calculation Status")
        metrics_results = gr.Dataframe(label="Metric Calculation Results")
        
        gr.Markdown("### Generate Analysis Report")
        with gr.Row():
            include_heatmap_final = gr.Checkbox(
                label="Include Spatial Heatmap",
                value=False
            )
            openai_key_2 = gr.Textbox(
                label="OpenAI API Key (for AI-powered analysis)",
                type="password",
                placeholder="sk-..."
            )
        
        generate_btn = gr.Button("Generate Report", variant="primary")
        report_status = gr.Textbox(label="Report Status")
        report_file = gr.File(label="Download Report")
        
        # 事件处理函数
        def calculate_metrics() -> Tuple[str, pd.DataFrame]:
            """Calculate selected metrics"""
            try:
                if not app_state.has_vision_results():
                    return "Please run vision analysis first", pd.DataFrame()
                
                selected_metrics = app_state.get_selected_metrics()
                if not selected_metrics:
                    return "Please select metrics first", pd.DataFrame()
                
                # Prepare metric info including required_images
                metrics_info = {}
                for metric in selected_metrics:
                    metric_name = metric['metric name']
                    metrics_info[metric_name] = {
                        'required_images': metric.get('required_images', ''),
                        'category': metric.get('Primary Category', ''),
                        'unit': metric.get('Unit', '')
                    }
                
                # Calculate each metric
                results = []
                errors = []
                
                for img_path, vision_result in app_state.get_vision_results().items():
                    img_metrics = {'Image': os.path.basename(img_path)}
                    
                    for metric in selected_metrics:
                        metric_name = metric['metric name']
                        metric_info = metrics_info.get(metric_name, {})
                        
                        # Use improved calculation method
                        value = components['metrics_calculator'].calculate_metric(
                            metric_name,
                            vision_result,
                            metric_info
                        )
                        
                        if value is None:
                            # Get detailed error info
                            validation = components['metrics_calculator']._validate_required_images(
                                vision_result,
                                metric_info.get('required_images', '').split(',') if metric_info.get('required_images') else []
                            )
                            if not validation['valid']:
                                errors.append(f"{metric_name}: Missing images {validation['missing']}")
                            value = 'N/A'
                        
                        img_metrics[metric_name] = value
                    
                    results.append(img_metrics)
                
                df_results = pd.DataFrame(results)
                app_state.set_metrics_results(df_results)
                
                status = "Metric calculation completed"
                if errors:
                    status += f"\n\nNote:\n" + "\n".join(errors[:5])  # Show first 5 errors
                    if len(errors) > 5:
                        status += f"\n... and {len(errors)-5} more errors"
                
                return status, df_results
                
            except Exception as e:
                return f"Calculation failed: {str(e)}", pd.DataFrame()
        
        def generate_report(include_heatmap: bool, openai_key: str) -> Tuple[str, str]:
            """Generate analysis report"""
            try:
                if not app_state.has_metrics_results():
                    return "No analysis results available", ""
                
                report_path = components['report_generator'].generate_report(
                    app_state.get_metrics_results(),
                    app_state.get_selected_metrics(),
                    app_state.get_vision_results(),
                    app_state.get_gps_data() if include_heatmap else None,
                    openai_key
                )
                
                return "Report generated successfully", report_path
                
            except Exception as e:
                return f"Generation failed: {str(e)}", ""
        
        def generate_final_report(include_heatmap, key):
            status, path = generate_report(include_heatmap, key)
            if path and os.path.exists(path):
                return status, path
            return status, None
        
        # 绑定事件
        calc_btn.click(
            fn=calculate_metrics,
            outputs=[calc_status, metrics_results]
        )
        
        generate_btn.click(
            fn=generate_final_report,
            inputs=[include_heatmap_final, openai_key_2],
            outputs=[report_status, report_file]
        )
        
        return {
            'calc_btn': calc_btn,
            'calc_status': calc_status,
            'metrics_results': metrics_results,
            'include_heatmap_final': include_heatmap_final,
            'openai_key_2': openai_key_2,
            'generate_btn': generate_btn,
            'report_status': report_status,
            'report_file': report_file
        }
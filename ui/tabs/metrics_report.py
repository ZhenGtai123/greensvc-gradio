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
    
    with gr.Tab("5. 指标计算与报告"):
        gr.Markdown("### 计算选定指标")
        calc_btn = gr.Button("计算指标", variant="primary")
        calc_status = gr.Textbox(label="计算状态")
        metrics_results = gr.Dataframe(label="指标计算结果")
        
        gr.Markdown("### 生成分析报告")
        with gr.Row():
            include_heatmap_final = gr.Checkbox(
                label="包含空间热力图",
                value=False
            )
            openai_key_2 = gr.Textbox(
                label="OpenAI API Key（用于生成分析文本）",
                type="password",
                placeholder="sk-..."
            )
        
        generate_btn = gr.Button("生成报告", variant="primary")
        report_status = gr.Textbox(label="报告状态")
        report_file = gr.File(label="下载报告")
        
        # 事件处理函数
        def calculate_metrics() -> Tuple[str, pd.DataFrame]:
            """计算选定的指标"""
            try:
                if not app_state.has_vision_results():
                    return "请先进行视觉分析", pd.DataFrame()
                
                selected_metrics = app_state.get_selected_metrics()
                if not selected_metrics:
                    return "请先选择指标", pd.DataFrame()
                
                # 准备指标信息（包含required_images）
                metrics_info = {}
                for metric in selected_metrics:
                    metric_name = metric['metric name']
                    metrics_info[metric_name] = {
                        'required_images': metric.get('required_images', ''),
                        'category': metric.get('Primary Category', ''),
                        'unit': metric.get('Unit', '')
                    }
                
                # 计算每个指标
                results = []
                errors = []
                
                for img_path, vision_result in app_state.get_vision_results().items():
                    img_metrics = {'图片': os.path.basename(img_path)}
                    
                    for metric in selected_metrics:
                        metric_name = metric['metric name']
                        metric_info = metrics_info.get(metric_name, {})
                        
                        # 使用改进的计算方法
                        value = components['metrics_calculator'].calculate_metric(
                            metric_name,
                            vision_result,
                            metric_info  # 传入指标信息
                        )
                        
                        if value is None:
                            # 获取详细的错误信息
                            validation = components['metrics_calculator']._validate_required_images(
                                vision_result,
                                metric_info.get('required_images', '').split(',') if metric_info.get('required_images') else []
                            )
                            if not validation['valid']:
                                errors.append(f"{metric_name}: 缺少图像 {validation['missing']}")
                            value = 'N/A'
                        
                        img_metrics[metric_name] = value
                    
                    results.append(img_metrics)
                
                df_results = pd.DataFrame(results)
                app_state.set_metrics_results(df_results)
                
                status = "指标计算完成"
                if errors:
                    status += f"\n\n注意：\n" + "\n".join(errors[:5])  # 显示前5个错误
                    if len(errors) > 5:
                        status += f"\n... 还有 {len(errors)-5} 个错误"
                
                return status, df_results
                
            except Exception as e:
                return f"计算失败: {str(e)}", pd.DataFrame()
        
        def generate_report(include_heatmap: bool, openai_key: str) -> Tuple[str, str]:
            """生成分析报告"""
            try:
                if not app_state.has_metrics_results():
                    return "没有可用的分析结果", ""
                
                report_path = components['report_generator'].generate_report(
                    app_state.get_metrics_results(),
                    app_state.get_selected_metrics(),
                    app_state.get_vision_results(),
                    app_state.get_gps_data() if include_heatmap else None,
                    openai_key
                )
                
                return "报告生成成功", report_path
                
            except Exception as e:
                return f"生成失败: {str(e)}", ""
        
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
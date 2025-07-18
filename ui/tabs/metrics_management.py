"""
Tab 2: 指标库管理
"""

import gradio as gr
import pandas as pd
from typing import Dict
import logging

logger = logging.getLogger(__name__)

def create_metrics_management_tab(components: dict, app_state):
    """创建指标库管理Tab"""
    
    with gr.Tab("2. 指标库管理"):
        gr.Markdown("### 更新指标库")
        with gr.Row():
            metrics_file = gr.File(label="上传新的指标库文件 (Excel)")
            update_lib_btn = gr.Button("更新指标库")
            update_status = gr.Textbox(label="更新状态")
        
        gr.Markdown("### 上传指标代码")
        gr.Markdown("上传Python代码文件，文件中需要包含 `calculate(vision_result)` 函数")
        with gr.Row():
            with gr.Column():
                # 显示所有指标的下拉列表
                metric_name_dropdown = gr.Dropdown(
                    label="选择指标",
                    choices=[],
                    interactive=True
                )
                code_file = gr.File(
                    label="Python代码文件",
                    file_types=[".py"]
                )
            with gr.Column():
                upload_code_btn = gr.Button("上传代码", variant="primary")
                upload_status = gr.Textbox(label="上传状态")
        
        gr.Markdown("### 当前指标代码状态")
        metrics_code_status = gr.Dataframe(
            label="指标代码状态",
            interactive=False
        )
        
        # 事件处理函数
        def update_metrics_library(file) -> str:
            """更新指标库文件"""
            try:
                if file is None:
                    return "请选择文件"
                
                # 保存上传的文件
                df = pd.read_excel(file.name)
                df.to_excel(components['metrics_manager'].metrics_library_path, index=False)
                
                # 重新加载
                components['metrics_manager'].reload_metrics()
                
                return "指标库已更新"
            except Exception as e:
                return f"更新失败: {str(e)}"
        
        def upload_metric_code_from_dropdown(metric_selection: str, code_file) -> str:
            """上传指标计算代码"""
            try:
                if not metric_selection:
                    return "请选择一个指标"
                
                if code_file is None:
                    return "请选择代码文件"
                
                # 从选择中提取指标名称（去掉状态标记）
                metric_name = metric_selection[2:] if metric_selection.startswith(('✓ ', '✗ ')) else metric_selection
                
                # 读取代码文件内容
                file_path = code_file.name if hasattr(code_file, 'name') else code_file
                with open(file_path, 'r', encoding='utf-8') as f:
                    code_content = f.read()
                
                # 保存代码
                success = components['metrics_manager'].save_metric_code(metric_name, code_content)
                
                if success:
                    return f"指标 '{metric_name}' 的代码已上传"
                else:
                    return "上传失败：代码中未找到必要的计算函数"
                    
            except Exception as e:
                return f"上传失败: {str(e)}"
        
        # 绑定事件
        update_lib_btn.click(
            fn=update_metrics_library,
            inputs=[metrics_file],
            outputs=[update_status]
        )
        
        # 导入需要的函数
        from .metrics_recommendation import refresh_metrics
        
        upload_code_btn.click(
            fn=upload_metric_code_from_dropdown,
            inputs=[metric_name_dropdown, code_file],
            outputs=[upload_status]
        ).then(
            fn=lambda: get_metrics_code_status(components),
            outputs=[metrics_code_status]
        ).then(
            fn=lambda: update_metric_dropdown(components),
            outputs=[metric_name_dropdown]
        )
        
        return {
            'metrics_file': metrics_file,
            'update_lib_btn': update_lib_btn,
            'update_status': update_status,
            'metric_name_dropdown': metric_name_dropdown,
            'code_file': code_file,
            'upload_code_btn': upload_code_btn,
            'upload_status': upload_status,
            'metrics_code_status': metrics_code_status
        }


def update_metric_dropdown(components):
    """更新指标下拉列表"""
    metrics = components['metrics_manager'].get_all_metrics()
    choices = []
    for metric in metrics:
        metric_name = metric['metric name']
        has_code = components['metrics_manager'].has_metric_code(metric_name)
        status = "✓" if has_code else "✗"
        choices.append(f"{status} {metric_name}")
    return gr.update(choices=choices)


def get_metrics_code_status(components):
    """获取指标代码状态"""
    metrics = components['metrics_manager'].get_all_metrics()
    status_data = []
    for metric in metrics:
        metric_name = metric['metric name']
        has_code = components['metrics_manager'].has_metric_code(metric_name)
        status_data.append({
            '指标名称': metric_name,
            '代码状态': '✓ 已上传' if has_code else '✗ 未上传',
            '类别': metric.get('Primary Category', ''),
            '数据输入': metric.get('Data Input', '')
        })
    return pd.DataFrame(status_data)
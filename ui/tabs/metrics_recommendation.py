"""
Tab 1: 指标推荐与选择
"""

import gradio as gr
import pandas as pd
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

def create_metrics_recommendation_tab(components: dict, app_state):
    """创建指标推荐与选择Tab"""
    
    with gr.Tab("1. 指标推荐与选择"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 描述您的分析需求")
                user_input = gr.Textbox(
                    label="需求描述",
                    placeholder="例如：我想分析公园的开放度和视觉层次...",
                    lines=5
                )
                openai_key_1 = gr.Textbox(
                    label="OpenAI API Key",
                    type="password",
                    placeholder="sk-..."
                )
                recommend_btn = gr.Button("获取AI推荐", variant="primary")
            
            with gr.Column():
                recommendation_text = gr.Textbox(label="推荐说明", lines=3)
                recommended_metrics = gr.Dataframe(label="推荐的指标")
        
        gr.Markdown("### 查看完整指标库")
        with gr.Row():
            refresh_btn = gr.Button("刷新指标库")
            metrics_library = gr.Dataframe(
                label="所有可用指标",
                interactive=False,
            )
        
        gr.Markdown("### 选择要使用的指标")
        gr.Markdown("只有已上传代码的指标才能被选择")
        selected_indices = gr.CheckboxGroup(
            label="选择指标（勾选要使用的指标）",
            choices=[],
            value=[],
            interactive=True
        )
        select_btn = gr.Button("确认选择", variant="primary")
        selection_status = gr.Textbox(label="选择状态")
        
        # 事件处理函数
        def recommend_metrics(user_input: str, openai_key: str) -> Tuple[str, pd.DataFrame]:
            """根据用户输入推荐指标"""
            try:
                if not user_input:
                    return "请输入您的需求描述", pd.DataFrame()
                
                # 调用推荐API
                recommendations = components['metrics_client'].recommend_metrics(
                    user_input, 
                    openai_api_key=openai_key
                )
                
                # 解析推荐结果
                if recommendations and 'recommendation' in recommendations:
                    rec_list = eval(recommendations['recommendation'])
                    df = pd.DataFrame(rec_list)
                    
                    # 返回推荐说明和数据框
                    return "推荐的指标如下，您可以在下方查看并选择", df
                else:
                    return "推荐失败，请检查API密钥", pd.DataFrame()
                    
            except Exception as e:
                return f"错误: {str(e)}", pd.DataFrame()
        
        def select_metrics_wrapper(selected_items: List[str]) -> str:
            """选择要使用的指标"""
            try:
                if not selected_items:
                    return "请选择至少一个指标"
                
                all_metrics = components['metrics_manager'].get_all_metrics()
                selected_metrics = []
                skipped_metrics = []
                
                for selection in selected_items:
                    # 提取索引号
                    idx = int(selection.split(':')[0])
                    metric = all_metrics[idx]
                    metric_name = metric['metric name']
                    
                    # 检查是否有代码
                    if components['metrics_manager'].has_metric_code(metric_name):
                        selected_metrics.append(metric)
                    else:
                        skipped_metrics.append(metric_name)
                
                app_state.set_selected_metrics(selected_metrics)
                
                status = f"已选择 {len(selected_metrics)} 个指标"
                if skipped_metrics:
                    status += f"\n跳过了 {len(skipped_metrics)} 个没有代码的指标: {', '.join(skipped_metrics[:3])}"
                    if len(skipped_metrics) > 3:
                        status += f" 等"
                
                return status
            except Exception as e:
                return f"选择失败: {str(e)}"
        
        # 绑定事件
        recommend_btn.click(
            fn=recommend_metrics,
            inputs=[user_input, openai_key_1],
            outputs=[recommendation_text, recommended_metrics]
        )
        
        refresh_btn.click(
            fn=lambda: refresh_metrics(components),
            outputs=[metrics_library, selected_indices]
        )
        
        select_btn.click(
            fn=select_metrics_wrapper,
            inputs=[selected_indices],
            outputs=[selection_status]
        )
        
        return {
            'user_input': user_input,
            'openai_key_1': openai_key_1,
            'recommend_btn': recommend_btn,
            'recommendation_text': recommendation_text,
            'recommended_metrics': recommended_metrics,
            'refresh_btn': refresh_btn,
            'metrics_library': metrics_library,
            'selected_indices': selected_indices,
            'select_btn': select_btn,
            'selection_status': selection_status
        }


def refresh_metrics(components):
    """刷新指标库"""
    df = components['metrics_manager'].load_metrics()
    if df.empty:
        return df, gr.update(choices=[], value=[])
    
    # 检查每个指标的状态
    choices = []
    selectable_choices = []
    metrics_with_status = []
    
    for i, row in df.iterrows():
        metric_name = row['metric name']
        has_code = components['metrics_manager'].has_metric_code(metric_name)
        required_images = row.get('required_images', '')
        
        # 创建带状态的指标数据
        row_dict = row.to_dict()
        row_dict['代码状态'] = '✓ 已上传' if has_code else '✗ 未上传'
        row_dict['所需图像'] = required_images if required_images else '未配置'
        metrics_with_status.append(row_dict)
        
        # 创建选项文本
        if has_code:
            choice_text = f"{i}: {metric_name} [可用]"
            if required_images:
                choice_text += f" (需要: {required_images})"
            choices.append(choice_text)
            selectable_choices.append(choice_text)
        else:
            choice_text = f"{i}: {metric_name} [需要上传代码]"
            choices.append(choice_text)
    
    # 创建新的DataFrame
    df_with_status = pd.DataFrame(metrics_with_status)
    
    # 重新排列列
    cols = df_with_status.columns.tolist()
    priority_cols = ['代码状态', '所需图像', 'metric name']
    for col in priority_cols:
        if col in cols:
            cols.remove(col)
    cols = priority_cols + cols
    df_with_status = df_with_status[cols]
    
    return df_with_status, gr.update(choices=selectable_choices, value=[])
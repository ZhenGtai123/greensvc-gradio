"""
Tab 3: 图片上传与处理
"""

import gradio as gr
import pandas as pd
import os
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

def create_image_processing_tab(components: dict, app_state):
    """创建图片上传与处理Tab"""
    
    with gr.Tab("3. 图片上传与处理"):
        gr.Markdown("### 上传待分析图片")
        image_files = gr.File(
            label="选择图片文件",
            file_count="multiple",
            file_types=["image"]
        )
        process_btn = gr.Button("处理图片", variant="primary")
        
        with gr.Row():
            process_status = gr.Textbox(label="处理状态")
            gps_info = gr.Dataframe(label="GPS信息")
        
        enable_heatmap = gr.Checkbox(
            label="生成空间热力图（需要所有图片都有GPS信息）",
            value=False,
            interactive=False
        )
        
        # 事件处理函数
        def process_images(files) -> Tuple[str, pd.DataFrame, bool]:
            """处理上传的图片"""
            try:
                if not files:
                    return "请上传图片", pd.DataFrame(), False
                
                # 处理图片
                results = components['image_processor'].batch_process_images([f.name for f in files])
                app_state.processed_images = results
                
                # 提取GPS信息
                gps_info = []
                for path, info in results.items():
                    gps_info.append({
                        '文件名': os.path.basename(path),
                        '有GPS': '是' if info['has_gps'] else '否',
                        '纬度': info['gps'][0] if info['has_gps'] else None,
                        '经度': info['gps'][1] if info['has_gps'] else None,
                        '状态': info['status']
                    })
                
                df_gps = pd.DataFrame(gps_info)
                
                # 检查是否所有图片都有GPS
                all_have_gps = all(info['has_gps'] for info in results.values())
                
                # 保存GPS数据
                gps_data = {
                    'all_have_gps': all_have_gps,
                    'locations': [(info['gps'][0], info['gps'][1]) 
                                 for info in results.values() if info['has_gps']]
                }
                app_state.set_gps_data(gps_data)
                
                status = f"已处理 {len(files)} 张图片"
                if all_have_gps:
                    status += "\n所有图片都包含GPS信息，可以生成空间热力图"
                
                return status, df_gps, all_have_gps
                
            except Exception as e:
                return f"处理失败: {str(e)}", pd.DataFrame(), False
        
        # 绑定事件
        process_btn.click(
            fn=process_images,
            inputs=[image_files],
            outputs=[process_status, gps_info, enable_heatmap]
        )
        
        return {
            'image_files': image_files,
            'process_btn': process_btn,
            'process_status': process_status,
            'gps_info': gps_info,
            'enable_heatmap': enable_heatmap
        }
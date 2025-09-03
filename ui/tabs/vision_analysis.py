"""
Tab 4: 视觉分析
简化版 - 适配Google Colab API
"""

import gradio as gr
import os
import logging

logger = logging.getLogger(__name__)

# 预设配置
PRESET_CONFIGS = {
    "默认配置（41类园林）": {
        "classes": "sky\nlawn\nherbaceous plants\ntrees\nshrubs\nwater\nground / land\nbuilding\nrock / stone\nperson / people\nfence / railing\nroad / highway\npavement / path / trail\nbridge\nvehicle / car\nchair / bench\nbase / pedestal\nsteps / curb\nrailing / barrier\nsign / plaque\nbin / trash can\ntower\nawning / pavilion / shade structure\nstreet light / lamp post\nboat\nfountain\nbicycle\nsculpture / outdoor art\npier / dock\naquatic plants\ngreen-covered building\ncouplet\nriverbank\nhill / mountain\nconstruction equipment\npole\nanimal\nmonument\ndoor\noutdoor sports equipment\nwaterfall",
        "countability": "0,0,1,1,1,0,0,1,1,1,0,0,0,0,1,1,1,0,0,1,1,1,1,1,1,1,1,1,0,1,1,1,0,0,1,1,1,1,1,1,0",
        "openness": "0,0,1,1,1,0,0,1,1,1,1,0,0,0,1,1,1,0,1,1,1,1,1,1,1,0,1,1,0,0,0,1,0,0,1,1,1,1,1,1,0"
    },
    "简单配置（8类）": {
        "classes": "sky\ngrass\ntrees\nbuilding\nwater\nperson\nroad\nvehicle",
        "countability": "0,0,1,1,0,1,0,1",
        "openness": "0,0,1,1,0,1,0,1"
    }
}

def create_vision_analysis_tab(components: dict, app_state, config: dict):
    """创建视觉分析Tab"""
    
    with gr.Tab("4. 视觉分析"):
        gr.Markdown("""
        ### 🎯 视觉分析
        使用AI模型进行语义分割、深度估计和前中背景分割。
        """)
        
        # API状态检查
        with gr.Row():
            api_status = gr.Textbox(
                label="API状态",
                value="未检查",
                interactive=False
            )
            check_api_btn = gr.Button("检查API", variant="secondary")
        
        # 预设配置选择
        with gr.Row():
            preset_dropdown = gr.Dropdown(
                label="选择预设配置",
                choices=list(PRESET_CONFIGS.keys()),
                value="默认配置（41类园林）"
            )
            apply_preset_btn = gr.Button("应用预设", variant="secondary")
        
        # 参数输入（折叠显示）
        with gr.Accordion("参数配置", open=False):
            semantic_classes = gr.Textbox(
                label="语义类别（每行一个）",
                lines=8,
                value=PRESET_CONFIGS["默认配置（41类园林）"]["classes"]
            )
            
            with gr.Row():
                semantic_countability = gr.Textbox(
                    label="可数性（0或1，逗号分隔）",
                    value=PRESET_CONFIGS["默认配置（41类园林）"]["countability"]
                )
                openness_list = gr.Textbox(
                    label="开放度（0或1，逗号分隔）",
                    value=PRESET_CONFIGS["默认配置（41类园林）"]["openness"]
                )
        
        # 编码器选项
        encoder_type = gr.Radio(
            label="模型大小",
            choices=[("标准", "vitb"), ("轻量", "vits")],
            value="vitb"
        )
        
        # 分析按钮
        analyze_btn = gr.Button("🚀 开始分析", variant="primary", size="lg")
        
        # 状态和进度
        analysis_status = gr.Textbox(
            label="分析状态",
            lines=2,
            interactive=False
        )
        
        # 结果展示
        result_gallery = gr.Gallery(
            label="分析结果",
            columns=4,
            rows=2,
            object_fit="contain",
            height="auto"
        )
        
        # 统计信息
        stats_text = gr.Textbox(
            label="分析统计",
            lines=5,
            interactive=False
        )
        
        # 事件处理函数
        def check_api():
            """检查API状态"""
            try:
                vision_client = components.get('vision_client')
                if vision_client and vision_client.check_health():
                    return f"✅ API正常 ({vision_client.base_url})"
                return "❌ API未连接"
            except:
                return "❌ 无法连接API"
        
        def apply_preset(preset_name):
            """应用预设"""
            if preset_name in PRESET_CONFIGS:
                cfg = PRESET_CONFIGS[preset_name]
                return cfg["classes"], cfg["countability"], cfg["openness"]
            return "", "", ""
        
        def run_analysis(classes_text, countability_text, openness_text, encoder):
            """执行分析 - 修复版"""
            try:
                # 检查是否有图片
                if not app_state.has_processed_images():
                    return "请先上传图片", [], ""
                
                # 解析参数
                classes = [c.strip() for c in classes_text.split('\n') if c.strip()]
                countability = [int(x.strip()) for x in countability_text.split(',')]
                openness = [int(x.strip()) for x in openness_text.split(',')]
                
                # 验证参数长度
                if len(countability) != len(classes):
                    return "❌ 可数性参数数量与类别数不匹配", [], ""
                if len(openness) != len(classes):
                    return "❌ 开放度参数数量与类别数不匹配", [], ""
                
                vision_client = components.get('vision_client')
                if not vision_client:
                    return "❌ 视觉客户端未初始化", [], ""
                
                # 处理图片
                display_images = []
                stats_info = {
                    'total_images': 0,
                    'success': 0,
                    'classes_detected': set()
                }
                
                processed_images = app_state.get_processed_images()
                for path, info in processed_images.items():
                    if info['status'] == 'success':
                        stats_info['total_images'] += 1
                        
                        # 调用API - 只传递支持的参数
                        result = vision_client.analyze_image(
                            info['processed_path'],
                            classes,
                            countability,
                            openness,
                            encoder=encoder
                        )
                        
                        if result.get('status') == 'success':
                            stats_info['success'] += 1
                            app_state.add_vision_result(path, result)
                            
                            # 保存关键图像用于显示
                            if 'images' in result:
                                img_name = os.path.basename(path).split('.')[0]
                                save_dir = os.path.join(config['temp_dir'], f'vision_{img_name}')
                                os.makedirs(save_dir, exist_ok=True)
                                
                                # 只显示4种主要结果
                                for img_type in ['semantic_map', 'depth_map', 'fmb_map', 'openness_map']:
                                    if img_type in result['images']:
                                        img_path = os.path.join(save_dir, f'{img_type}.png')
                                        with open(img_path, 'wb') as f:
                                            f.write(result['images'][img_type])
                                        
                                        type_names = {
                                            'semantic_map': '语义分割',
                                            'depth_map': '深度图',
                                            'fmb_map': '前中背景',
                                            'openness_map': '开放度'
                                        }
                                        display_images.append((img_path, f"{img_name}-{type_names[img_type]}"))
                            
                            # 收集检测到的类别
                            if 'statistics' in result and 'class_statistics' in result['statistics']:
                                stats_info['classes_detected'].update(result['statistics']['class_statistics'].keys())
                
                # 生成统计文本
                stats_summary = f"处理图片: {stats_info['success']}/{stats_info['total_images']}\n"
                stats_summary += f"检测到的类别: {len(stats_info['classes_detected'])}个\n"
                if stats_info['classes_detected']:
                    detected_list = list(stats_info['classes_detected'])[:10]
                    stats_summary += f"包含: {', '.join(detected_list)}"
                    if len(stats_info['classes_detected']) > 10:
                        stats_summary += f" 等{len(stats_info['classes_detected'])}个类别"
                
                status = f"✅ 分析完成！成功处理 {stats_info['success']} 张图片"
                
                return status, display_images, stats_summary
                
            except Exception as e:
                logger.error(f"Analysis error: {e}")
                # 确保返回3个值
                return f"❌ 分析失败: {str(e)}", [], ""
        
        # 绑定事件
        check_api_btn.click(check_api, outputs=api_status)
        apply_preset_btn.click(
            apply_preset,
            inputs=preset_dropdown,
            outputs=[semantic_classes, semantic_countability, openness_list]
        )
        
        analyze_btn.click(
            run_analysis,
            inputs=[
                semantic_classes, 
                semantic_countability, 
                openness_list,
                encoder_type
            ],
            outputs=[analysis_status, result_gallery, stats_text]
        )
        
        # 初始检查
        api_status.value = check_api()
        
        return {
            'analyze_btn': analyze_btn,
            'status': analysis_status,
            'gallery': result_gallery
        }
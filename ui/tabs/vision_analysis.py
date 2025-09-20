"""
Tab 4: 视觉分析
增强版 - 支持语义颜色配置
"""

import gradio as gr
import os
import logging
import json
import requests
from typing import Dict, List, Tuple

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

# 默认颜色配置
DEFAULT_COLORS = {
    "sky": "#06e6e6",
    "lawn": "#04fa07",
    "herbaceous plants": "#fa7f04",
    "trees": "#04c803",
    "shrubs": "#ccff04",
    "water": "#0907e6",
    "ground / land": "#787846",
    "building": "#b47878",
    "rock / stone": "#ff290a",
    "person / people": "#96053d",
    "fence / railing": "#787878",
    "road / highway": "#8c8c8c",
    "pavement / path / trail": "#ebff07",
    "bridge": "#ff5200",
    "vehicle / car": "#0066c8",
    "chair / bench": "#cc4603",
    "base / pedestal": "#ff1f00",
    "steps / curb": "#ffe000",
    "railing / barrier": "#ffb806",
    "sign / plaque": "#ff0599",
    "bin / trash can": "#ad00ff",
    "tower": "#ffb8b8",
    "awning / pavilion / shade structure": "#ffd000",
    "street light / lamp post": "#0047ff",
    "boat": "#ffeb00",
    "fountain": "#08b8aa",
    "bicycle": "#fff500",
    "sculpture / outdoor art": "#ffff00",
    "pier / dock": "#4700ff",
    "aquatic plants": "#4eff00",
    "green-covered building": "#00ff4e",
    "couplet": "#82513e",
    "riverbank": "#e2c8a0",
    "hill / mountain": "#8fff8c",
    "construction equipment": "#ff7104",
    "pole": "#b5a6ae",
    "animal": "#6edca7",
    "monument": "#484846",
    "door": "#36283b",
    "outdoor sports equipment": "#37393a",
    "waterfall": "#27c4c4",
    "grass": "#04fa07",
    "person": "#96053d",
    "road": "#8c8c8c",
    "vehicle": "#0066c8"
}

def hex_to_rgb(hex_color: str, bgr_mode: bool = True) -> List[int]:
    """将16进制颜色转换为RGB或BGR列表
    
    Args:
        hex_color: 16进制颜色字符串
        bgr_mode: 如果为True，返回BGR格式（用于OpenCV）
    """
    hex_color = hex_color.lstrip('#')
    r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    if bgr_mode:
        # OpenCV使用BGR格式
        return [b, g, r]
    else:
        # 标准RGB格式
        return [r, g, b]

def rgb_to_hex(rgb: List[int], from_bgr: bool = False) -> str:
    """将RGB或BGR列表转换为16进制颜色
    
    Args:
        rgb: RGB或BGR颜色列表
        from_bgr: 如果为True，输入是BGR格式
    """
    if from_bgr:
        # 如果输入是BGR，转换为RGB
        b, g, r = rgb
        return '#{:02x}{:02x}{:02x}'.format(r, g, b)
    else:
        return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

def analyze_image_with_colors(vision_client, image_path: str, classes: List[str], 
                              countability: List[int], openness: List[int],
                              encoder: str, semantic_colors: Dict, 
                              enable_hole_fill: bool, enable_blur: bool) -> Dict:
    """使用自定义颜色调用视觉分析API"""
    try:
        import requests
        
        # 记录调用信息
        logger.info(f"Calling API with image: {image_path}")
        logger.info(f"Classes count: {len(classes)}, Encoder: {encoder}")
        
        # 准备请求数据
        request_data = {
            "image_id": f"img_{os.path.basename(image_path).split('.')[0]}",
            "semantic_classes": classes,
            "semantic_countability": countability,
            "openness_list": openness,
            "encoder": encoder,
            "semantic_colors": semantic_colors,
            "enable_hole_filling": enable_hole_fill,
            "enable_median_blur": enable_blur
        }
        
        # 发送请求到API
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
            data = {'request_data': json.dumps(request_data)}
            
            logger.info(f"Sending POST request to: {vision_client.base_url}/analyze")
            response = requests.post(
                f"{vision_client.base_url}/analyze",
                files=files,
                data=data,
                timeout=600
            )
        
        logger.info(f"Response status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"API response status: {result.get('status')}")
            
            # 处理图像数据（从hex转换为bytes）
            if result.get('status') == 'success' and 'images' in result:
                processed_images = {}
                for key, hex_data in result['images'].items():
                    if isinstance(hex_data, str):
                        img_bytes = bytes.fromhex(hex_data)
                        processed_images[key] = img_bytes
                result['images'] = processed_images
                logger.info(f"Successfully processed {len(processed_images)} images")
            else:
                logger.warning(f"API returned non-success status or no images: {result.get('status')}")
                if 'error' in result:
                    logger.error(f"API error: {result['error']}")
            
            return result
        else:
            error_msg = f"API返回错误: {response.status_code}"
            try:
                error_detail = response.json()
                error_msg += f" - {error_detail.get('detail', response.text[:200])}"
            except:
                error_msg += f" - {response.text[:200]}"
            
            logger.error(error_msg)
            return {
                'status': 'error',
                'error': error_msg
            }
            
    except Exception as e:
        logger.error(f"API call exception: {e}", exc_info=True)
        return {
            'status': 'error',
            'error': str(e)
        }

def create_vision_analysis_tab(components: dict, app_state, config: dict):
    """创建视觉分析Tab"""
    
    with gr.Tab("视觉分析"):  # 使用简单的名称，不带数字
        # 存储用户自定义颜色的状态
        custom_colors = gr.State({})
        
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
        
        # 颜色配置（新增部分）
        with gr.Accordion("🎨 颜色配置", open=False):
            gr.Markdown("""
            **颜色配置说明：**
            - 点击"生成颜色配置"按钮查看当前类别对应的颜色
            - 在下方的文本框中修改颜色代码（格式：类别名=颜色代码）
            - 颜色代码支持16进制格式（如 #FF0000）
            - 修改后点击"应用颜色配置"来更新颜色
            """)
            
            generate_colors_btn = gr.Button("🎨 生成颜色配置", variant="secondary")
            
            # 颜色配置显示
            color_config_display = gr.HTML("")
            
            # 颜色编辑区域
            color_edit_text = gr.Textbox(
                label="编辑颜色配置（每行一个：类别名=#颜色代码）",
                lines=10,
                visible=False,
                placeholder="sky=#06e6e6\nlawn=#04fa07\ntrees=#04c803"
            )
            
            # 应用按钮
            apply_colors_btn = gr.Button("应用颜色配置", variant="secondary", visible=False)
            
            # 颜色预览
            with gr.Row():
                color_preview_btn = gr.Button("预览颜色映射", variant="secondary", visible=False)
                reset_colors_btn = gr.Button("重置为默认颜色", variant="secondary", visible=False)
            
            color_preview_image = gr.Image(label="颜色映射预览", visible=False)
        
        # 编码器选项
        encoder_type = gr.Radio(
            label="模型大小",
            choices=[("标准", "vitb"), ("轻量", "vits")],
            value="vitb"
        )
        
        # 高级选项（新增）
        with gr.Accordion("⚙️ 高级选项", open=False):
            enable_hole_filling = gr.Checkbox(
                label="启用智能空洞填充",
                value=False
            )
            enable_median_blur = gr.Checkbox(
                label="启用中值滤波平滑",
                value=True
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
            label="分析结果（全部20张图片）",
            columns=5,  # 增加列数以适应更多图片
            rows=4,     # 调整行数
            object_fit="contain",
            height="auto",
            show_label=True,
            elem_id="vision_results_gallery"
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
        
        def generate_color_config(classes_text, current_custom_colors):
            """生成颜色配置界面"""
            try:
                classes = [c.strip() for c in classes_text.split('\n') if c.strip()]
                if not classes:
                    return "", "", current_custom_colors, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
                
                # 初始化或更新自定义颜色字典
                if not current_custom_colors:
                    current_custom_colors = {}
                
                # 为每个类别设置颜色
                color_text_lines = []
                for i, cls in enumerate(classes):
                    if cls not in current_custom_colors:
                        # 使用预设颜色或生成新颜色
                        if cls in DEFAULT_COLORS:
                            current_custom_colors[cls] = DEFAULT_COLORS[cls]
                        else:
                            # 为新类别生成颜色
                            hue = (i * 360 / len(classes)) % 360
                            current_custom_colors[cls] = f"#{int(hue/360*255):02x}{int((1-abs((hue/60)%2-1))*255):02x}{128:02x}"
                    
                    color_text_lines.append(f"{cls}={current_custom_colors[cls]}")
                
                # 生成HTML表格显示颜色配置
                html = """
                <style>
                    .color-table { width: 100%; border-collapse: collapse; margin: 10px 0; }
                    .color-table th, .color-table td { 
                        padding: 8px; 
                        border: 1px solid #ddd; 
                        text-align: left; 
                    }
                    .color-table th { background-color: #f5f5f5; font-weight: bold; }
                    .color-preview { 
                        width: 60px; 
                        height: 25px; 
                        border: 1px solid #ccc; 
                        display: inline-block; 
                        vertical-align: middle;
                    }
                    .class-index { color: #666; font-size: 0.9em; }
                </style>
                <table class="color-table">
                    <thead>
                        <tr>
                            <th width="10%">序号</th>
                            <th width="40%">类别名称</th>
                            <th width="25%">当前颜色</th>
                            <th width="25%">颜色代码</th>
                        </tr>
                    </thead>
                    <tbody>
                """
                
                for i, cls in enumerate(classes, 1):
                    color = current_custom_colors.get(cls, "#808080")
                    html += f"""
                        <tr>
                            <td class="class-index">{i}</td>
                            <td><strong>{cls}</strong></td>
                            <td><span class="color-preview" style="background-color: {color};"></span></td>
                            <td><code>{color}</code></td>
                        </tr>
                    """
                
                html += """
                    </tbody>
                </table>
                <p style="color: #666; font-size: 0.9em;">
                    💡 提示：在下方文本框中修改颜色，格式为 "类别名=#颜色代码"，然后点击"应用颜色配置"
                </p>
                """
                
                # 生成可编辑的文本
                color_edit_text = "\n".join(color_text_lines)
                
                # 显示编辑框和按钮
                return (
                    html, 
                    color_edit_text, 
                    current_custom_colors,
                    gr.update(visible=True),  # color_edit_text
                    gr.update(visible=True),  # apply_colors_btn
                    gr.update(visible=True),  # color_preview_btn
                    gr.update(visible=True)   # reset_colors_btn
                )
                
            except Exception as e:
                logger.error(f"Error generating color config: {e}")
                return (
                    "生成颜色配置失败", 
                    "", 
                    current_custom_colors,
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False)
                )
        
        def apply_color_config(color_text, classes_text):
            """应用用户编辑的颜色配置"""
            try:
                new_colors = {}
                lines = color_text.strip().split('\n')
                
                for line in lines:
                    if '=' in line:
                        parts = line.split('=', 1)
                        if len(parts) == 2:
                            cls = parts[0].strip()
                            color = parts[1].strip()
                            # 验证颜色格式
                            if color.startswith('#') and len(color) in [4, 7]:
                                new_colors[cls] = color
                
                # 重新生成显示
                classes = [c.strip() for c in classes_text.split('\n') if c.strip()]
                html = generate_color_display(classes, new_colors)
                
                return new_colors, html, "✅ 颜色配置已更新"
                
            except Exception as e:
                logger.error(f"Error applying color config: {e}")
                return gr.State(), "", f"❌ 应用失败: {str(e)}"
        
        def generate_color_display(classes, colors_dict):
            """生成颜色显示HTML"""
            html = """
            <style>
                .color-table { width: 100%; border-collapse: collapse; margin: 10px 0; }
                .color-table th, .color-table td { 
                    padding: 8px; 
                    border: 1px solid #ddd; 
                    text-align: left; 
                }
                .color-table th { background-color: #f5f5f5; font-weight: bold; }
                .color-preview { 
                    width: 60px; 
                    height: 25px; 
                    border: 1px solid #ccc; 
                    display: inline-block; 
                    vertical-align: middle;
                }
                .class-index { color: #666; font-size: 0.9em; }
            </style>
            <table class="color-table">
                <thead>
                    <tr>
                        <th width="10%">序号</th>
                        <th width="40%">类别名称</th>
                        <th width="25%">当前颜色</th>
                        <th width="25%">颜色代码</th>
                    </tr>
                </thead>
                <tbody>
            """
            
            for i, cls in enumerate(classes, 1):
                color = colors_dict.get(cls, "#808080")
                html += f"""
                    <tr>
                        <td class="class-index">{i}</td>
                        <td><strong>{cls}</strong></td>
                        <td><span class="color-preview" style="background-color: {color};"></span></td>
                        <td><code>{color}</code></td>
                    </tr>
                """
            
            html += """
                </tbody>
            </table>
            """
            return html
        
        def preview_colors(classes_text, current_custom_colors):
            """生成颜色映射预览图"""
            try:
                import numpy as np
                from PIL import Image, ImageDraw, ImageFont
                
                classes = [c.strip() for c in classes_text.split('\n') if c.strip()]
                if not classes:
                    return None
                
                # 创建预览图
                cols = 4
                rows = (len(classes) + cols - 1) // cols
                cell_width = 200
                cell_height = 40
                
                img_width = cols * cell_width
                img_height = rows * cell_height
                
                img = Image.new('RGB', (img_width, img_height), 'white')
                draw = ImageDraw.Draw(img)
                
                for i, cls in enumerate(classes):
                    row = i // cols
                    col = i % cols
                    x = col * cell_width
                    y = row * cell_height
                    
                    # 获取颜色
                    if cls in current_custom_colors:
                        color_hex = current_custom_colors[cls]
                    elif cls in DEFAULT_COLORS:
                        color_hex = DEFAULT_COLORS[cls]
                    else:
                        color_hex = "#808080"
                    
                    # 绘制颜色块
                    color_rgb = tuple(hex_to_rgb(color_hex))
                    draw.rectangle([x + 5, y + 5, x + 35, y + 35], fill=color_rgb, outline='black')
                    
                    # 绘制类别名称
                    text = f"{i+1}. {cls[:20]}"
                    draw.text((x + 40, y + 12), text, fill='black')
                
                return img
                
            except Exception as e:
                logger.error(f"Error creating color preview: {e}")
                return None
        
        def reset_colors(classes_text):
            """重置为默认颜色"""
            classes = [c.strip() for c in classes_text.split('\n') if c.strip()]
            new_custom_colors = {}
            for cls in classes:
                if cls in DEFAULT_COLORS:
                    new_custom_colors[cls] = DEFAULT_COLORS[cls]
            return new_custom_colors
        
        def run_analysis(classes_text, countability_text, openness_text, encoder, 
                        current_custom_colors, enable_hole_fill, enable_blur):
            """执行分析 - 增强版"""
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
                
                # 准备颜色配置（转换为API需要的格式）
                semantic_colors = {}
                for i, cls in enumerate(classes):
                    if cls in current_custom_colors:
                        color_hex = current_custom_colors[cls]
                    elif cls in DEFAULT_COLORS:
                        color_hex = DEFAULT_COLORS[cls]
                    else:
                        # 生成默认颜色
                        hue = (i * 360 / len(classes)) % 360
                        color_hex = f"#{int(hue/360*255):02x}{int((1-abs((hue/60)%2-1))*255):02x}{128:02x}"
                    
                    # 转换为BGR列表格式（OpenCV使用BGR而不是RGB）
                    # API需要的格式是 {"1": [b,g,r], "2": [b,g,r], ...}
                    semantic_colors[str(i+1)] = hex_to_rgb(color_hex, bgr_mode=True)
                
                # 添加背景颜色（索引0）
                semantic_colors["0"] = [0, 0, 0]
                
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
                        logger.info(f"Processing image {stats_info['total_images']}: {path}")
                        
                        # 调用API（使用自定义analyze_image_with_colors方法）
                        result = analyze_image_with_colors(
                            vision_client,
                            info['processed_path'],
                            classes,
                            countability,
                            openness,
                            encoder,
                            semantic_colors,
                            enable_hole_fill,
                            enable_blur
                        )
                        
                        if result.get('status') == 'success':
                            stats_info['success'] += 1
                            app_state.add_vision_result(path, result)
                            logger.info(f"Successfully analyzed image: {path}")
                            
                            # 保存所有图像用于显示
                            if 'images' in result:
                                img_name = os.path.basename(path).split('.')[0]
                                save_dir = os.path.join(config['temp_dir'], f'vision_{img_name}')
                                os.makedirs(save_dir, exist_ok=True)
                                
                                # 定义所有图片类型的中文名称
                                type_names = {
                                    'semantic_map': '语义分割',
                                    'depth_map': '深度图',
                                    'fmb_map': '前中背景',
                                    'openness_map': '开放度',
                                    'foreground_map': '前景掩码',
                                    'middleground_map': '中景掩码',
                                    'background_map': '背景掩码',
                                    'original': '原图',
                                    'semantic_foreground': '语义-前景',
                                    'semantic_middleground': '语义-中景',
                                    'semantic_background': '语义-背景',
                                    'depth_foreground': '深度-前景',
                                    'depth_middleground': '深度-中景',
                                    'depth_background': '深度-背景',
                                    'openness_foreground': '开放度-前景',
                                    'openness_middleground': '开放度-中景',
                                    'openness_background': '开放度-背景',
                                    'original_foreground': '原图-前景',
                                    'original_middleground': '原图-中景',
                                    'original_background': '原图-背景'
                                }
                                
                                # 保存并显示所有图片
                                for img_type, img_data in result['images'].items():
                                    if img_data:  # 确保有数据
                                        img_path = os.path.join(save_dir, f'{img_type}.png')
                                        with open(img_path, 'wb') as f:
                                            f.write(img_data)
                                        
                                        # 使用中文名称，如果没有定义则使用原始名称
                                        display_name = type_names.get(img_type, img_type)
                                        display_images.append((img_path, f"{img_name}-{display_name}"))
                                
                                logger.info(f"Saved {len(result['images'])} images for {img_name}")
                            
                            # 收集检测到的类别
                            if 'statistics' in result and 'class_statistics' in result['statistics']:
                                stats_info['classes_detected'].update(result['statistics']['class_statistics'].keys())
                        else:
                            logger.error(f"Failed to analyze image {path}: {result.get('error', 'Unknown error')}")
                            # 记录失败原因
                            if 'error' in result:
                                logger.error(f"Error details: {result['error']}")
                    else:
                        logger.warning(f"Skipping image with status {info['status']}: {path}")
                
                # 生成统计文本
                stats_summary = f"处理图片: {stats_info['success']}/{stats_info['total_images']}\n"
                stats_summary += f"检测到的类别: {len(stats_info['classes_detected'])}个\n"
                if stats_info['classes_detected']:
                    detected_list = list(stats_info['classes_detected'])[:10]
                    stats_summary += f"包含: {', '.join(detected_list)}"
                    if len(stats_info['classes_detected']) > 10:
                        stats_summary += f" 等{len(stats_info['classes_detected'])}个类别"
                
                if enable_hole_fill:
                    stats_summary += "\n✅ 已启用智能空洞填充"
                if enable_blur:
                    stats_summary += "\n✅ 已启用中值滤波平滑"
                
                status = f"✅ 分析完成，成功处理 {stats_info['success']} 张图片"
                
                return status, display_images, stats_summary
                
            except Exception as e:
                logger.error(f"Analysis error: {e}")
                return f"❌ 分析失败: {str(e)}", [], ""
        
        # 绑定事件
        check_api_btn.click(check_api, outputs=api_status)
        
        apply_preset_btn.click(
            apply_preset,
            inputs=preset_dropdown,
            outputs=[semantic_classes, semantic_countability, openness_list]
        )
        
        generate_colors_btn.click(
            generate_color_config,
            inputs=[semantic_classes, custom_colors],
            outputs=[
                color_config_display,
                color_edit_text,
                custom_colors,
                color_edit_text,  # visibility
                apply_colors_btn,  # visibility
                color_preview_btn, # visibility
                reset_colors_btn   # visibility
            ]
        )
        
        apply_colors_btn.click(
            apply_color_config,
            inputs=[color_edit_text, semantic_classes],
            outputs=[custom_colors, color_config_display, analysis_status]
        )
        
        color_preview_btn.click(
            preview_colors,
            inputs=[semantic_classes, custom_colors],
            outputs=color_preview_image
        ).then(
            lambda: gr.update(visible=True),
            outputs=color_preview_image
        )
        
        reset_colors_btn.click(
            reset_colors,
            inputs=semantic_classes,
            outputs=custom_colors
        ).then(
            generate_color_config,
            inputs=[semantic_classes, custom_colors],
            outputs=[
                color_config_display,
                color_edit_text,
                custom_colors,
                color_edit_text,  # visibility
                apply_colors_btn,  # visibility
                color_preview_btn, # visibility
                reset_colors_btn   # visibility
            ]
        )
        
        analyze_btn.click(
            run_analysis,
            inputs=[
                semantic_classes, 
                semantic_countability, 
                openness_list,
                encoder_type,
                custom_colors,
                enable_hole_filling,
                enable_median_blur
            ],
            outputs=[analysis_status, result_gallery, stats_text]
        )
        
        # 初始检查
        api_status.value = check_api()
        
        return {
            'analyze_btn': analyze_btn,
            'status': analysis_status,
            'gallery': result_gallery,
            'custom_colors': custom_colors
        }
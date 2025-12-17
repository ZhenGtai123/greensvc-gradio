"""
视觉分析 Tab
完整版 - 图片上传 + GPS提取 + 预设配置 + 颜色配置 + 视觉分析
"""

import gradio as gr
import pandas as pd
import numpy as np
import os
import logging
import json
import requests
from typing import Dict, List, Tuple
from PIL import Image, ImageDraw
from PIL.ExifTags import TAGS, GPSTAGS

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
    "sky": "#06e6e6", "lawn": "#04fa07", "herbaceous plants": "#fa7f04",
    "trees": "#04c803", "shrubs": "#ccff04", "water": "#0907e6",
    "ground / land": "#787846", "building": "#b47878", "rock / stone": "#ff290a",
    "person / people": "#96053d", "fence / railing": "#787878", "road / highway": "#8c8c8c",
    "pavement / path / trail": "#ebff07", "bridge": "#ff5200", "vehicle / car": "#0066c8",
    "chair / bench": "#cc4603", "base / pedestal": "#ff1f00", "steps / curb": "#ffe000",
    "railing / barrier": "#ffb806", "sign / plaque": "#ff0599", "bin / trash can": "#ad00ff",
    "tower": "#ffb8b8", "awning / pavilion / shade structure": "#ffd000",
    "street light / lamp post": "#0047ff", "boat": "#ffeb00", "fountain": "#08b8aa",
    "bicycle": "#fff500", "sculpture / outdoor art": "#ffff00", "pier / dock": "#4700ff",
    "aquatic plants": "#4eff00", "green-covered building": "#00ff4e", "couplet": "#82513e",
    "riverbank": "#e2c8a0", "hill / mountain": "#8fff8c", "construction equipment": "#ff7104",
    "pole": "#b5a6ae", "animal": "#6edca7", "monument": "#484846", "door": "#36283b",
    "outdoor sports equipment": "#37393a", "waterfall": "#27c4c4",
    "grass": "#04fa07", "person": "#96053d", "road": "#8c8c8c", "vehicle": "#0066c8"
}


def hex_to_rgb(hex_color: str, bgr_mode: bool = True) -> List[int]:
    """将16进制颜色转换为RGB或BGR列表"""
    hex_color = hex_color.lstrip('#')
    r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return [b, g, r] if bgr_mode else [r, g, b]


def extract_gps_from_image(image_path: str) -> Tuple[bool, Tuple[float, float]]:
    """从图片中提取GPS信息"""
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        
        if not exif_data:
            return False, (None, None)
        
        gps_info = {}
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            if tag == 'GPSInfo':
                for gps_tag_id, gps_value in value.items():
                    gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                    gps_info[gps_tag] = gps_value
        
        if not gps_info:
            return False, (None, None)
        
        def convert_to_degrees(value):
            d, m, s = value
            return float(d) + float(m) / 60 + float(s) / 3600
        
        if 'GPSLatitude' in gps_info and 'GPSLongitude' in gps_info:
            lat = convert_to_degrees(gps_info['GPSLatitude'])
            lon = convert_to_degrees(gps_info['GPSLongitude'])
            
            if gps_info.get('GPSLatitudeRef', 'N') == 'S':
                lat = -lat
            if gps_info.get('GPSLongitudeRef', 'E') == 'W':
                lon = -lon
            
            return True, (lat, lon)
        
        return False, (None, None)
    except Exception as e:
        logger.warning(f"Failed to extract GPS from {image_path}: {e}")
        return False, (None, None)


def analyze_image(vision_client, image_path: str, use_custom_config: bool,
                  classes: List[str], countability: List[int], openness: List[int],
                  encoder: str, semantic_colors: Dict,
                  enable_hole_fill: bool, enable_blur: bool) -> Dict:
    """调用视觉分析API"""
    try:
        # 准备请求数据
        if use_custom_config and classes:
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
        else:
            # 使用后端默认配置
            request_data = {
                "image_id": f"img_{os.path.basename(image_path).split('.')[0]}",
                "encoder": encoder,
                "enable_hole_filling": enable_hole_fill,
                "enable_median_blur": enable_blur
            }
        
        logger.info(f"Calling API with config: {request_data.get('image_id')}, custom={use_custom_config}")
        
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
            data = {'request_data': json.dumps(request_data)}
            
            response = requests.post(
                f"{vision_client.base_url}/analyze",
                files=files,
                data=data,
                timeout=600
            )
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get('status') == 'success' and 'images' in result:
                processed_images = {}
                for key, hex_data in result['images'].items():
                    if isinstance(hex_data, str):
                        img_bytes = bytes.fromhex(hex_data)
                        processed_images[key] = img_bytes
                result['images'] = processed_images
            
            return result
        else:
            error_msg = f"API返回错误: {response.status_code}"
            try:
                error_detail = response.json()
                error_msg += f" - {error_detail.get('detail', response.text[:200])}"
            except:
                error_msg += f" - {response.text[:200]}"
            
            return {'status': 'error', 'error': error_msg}
            
    except Exception as e:
        logger.error(f"API call exception: {e}", exc_info=True)
        return {'status': 'error', 'error': str(e)}


def create_vision_analysis_tab(components: dict, app_state, config: dict):
    """创建视觉分析Tab"""
    
    with gr.Tab("3. 视觉分析"):
        custom_colors = gr.State({})
        
        gr.Markdown("""
        ### 🎯 视觉分析
        上传图片，使用AI模型进行语义分割、深度估计和前中背景分割。
        """)
        
        # API状态检查
        with gr.Row():
            api_status = gr.Textbox(label="API状态", value="未检查", interactive=False)
            check_api_btn = gr.Button("检查API", variant="secondary")
        
        # 图片上传
        gr.Markdown("#### 📁 上传图片")
        image_files = gr.File(
            label="选择图片文件（支持多选）",
            file_count="multiple",
            file_types=["image"]
        )
        
        # GPS信息显示
        with gr.Accordion("📍 GPS信息", open=False):
            gps_info = gr.Dataframe(
                label="图片GPS信息",
                headers=["文件名", "有GPS", "纬度", "经度"],
                interactive=False
            )
            enable_heatmap = gr.Checkbox(
                label="生成空间热力图（需要所有图片都有GPS信息）",
                value=False,
                interactive=False
            )
        
        # 配置模式选择
        gr.Markdown("#### ⚙️ 分析配置")
        use_custom_config = gr.Checkbox(
            label="使用自定义配置（不勾选则使用后端默认的41类园林配置）",
            value=False
        )
        
        # 预设配置（折叠）
        with gr.Accordion("📋 预设与自定义配置", open=False, visible=True) as config_accordion:
            with gr.Row():
                preset_dropdown = gr.Dropdown(
                    label="选择预设配置",
                    choices=list(PRESET_CONFIGS.keys()),
                    value="默认配置（41类园林）"
                )
                apply_preset_btn = gr.Button("应用预设", variant="secondary")
            
            semantic_classes = gr.Textbox(
                label="语义类别（每行一个）",
                lines=6,
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
        
        # 颜色配置（折叠）
        with gr.Accordion("🎨 颜色配置", open=False, visible=True) as color_accordion:
            gr.Markdown("点击生成颜色配置查看和修改各类别的颜色")
            generate_colors_btn = gr.Button("生成颜色配置", variant="secondary")
            color_config_display = gr.HTML("")
            color_edit_text = gr.Textbox(
                label="编辑颜色配置（每行：类别名=#颜色代码）",
                lines=8,
                visible=False
            )
            with gr.Row():
                apply_colors_btn = gr.Button("应用颜色", variant="secondary", visible=False)
                reset_colors_btn = gr.Button("重置颜色", variant="secondary", visible=False)
            color_preview_image = gr.Image(label="颜色预览", visible=False)
        
        # 基础配置
        with gr.Row():
            encoder_type = gr.Radio(
                label="模型大小",
                choices=[("标准", "vitb"), ("轻量", "vits")],
                value="vitb"
            )
        
        with gr.Row():
            enable_hole_filling = gr.Checkbox(label="启用智能空洞填充", value=True)
            enable_median_blur = gr.Checkbox(label="启用中值滤波平滑", value=True)
        
        # 分析按钮
        analyze_btn = gr.Button("🚀 开始分析", variant="primary", size="lg")
        
        # 状态和结果
        analysis_status = gr.Textbox(label="分析状态", lines=2, interactive=False)
        result_gallery = gr.Gallery(
            label="分析结果",
            columns=5, rows=4,
            object_fit="contain",
            height="auto"
        )
        stats_text = gr.Textbox(label="分析统计", lines=3, interactive=False)
        
        # ========== 事件处理函数 ==========
        
        def check_api():
            try:
                vision_client = components.get('vision_client')
                if vision_client and vision_client.check_health():
                    return f"✅ API正常 ({vision_client.base_url})"
                return "❌ API未连接"
            except:
                return "❌ 无法连接API"
        
        def extract_gps_info(files):
            if not files:
                return pd.DataFrame(), False
            
            gps_data_list = []
            all_have_gps = True
            locations = []
            
            for file in files:
                image_path = file.name
                has_gps, (lat, lon) = extract_gps_from_image(image_path)
                
                gps_data_list.append({
                    '文件名': os.path.basename(image_path),
                    '有GPS': '是' if has_gps else '否',
                    '纬度': lat,
                    '经度': lon
                })
                
                if has_gps:
                    locations.append((lat, lon))
                else:
                    all_have_gps = False
            
            app_state.set_gps_data({'all_have_gps': all_have_gps, 'locations': locations})
            return pd.DataFrame(gps_data_list), all_have_gps
        
        def apply_preset(preset_name):
            if preset_name in PRESET_CONFIGS:
                cfg = PRESET_CONFIGS[preset_name]
                return cfg["classes"], cfg["countability"], cfg["openness"]
            return "", "", ""
        
        def generate_color_config(classes_text, current_custom_colors):
            try:
                classes = [c.strip() for c in classes_text.split('\n') if c.strip()]
                if not classes:
                    return "", "", current_custom_colors, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
                
                if not current_custom_colors:
                    current_custom_colors = {}
                
                color_text_lines = []
                html = '<table style="width:100%; border-collapse:collapse;"><tr><th>序号</th><th>类别</th><th>颜色</th><th>代码</th></tr>'
                
                for i, cls in enumerate(classes):
                    if cls not in current_custom_colors:
                        current_custom_colors[cls] = DEFAULT_COLORS.get(cls, f"#{(i*37)%256:02x}{(i*73)%256:02x}{(i*113)%256:02x}")
                    
                    color = current_custom_colors[cls]
                    color_text_lines.append(f"{cls}={color}")
                    html += f'<tr><td>{i+1}</td><td>{cls}</td><td><span style="display:inline-block;width:40px;height:20px;background:{color};border:1px solid #ccc;"></span></td><td><code>{color}</code></td></tr>'
                
                html += '</table>'
                
                return (html, "\n".join(color_text_lines), current_custom_colors,
                        gr.update(visible=True), gr.update(visible=True), gr.update(visible=True))
            except Exception as e:
                return f"错误: {e}", "", current_custom_colors, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
        
        def apply_color_config(color_text, classes_text):
            try:
                new_colors = {}
                for line in color_text.strip().split('\n'):
                    if '=' in line:
                        parts = line.split('=', 1)
                        if len(parts) == 2:
                            cls, color = parts[0].strip(), parts[1].strip()
                            if color.startswith('#'):
                                new_colors[cls] = color
                return new_colors, "✅ 颜色已更新"
            except Exception as e:
                return {}, f"❌ 错误: {e}"
        
        def reset_colors(classes_text):
            classes = [c.strip() for c in classes_text.split('\n') if c.strip()]
            return {cls: DEFAULT_COLORS.get(cls, "#808080") for cls in classes}
        
        def run_analysis(files, use_custom, classes_text, countability_text, openness_text,
                        encoder, current_custom_colors, enable_hole_fill, enable_blur):
            try:
                vision_client = components.get('vision_client')
                if not vision_client:
                    return "❌ Vision API未配置", [], ""
                
                if not files:
                    return "❌ 请先上传图片", [], ""
                
                # 解析自定义配置
                classes, countability, openness, semantic_colors = [], [], [], {}
                if use_custom:
                    classes = [c.strip() for c in classes_text.split('\n') if c.strip()]
                    countability = [int(x.strip()) for x in countability_text.split(',')]
                    openness = [int(x.strip()) for x in openness_text.split(',')]
                    
                    if len(countability) != len(classes) or len(openness) != len(classes):
                        return "❌ 参数数量与类别数不匹配", [], ""
                    
                    # 准备颜色
                    for i, cls in enumerate(classes):
                        color_hex = current_custom_colors.get(cls, DEFAULT_COLORS.get(cls, "#808080"))
                        semantic_colors[str(i+1)] = hex_to_rgb(color_hex, bgr_mode=True)
                    semantic_colors["0"] = [0, 0, 0]
                
                display_images = []
                stats_info = {'total_images': len(files), 'success': 0}
                
                type_names = {
                    'semantic_map': '语义分割', 'depth_map': '深度图', 'fmb_map': '前中背景',
                    'openness_map': '开放度', 'foreground_map': '前景掩码',
                    'middleground_map': '中景掩码', 'background_map': '背景掩码', 'original': '原图',
                    'semantic_foreground': '语义-前景', 'semantic_middleground': '语义-中景',
                    'semantic_background': '语义-背景', 'depth_foreground': '深度-前景',
                    'depth_middleground': '深度-中景', 'depth_background': '深度-背景',
                    'openness_foreground': '开放度-前景', 'openness_middleground': '开放度-中景',
                    'openness_background': '开放度-背景', 'original_foreground': '原图-前景',
                    'original_middleground': '原图-中景', 'original_background': '原图-背景'
                }
                
                for file in files:
                    image_path = file.name
                    img_name = os.path.basename(image_path).split('.')[0]
                    
                    result = analyze_image(
                        vision_client, image_path, use_custom,
                        classes, countability, openness,
                        encoder, semantic_colors, enable_hole_fill, enable_blur
                    )
                    
                    if result.get('status') == 'success':
                        stats_info['success'] += 1
                        app_state.add_vision_result(image_path, result)
                        
                        if 'images' in result:
                            save_dir = os.path.join(config['temp_dir'], f'vision_{img_name}')
                            os.makedirs(save_dir, exist_ok=True)
                            
                            for img_type, img_data in result['images'].items():
                                if img_data:
                                    img_path = os.path.join(save_dir, f'{img_type}.png')
                                    with open(img_path, 'wb') as f:
                                        f.write(img_data)
                                    display_name = type_names.get(img_type, img_type)
                                    display_images.append((img_path, f"{img_name}-{display_name}"))
                    else:
                        logger.error(f"Failed: {image_path} - {result.get('error')}")
                
                stats_summary = f"处理图片: {stats_info['success']}/{stats_info['total_images']}"
                stats_summary += f"\n配置模式: {'自定义' if use_custom else '后端默认'}"
                if enable_hole_fill:
                    stats_summary += "\n✅ 已启用智能空洞填充"
                if enable_blur:
                    stats_summary += "\n✅ 已启用中值滤波平滑"
                
                return f"✅ 分析完成，成功处理 {stats_info['success']} 张图片", display_images, stats_summary
                
            except Exception as e:
                logger.error(f"Analysis error: {e}")
                return f"❌ 分析失败: {str(e)}", [], ""
        
        # ========== 绑定事件 ==========
        check_api_btn.click(check_api, outputs=api_status)
        
        image_files.change(extract_gps_info, inputs=[image_files], outputs=[gps_info, enable_heatmap])
        
        apply_preset_btn.click(
            apply_preset,
            inputs=[preset_dropdown],
            outputs=[semantic_classes, semantic_countability, openness_list]
        )
        
        generate_colors_btn.click(
            generate_color_config,
            inputs=[semantic_classes, custom_colors],
            outputs=[color_config_display, color_edit_text, custom_colors,
                    color_edit_text, apply_colors_btn, reset_colors_btn]
        )
        
        apply_colors_btn.click(
            apply_color_config,
            inputs=[color_edit_text, semantic_classes],
            outputs=[custom_colors, analysis_status]
        )
        
        reset_colors_btn.click(
            reset_colors,
            inputs=[semantic_classes],
            outputs=[custom_colors]
        )
        
        analyze_btn.click(
            run_analysis,
            inputs=[image_files, use_custom_config, semantic_classes, semantic_countability,
                   openness_list, encoder_type, custom_colors, enable_hole_filling, enable_median_blur],
            outputs=[analysis_status, result_gallery, stats_text]
        )
        
        api_status.value = check_api()
        
        return {
            'image_files': image_files,
            'gps_info': gps_info,
            'enable_heatmap': enable_heatmap,
            'analyze_btn': analyze_btn,
            'status': analysis_status,
            'gallery': result_gallery,
            'custom_colors': custom_colors
        }
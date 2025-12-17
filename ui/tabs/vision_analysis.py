"""
视觉分析 Tab
合并版 - 图片上传 + GPS提取 + 视觉分析（无图片预处理）
"""

import gradio as gr
import pandas as pd
import os
import logging
import json
import requests
from typing import Dict, List, Tuple
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

logger = logging.getLogger(__name__)


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
        
        # 解析经纬度
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


def analyze_image_simple(vision_client, image_path: str, encoder: str,
                         enable_hole_fill: bool, enable_blur: bool) -> Dict:
    """使用简化配置调用视觉分析API"""
    try:
        request_data = {
            "image_id": f"img_{os.path.basename(image_path).split('.')[0]}",
            "encoder": encoder,
            "enable_hole_filling": enable_hole_fill,
            "enable_median_blur": enable_blur
        }
        
        logger.info(f"Calling API with config: {request_data}")
        
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
    """创建视觉分析Tab（合并版）"""
    
    with gr.Tab("3. 视觉分析"):
        gr.Markdown("""
        ### 🎯 视觉分析
        上传图片，使用AI模型进行语义分割、深度估计和前中背景分割。
        """)
        
        # API状态检查
        with gr.Row():
            api_status = gr.Textbox(
                label="API状态",
                value="未检查",
                interactive=False
            )
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
        
        # 配置选项
        gr.Markdown("#### ⚙️ 分析配置")
        with gr.Row():
            encoder_type = gr.Radio(
                label="模型大小",
                choices=[("标准", "vitb"), ("轻量", "vits")],
                value="vitb"
            )
        
        with gr.Row():
            enable_hole_filling = gr.Checkbox(
                label="启用智能空洞填充",
                value=True
            )
            enable_median_blur = gr.Checkbox(
                label="启用中值滤波平滑",
                value=True
            )
        
        # 分析按钮
        analyze_btn = gr.Button("🚀 开始分析", variant="primary", size="lg")
        
        # 状态
        analysis_status = gr.Textbox(
            label="分析状态",
            lines=2,
            interactive=False
        )
        
        # 结果展示
        result_gallery = gr.Gallery(
            label="分析结果",
            columns=5,
            rows=4,
            object_fit="contain",
            height="auto",
            show_label=True
        )
        
        # 统计信息
        stats_text = gr.Textbox(
            label="分析统计",
            lines=3,
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
        
        def extract_gps_info(files):
            """提取上传图片的GPS信息"""
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
            
            # 保存GPS数据到状态
            gps_data = {
                'all_have_gps': all_have_gps,
                'locations': locations
            }
            app_state.set_gps_data(gps_data)
            
            df_gps = pd.DataFrame(gps_data_list)
            return df_gps, all_have_gps
        
        def run_analysis(files, encoder, enable_hole_fill, enable_blur):
            """运行分析"""
            try:
                vision_client = components.get('vision_client')
                if not vision_client:
                    return "❌ Vision API未配置", [], ""
                
                if not files:
                    return "❌ 请先上传图片", [], ""
                
                display_images = []
                stats_info = {'total_images': len(files), 'success': 0}
                
                # 图片类型中文名称
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
                
                for file in files:
                    image_path = file.name  # 直接使用原始图片路径，不做预处理
                    img_name = os.path.basename(image_path).split('.')[0]
                    
                    logger.info(f"Processing: {image_path}")
                    
                    result = analyze_image_simple(
                        vision_client,
                        image_path,
                        encoder,
                        enable_hole_fill,
                        enable_blur
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
                
                # 统计
                stats_summary = f"处理图片: {stats_info['success']}/{stats_info['total_images']}"
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
        
        # 上传图片时自动提取GPS信息
        image_files.change(
            extract_gps_info,
            inputs=[image_files],
            outputs=[gps_info, enable_heatmap]
        )
        
        analyze_btn.click(
            run_analysis,
            inputs=[image_files, encoder_type, enable_hole_filling, enable_median_blur],
            outputs=[analysis_status, result_gallery, stats_text]
        )
        
        # 初始检查
        api_status.value = check_api()
        
        return {
            'image_files': image_files,
            'gps_info': gps_info,
            'enable_heatmap': enable_heatmap,
            'analyze_btn': analyze_btn,
            'status': analysis_status,
            'gallery': result_gallery
        }
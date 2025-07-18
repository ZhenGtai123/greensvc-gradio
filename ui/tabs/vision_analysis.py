"""
Tab 4: 视觉分析
"""

import gradio as gr
import os
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)

# 预设配置字典
PRESET_CONFIGS = {
    "默认配置（8类）": {
        "classes": "Sky\nLawn, Grass, Grassland\nTrees, Tree\nBuilding, Buildings\nWater, River, Lake\nPeople, Person, Human\nRoads, Street\nCars, Vehicles",
        "countability": "0,0,1,1,0,1,0,1",
        "openness": "0,0,1,1,0,1,0,1"
    },
    "简单配置（3类）": {
        "classes": "Sky\nVegetation, Plants, Green\nBuilt Environment, Buildings, Structures",
        "countability": "0,0,1",
        "openness": "0,1,1"
    },
    "详细配置（15类）": {
        "classes": "Sky\nGrass, Lawn\nTrees\nShrubs\nFlowers\nWater\nSoil, Dirt\nBuilding\nRoad\nSidewalk\nPeople\nCars\nBikes\nFences\nSigns",
        "countability": "0,0,1,1,1,0,0,1,0,0,1,1,1,1,1",
        "openness": "0,1,1,1,1,0,1,1,0,0,1,1,1,1,1"
    },
    "建筑分析（10类）": {
        "classes": "Sky\nGround, Floor\nWalls\nWindows\nDoors\nRoof\nVegetation\nPeople\nVehicles\nFurniture",
        "countability": "0,0,0,1,1,0,0,1,1,1",
        "openness": "0,1,1,1,1,1,1,1,1,1"
    },
    "自然景观（12类）": {
        "classes": "Sky\nClouds\nMountains, Hills\nTrees\nGrass\nFlowers\nWater, Lakes, Rivers\nRocks\nSoil\nAnimals\nPeople\nPaths, Trails",
        "countability": "0,1,1,1,0,1,0,1,0,1,1,0",
        "openness": "0,0,1,1,1,1,0,1,1,1,1,0"
    }
}

def create_vision_analysis_tab(components: dict, app_state, config: dict):
    """创建视觉分析Tab"""
    
    with gr.Tab("4. 视觉分析"):
        gr.Markdown("### 配置语义分割参数")
        
        # 添加参数验证状态显示
        param_validation_status = gr.Textbox(
            label="参数验证状态",
            interactive=False,
            visible=True
        )
        
        with gr.Row():
            semantic_classes = gr.Textbox(
                label="语义类别（每行一个）",
                lines=10,
                value="Sky\nLawn, Grass, Grassland\nTrees, Tree\nBuilding, Buildings\nWater, River, Lake\nPeople, Person, Human\nRoads, Street\nCars, Vehicles",
                placeholder="每行输入一个类别，可以用逗号分隔同义词"
            )
            
            with gr.Column():
                semantic_countability = gr.Textbox(
                    label="可数性（用逗号分隔，1=可数，0=不可数）",
                    value="0,0,1,1,0,1,0,1",
                    placeholder="例如: 1,0,0,1,0,1,0,1"
                )
                openness_list = gr.Textbox(
                    label="开放度（用逗号分隔，1=开放，0=封闭）",
                    value="0,0,1,1,0,1,0,1",
                    placeholder="例如: 1,1,0,0,1,0,1,0"
                )
                
                # 添加快速填充按钮
                with gr.Row():
                    fill_zeros_btn = gr.Button("全部填0", scale=1)
                    fill_ones_btn = gr.Button("全部填1", scale=1)
                    auto_detect_btn = gr.Button("自动检测类别数", scale=2)
        
        # 添加高级选项
        with gr.Accordion("高级选项", open=False):
            with gr.Row():
                segmentation_mode = gr.Radio(
                    label="分割模式",
                    choices=["single_label", "instance"],
                    value="single_label",
                    info="single_label: 每个像素只属于一个类别; instance: 区分同类别的不同实例"
                )
                detection_threshold = gr.Slider(
                    label="检测阈值",
                    minimum=0.01,
                    maximum=0.5,
                    value=0.05,
                    step=0.01,
                    info="较低的值检测更多对象，较高的值只检测置信度高的对象"
                )
            with gr.Row():
                min_object_area_ratio = gr.Slider(
                    label="最小对象面积比例",
                    minimum=0.00001,
                    maximum=0.01,
                    value=0.00005,
                    step=0.00001,
                    info="过滤掉太小的对象"
                )
                enable_hole_filling = gr.Checkbox(
                    label="启用空洞填充",
                    value=False,
                    info="填充FMB分割中的空洞"
                )
        
        # 添加预设配置
        gr.Markdown("### 预设配置")
        with gr.Row():
            preset_configs = gr.Dropdown(
                label="选择预设配置",
                choices=list(PRESET_CONFIGS.keys()),
                value="默认配置（8类）"
            )
            apply_preset_btn = gr.Button("应用预设", variant="secondary")
        
        analyze_btn = gr.Button("开始分析", variant="primary")
        analysis_status = gr.Textbox(label="分析状态")
        result_images = gr.Gallery(
            label="分析结果示例", 
            columns=4,
            rows=3,
            object_fit="contain",
            height="auto"
        )
        
        # 事件处理函数
        def validate_semantic_params(classes_text, countability_text, openness_text):
            """验证语义参数的一致性"""
            try:
                # 解析类别
                classes = [c.strip() for c in classes_text.strip().split('\n') if c.strip()]
                num_classes = len(classes)
                
                # 解析可数性和开放度
                countability = [int(x.strip()) for x in countability_text.split(',') if x.strip()]
                openness = [int(x.strip()) for x in openness_text.split(',') if x.strip()]
                
                # 验证长度
                if len(countability) != num_classes:
                    return f"❌ 错误：类别数({num_classes})与可数性参数数量({len(countability)})不匹配"
                if len(openness) != num_classes:
                    return f"❌ 错误：类别数({num_classes})与开放度参数数量({len(openness)})不匹配"
                
                # 验证值范围
                if not all(x in [0, 1] for x in countability):
                    return "❌ 错误：可数性参数只能是0或1"
                if not all(x in [0, 1] for x in openness):
                    return "❌ 错误：开放度参数只能是0或1"
                
                return f"✅ 参数验证通过：{num_classes}个类别，参数长度匹配"
                
            except ValueError as e:
                return f"❌ 错误：参数格式不正确 - {str(e)}"
            except Exception as e:
                return f"❌ 错误：{str(e)}"
        
        def auto_detect_classes(classes_text):
            """自动检测类别数并生成对应的参数"""
            classes = [c.strip() for c in classes_text.strip().split('\n') if c.strip()]
            num_classes = len(classes)
            
            # 生成默认参数（全0）
            countability = ','.join(['0'] * num_classes)
            openness = ','.join(['0'] * num_classes)
            
            return countability, openness, f"已检测到{num_classes}个类别，已生成对应参数"
        
        def fill_all_zeros(classes_text):
            """填充全0"""
            classes = [c.strip() for c in classes_text.strip().split('\n') if c.strip()]
            num_classes = len(classes)
            params = ','.join(['0'] * num_classes)
            return params, params
        
        def fill_all_ones(classes_text):
            """填充全1"""
            classes = [c.strip() for c in classes_text.strip().split('\n') if c.strip()]
            num_classes = len(classes)
            params = ','.join(['1'] * num_classes)
            return params, params
        
        def apply_preset_config(preset_name):
            """应用预设配置"""
            if preset_name in PRESET_CONFIGS:
                config = PRESET_CONFIGS[preset_name]
                return config["classes"], config["countability"], config["openness"]
            return "", "", ""
        
        def run_vision_analysis_with_validation(semantic_classes, semantic_countability, openness_list,
                                              segmentation_mode, detection_threshold, 
                                              min_object_area_ratio, enable_hole_filling):
            """带参数验证的视觉分析"""
            # 首先验证参数
            validation_result = validate_semantic_params(semantic_classes, semantic_countability, openness_list)
            if not validation_result.startswith("✅"):
                return validation_result, []
            
            try:
                if not app_state.has_processed_images():
                    return "请先上传图片", []
                
                # 准备语义类别和参数
                classes = [c.strip() for c in semantic_classes.split('\n') if c.strip()]
                countability = [int(x) for x in semantic_countability.split(',')]
                openness = [int(x) for x in openness_list.split(',')]
                
                # 对每张图片进行分析
                results = []
                sample_images = []
                
                for path, info in app_state.get_processed_images().items():
                    if info['status'] == 'success':
                        # 使用高级API调用
                        result = components['vision_client'].analyze_image_advanced(
                            info['processed_path'],
                            classes,
                            countability,
                            openness,
                            segmentation_mode=segmentation_mode,
                            detection_threshold=detection_threshold,
                            min_object_area_ratio=min_object_area_ratio,
                            enable_hole_filling=enable_hole_filling
                        )
                        results.append(result)
                        app_state.add_vision_result(path, result)
                        
                        # 处理结果图片
                        if result['status'] == 'success' and 'images' in result:
                            img_name = os.path.splitext(os.path.basename(path))[0]
                            result_dir = os.path.join(config['temp_dir'], f'vision_results_{img_name}')
                            os.makedirs(result_dir, exist_ok=True)
                            
                            available_images = list(result['images'].keys())
                            
                            for img_type in available_images[:12]:  # 最多显示12张
                                if img_type in result['images']:
                                    img_data = result['images'][img_type]
                                    if isinstance(img_data, bytes):
                                        img_path = os.path.join(result_dir, f'{img_type}.png')
                                        with open(img_path, 'wb') as f:
                                            f.write(img_data)
                                        sample_images.append((img_path, f'{img_name} - {img_type}'))
                
                if not results:
                    return "没有成功分析的图片", []
                
                success_count = sum(1 for r in results if r['status'] == 'success')
                status_msg = f"已完成 {len(results)} 张图片的视觉分析\n"
                status_msg += f"成功: {success_count}, 失败: {len(results) - success_count}\n"
                status_msg += f"分割模式: {segmentation_mode}"
                
                return status_msg, sample_images
                
            except Exception as e:
                return f"分析失败: {str(e)}", []
        
        # 绑定事件
        semantic_classes.change(
            fn=validate_semantic_params,
            inputs=[semantic_classes, semantic_countability, openness_list],
            outputs=[param_validation_status]
        )
        semantic_countability.change(
            fn=validate_semantic_params,
            inputs=[semantic_classes, semantic_countability, openness_list],
            outputs=[param_validation_status]
        )
        openness_list.change(
            fn=validate_semantic_params,
            inputs=[semantic_classes, semantic_countability, openness_list],
            outputs=[param_validation_status]
        )
        
        fill_zeros_btn.click(
            fn=fill_all_zeros,
            inputs=[semantic_classes],
            outputs=[semantic_countability, openness_list]
        )
        fill_ones_btn.click(
            fn=fill_all_ones,
            inputs=[semantic_classes],
            outputs=[semantic_countability, openness_list]
        )
        auto_detect_btn.click(
            fn=auto_detect_classes,
            inputs=[semantic_classes],
            outputs=[semantic_countability, openness_list, param_validation_status]
        )
        
        apply_preset_btn.click(
            fn=apply_preset_config,
            inputs=[preset_configs],
            outputs=[semantic_classes, semantic_countability, openness_list]
        )
        
        analyze_btn.click(
            fn=run_vision_analysis_with_validation,
            inputs=[
                semantic_classes, 
                semantic_countability, 
                openness_list,
                segmentation_mode,
                detection_threshold,
                min_object_area_ratio,
                enable_hole_filling
            ],
            outputs=[analysis_status, result_images]
        )
        
        return {
            'semantic_classes': semantic_classes,
            'semantic_countability': semantic_countability,
            'openness_list': openness_list,
            'analyze_btn': analyze_btn,
            'analysis_status': analysis_status,
            'result_images': result_images
        }
"""
Tab 4: 视觉分析 - 增强版
支持新API的所有功能
"""

import gradio as gr
import os
from typing import Tuple, List, Dict
import logging
import json

logger = logging.getLogger(__name__)

# 预设配置字典 - 扩展版
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
    },
    "城市景观（20类）": {
        "classes": "Sky\nClouds\nBuilding\nSkyscraper\nBridge\nRoad\nSidewalk\nTraffic Light\nStreet Sign\nCar\nBus\nTruck\nBicycle\nMotorcycle\nPerson\nTree\nGrass\nFlower\nWater\nFence",
        "countability": "0,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,0,1,0,1",
        "openness": "0,0,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,0,1"
    },
    "园林景观（42类）": {
        "classes": "Sky\nLawn\nHerbaceous\nTrees\nShrubs\nWater\nLand\nBuilding\nRock; stone\nPeople\nWall\nRoads\nPavements\nBridge\nAutomobiles\nChairs\nBases, plinths, pedestals, bases for sculptures and planters\nSteps\nFences\nSigns, plaques\nBins\nTowers\nAwnings\nStreet Lights\nBoat\nFountains\nBicycles\nSculptures\nPiers\nAquatic plants\nGreen-covered buildings\nCouplets\nRiverbanks\nHills\nConstruction equipment\nPoles\nAnimal\nMonuments\nDoors\nOutdoor sports equipment\nWaterfalls\nPavilion",
        "countability": "0,0,1,1,1,0,0,1,1,1,0,0,0,0,1,1,1,0,0,1,1,1,1,1,1,1,1,1,0,1,1,1,0,0,1,1,1,1,1,1,0,1",
        "openness": "0,0,1,1,1,0,0,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,0,0,0,1,0,0,1,1,1,1,1,1,0,1"
    }
}

# 图像类型描述
IMAGE_TYPE_DESCRIPTIONS = {
    'semantic_map': '语义分割图',
    'depth_map': '深度图',
    'fmb_map': '前中背景图',
    'openness_map': '开放度图',
    'foreground_map': '前景掩码',
    'middleground_map': '中景掩码',
    'background_map': '背景掩码',
    'original': '原图（调整尺寸）',
    'instance_map': '实例分割图',
    'colored_instance_map': '彩色实例图',
    'semantic_foreground': '语义前景',
    'semantic_middleground': '语义中景',
    'semantic_background': '语义背景',
    'depth_foreground': '深度前景',
    'depth_middleground': '深度中景',
    'depth_background': '深度背景',
    'openness_foreground': '开放度前景',
    'openness_middleground': '开放度中景',
    'openness_background': '开放度背景',
    'original_foreground': '原图前景',
    'original_middleground': '原图中景',
    'original_background': '原图背景'
}

def create_vision_analysis_tab(components: dict, app_state, config: dict):
    """创建增强版视觉分析Tab"""
    
    with gr.Tab("4. 视觉分析"):
        gr.Markdown("""
        ### 🎯 视觉分析 - 增强版
        
        支持语义分割、深度估计、前中背景分割、实例分割等多种分析模式。
        新功能：实例分割、智能空洞填充、更多输出图像（20张）。
        """)
        
        # 参数验证状态
        with gr.Row():
            param_validation_status = gr.Textbox(
                label="参数验证状态",
                interactive=False,
                visible=True,
                elem_classes=["status-box"]
            )
            api_health_status = gr.Textbox(
                label="API状态",
                interactive=False,
                value="未检查",
                visible=True,
                elem_classes=["status-box"]
            )
        
        # 主要配置区
        with gr.Row():
            with gr.Column(scale=2):
                semantic_classes = gr.Textbox(
                    label="语义类别（每行一个，可用逗号分隔同义词）",
                    lines=12,
                    value=PRESET_CONFIGS["默认配置（8类）"]["classes"],
                    placeholder="每行输入一个类别，可以用逗号分隔同义词\n例如: Trees, Tree",
                    elem_classes=["code-text"]
                )
            
            with gr.Column(scale=1):
                semantic_countability = gr.Textbox(
                    label="可数性（1=可数，0=不可数）",
                    value=PRESET_CONFIGS["默认配置（8类）"]["countability"],
                    placeholder="例如: 1,0,0,1,0,1,0,1",
                    lines=3
                )
                openness_list = gr.Textbox(
                    label="开放度（1=开放，0=封闭）",
                    value=PRESET_CONFIGS["默认配置（8类）"]["openness"],
                    placeholder="例如: 1,1,0,0,1,0,1,0",
                    lines=3
                )
                
                # 快速填充按钮
                with gr.Row():
                    fill_zeros_btn = gr.Button("全部填0", size="sm", variant="secondary")
                    fill_ones_btn = gr.Button("全部填1", size="sm", variant="secondary")
                    auto_detect_btn = gr.Button("自动检测", size="sm", variant="primary")
        
        # 预设配置
        with gr.Accordion("📋 预设配置", open=True):
            with gr.Row():
                preset_configs = gr.Dropdown(
                    label="选择预设配置",
                    choices=list(PRESET_CONFIGS.keys()),
                    value="默认配置（8类）",
                    scale=3
                )
                apply_preset_btn = gr.Button("应用预设", variant="secondary", scale=1)
                save_preset_btn = gr.Button("保存当前配置", variant="secondary", scale=1)
        
        # 高级选项 - 默认展开
        with gr.Accordion("⚙️ 高级选项", open=True):
            with gr.Row():
                with gr.Column():
                    segmentation_mode = gr.Radio(
                        label="分割模式",
                        choices=[
                            ("单标签分割", "single_label"),
                            ("实例分割", "instance")
                        ],
                        value="single_label",
                        info="实例分割会区分同类别的不同对象"
                    )
                    encoder_type = gr.Dropdown(
                        label="深度模型编码器",
                        choices=["vitb", "vitl", "vits"],
                        value="vitb",
                        info="选择深度估计模型的编码器类型"
                    )
                
                with gr.Column():
                    detection_threshold = gr.Slider(
                        label="检测阈值",
                        minimum=0.1,  # 从 0.01 改为 0.1
                        maximum=0.9,  # 从 0.5 改为 0.9，给更大的范围
                        value=0.3,    # 保持默认值 0.3
                        step=0.01,
                        info="较低检测更多对象，较高只检测高置信度对象"
                    )
                    min_object_area_ratio = gr.Slider(
                        label="最小对象面积比例",
                        minimum=0.00001,
                        maximum=0.01,
                        value=0.0001,
                        step=0.00001,
                        info="过滤掉过小的检测对象"
                    )
                
                with gr.Column():
                    enable_hole_filling = gr.Checkbox(
                        label="启用智能空洞填充",
                        value=False,
                        info="使用智能算法填充FMB分割中的空洞"
                    )
                    enable_zip_download = gr.Checkbox(
                        label="生成ZIP下载包",
                        value=True,
                        info="将所有结果打包为ZIP文件"
                    )
        
        # 分析按钮和进度
        with gr.Row():
            analyze_btn = gr.Button("🚀 开始分析", variant="primary", scale=2)
            check_api_btn = gr.Button("🔍 检查API状态", variant="secondary", scale=1)
            download_config_btn = gr.Button("💾 下载配置", variant="secondary", scale=1)
        
        # 进度条
        analysis_progress = gr.Progress()
        
        # 状态和统计
        with gr.Row():
            analysis_status = gr.Textbox(
                label="分析状态",
                lines=3,
                interactive=False
            )
            analysis_stats = gr.JSON(
                label="分析统计",
                visible=False
            )
        
        # 结果展示 - 改进的画廊
        with gr.Tabs():
            with gr.Tab("🖼️ 分析结果"):
                result_images = gr.Gallery(
                    label="分析结果图像",
                    columns=4,
                    rows=5,
                    object_fit="contain",
                    height="auto",
                    show_label=True,
                    elem_classes=["result-gallery"]
                )
            
            with gr.Tab("📊 实例信息"):
                instance_info = gr.DataFrame(
                    label="检测到的实例",
                    headers=["实例ID", "类别", "置信度", "面积", "边界框"],
                    visible=False
                )
            
            with gr.Tab("📈 类别统计"):
                class_statistics = gr.DataFrame(
                    label="类别统计信息",
                    headers=["类别", "像素数", "占比(%)"],
                    visible=False
                )
        
        # 下载区域
        with gr.Row():
            download_link = gr.File(
                label="下载结果文件",
                visible=False
            )
            download_status = gr.Textbox(
                label="下载状态",
                visible=False,
                interactive=False
            )
        
        # 事件处理函数
        def validate_semantic_params(classes_text, countability_text, openness_text):
            """增强的参数验证"""
            try:
                # 解析类别
                classes = [c.strip() for c in classes_text.strip().split('\n') if c.strip()]
                num_classes = len(classes)
                
                if num_classes == 0:
                    return "❌ 错误：请至少输入一个类别"
                
                if num_classes > 99:
                    return f"❌ 错误：类别数量({num_classes})超过最大限制(99)"
                
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
                
                # 统计信息
                countable_num = sum(countability)
                open_num = sum(openness)
                
                return f"✅ 参数验证通过：{num_classes}个类别 | 可数类别：{countable_num} | 开放类别：{open_num}"
                
            except ValueError as e:
                return f"❌ 错误：参数格式不正确 - {str(e)}"
            except Exception as e:
                return f"❌ 错误：{str(e)}"
        
        def check_api_status():
            """检查API健康状态"""
            try:
                if components['vision_client'].check_health():
                    # 获取配置信息
                    config_info = components['vision_client'].get_config()
                    if config_info:
                        return f"✅ API运行正常 | 支持{config_info.get('total_classes', 0)}个类别 | 输出{config_info.get('output_images', 0)}张图像"
                    return "✅ API运行正常"
                else:
                    return "❌ API无响应，请检查服务是否启动"
            except:
                return "❌ 无法连接到API服务"
        
        def auto_detect_classes(classes_text):
            """自动检测并生成智能默认值"""
            classes = [c.strip() for c in classes_text.strip().split('\n') if c.strip()]
            num_classes = len(classes)
            
            # 智能生成默认值
            countability = []
            openness = []
            
            for cls in classes:
                cls_lower = cls.lower()
                
                # 可数性判断
                if any(word in cls_lower for word in ['sky', 'water', 'grass', 'road', 'ground', 'land', 'lawn', 'pavement']):
                    countability.append('0')
                else:
                    countability.append('1')
                
                # 开放度判断
                if any(word in cls_lower for word in ['sky', 'ground', 'road', 'water', 'wall', 'fence']):
                    openness.append('0')
                else:
                    openness.append('1')
            
            countability_str = ','.join(countability)
            openness_str = ','.join(openness)
            
            return countability_str, openness_str, f"已检测到{num_classes}个类别，已生成智能默认参数"
        
        def apply_preset_config(preset_name):
            """应用预设配置"""
            if preset_name in PRESET_CONFIGS:
                config = PRESET_CONFIGS[preset_name]
                return config["classes"], config["countability"], config["openness"]
            return "", "", ""
        
        def save_current_config(classes, countability, openness):
            """保存当前配置到文件"""
            try:
                config_data = {
                    "classes": classes.split('\n'),
                    "countability": countability,
                    "openness": openness
                }
                
                config_path = os.path.join(config['temp_dir'], 'saved_config.json')
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, ensure_ascii=False, indent=2)
                
                return gr.File.update(value=config_path, visible=True), "配置已保存"
            except Exception as e:
                return gr.File.update(visible=False), f"保存失败: {str(e)}"
        
        def run_vision_analysis_enhanced(semantic_classes, semantic_countability, openness_list,
                                        segmentation_mode, encoder_type, detection_threshold,
                                        min_object_area_ratio, enable_hole_filling, enable_zip,
                                        progress=gr.Progress()):
            """增强版视觉分析"""
            # 验证参数
            validation_result = validate_semantic_params(
                semantic_classes, semantic_countability, openness_list
            )
            if not validation_result.startswith("✅"):
                return validation_result, [], gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            
            try:
                if not app_state.has_processed_images():
                    return "请先上传图片", [], gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
                
                # 准备参数
                classes = [c.strip() for c in semantic_classes.split('\n') if c.strip()]
                countability = [int(x) for x in semantic_countability.split(',')]
                openness = [int(x) for x in openness_list.split(',')]
                
                # 结果收集
                all_results = []
                all_sample_images = []
                all_instances = []
                class_stats_data = {}
                
                processed_images = app_state.get_processed_images()
                total_images = len([p for p, info in processed_images.items() if info['status'] == 'success'])
                
                progress(0, desc="开始视觉分析...")
                
                for idx, (path, info) in enumerate(processed_images.items()):
                    if info['status'] == 'success':
                        progress((idx + 1) / total_images, desc=f"分析图片 {idx + 1}/{total_images}")
                        
                        # 调用增强API
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
                        
                        all_results.append(result)
                        app_state.add_vision_result(path, result)
                        
                        # 处理结果
                        if result['status'] == 'success' and 'images' in result:
                            img_name = os.path.splitext(os.path.basename(path))[0]
                            result_dir = os.path.join(config['temp_dir'], f'vision_results_{img_name}')
                            os.makedirs(result_dir, exist_ok=True)
                            
                            # 保存图片并准备展示
                            saved_images = []
                            for img_type in result['images']:
                                img_data = result['images'][img_type]
                                if isinstance(img_data, bytes):
                                    img_path = os.path.join(result_dir, f'{img_type}.png')
                                    with open(img_path, 'wb') as f:
                                        f.write(img_data)
                                    
                                    # 添加到展示列表
                                    description = IMAGE_TYPE_DESCRIPTIONS.get(img_type, img_type)
                                    saved_images.append((img_path, f'{img_name} - {description}'))
                            
                            # 优先展示的图片类型
                            priority_types = [
                                'colored_instance_map' if segmentation_mode == 'instance' else 'semantic_map',
                                'depth_map', 'fmb_map', 'openness_map'
                            ]
                            
                            # 按优先级排序
                            for img_type in priority_types:
                                for img_path, desc in saved_images:
                                    if img_type in os.path.basename(img_path):
                                        all_sample_images.append((img_path, desc))
                                        break
                            
                            # 收集实例信息
                            if 'instances' in result and result['instances']:
                                for inst in result['instances']:
                                    all_instances.append({
                                        '实例ID': inst['instance_id'],
                                        '类别': classes[inst['class_id'] - 1] if inst['class_id'] <= len(classes) else 'Unknown',
                                        '置信度': f"{inst['score']:.3f}",
                                        '面积': inst['area'],
                                        '边界框': f"({inst['bbox']['x_min']},{inst['bbox']['y_min']}) - ({inst['bbox']['x_max']},{inst['bbox']['y_max']})"
                                    })
                            
                            # 收集类别统计
                            if 'statistics' in result and 'class_statistics' in result['statistics']:
                                for class_name, stats in result['statistics']['class_statistics'].items():
                                    if class_name not in class_stats_data:
                                        class_stats_data[class_name] = {
                                            'pixels': 0,
                                            'count': 0
                                        }
                                    class_stats_data[class_name]['pixels'] += stats['pixels']
                                    class_stats_data[class_name]['count'] += 1
                
                progress(1.0, desc="分析完成！")
                
                # 准备统计数据
                if not all_results:
                    return "没有成功分析的图片", [], gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
                
                success_count = sum(1 for r in all_results if r['status'] == 'success')
                
                # 生成状态消息
                status_msg = f"✅ 分析完成！\n"
                status_msg += f"📊 处理图片: {success_count}/{len(all_results)}\n"
                status_msg += f"🎯 分割模式: {'实例分割' if segmentation_mode == 'instance' else '单标签分割'}\n"
                status_msg += f"🔧 高级选项: "
                if enable_hole_filling:
                    status_msg += "空洞填充 "
                status_msg += f"阈值={detection_threshold}"
                
                # 准备统计信息
                analysis_statistics = {
                    "total_images": len(all_results),
                    "success_count": success_count,
                    "segmentation_mode": segmentation_mode,
                    "total_classes": len(classes),
                    "detected_classes": len(class_stats_data),
                    "advanced_options": {
                        "hole_filling": enable_hole_filling,
                        "detection_threshold": detection_threshold,
                        "min_area_ratio": min_object_area_ratio
                    }
                }
                
                # 准备类别统计表格
                class_stats_rows = []
                total_pixels = sum(data['pixels'] for data in class_stats_data.values())
                for class_name, data in sorted(class_stats_data.items(), key=lambda x: x[1]['pixels'], reverse=True):
                    percentage = (data['pixels'] / total_pixels * 100) if total_pixels > 0 else 0
                    class_stats_rows.append({
                        '类别': class_name,
                        '像素数': data['pixels'],
                        '占比(%)': f"{percentage:.2f}"
                    })
                
                # 准备下载文件
                download_file = None
                download_msg = ""
                if enable_zip and success_count > 0:
                    # 这里可以调用download_zip API或本地打包
                    download_msg = "结果文件已准备好下载"
                
                return (
                    status_msg,
                    all_sample_images[:20],  # 限制显示数量
                    gr.update(value=analysis_statistics, visible=True),
                    gr.update(value=all_instances[:50] if all_instances else None, visible=bool(all_instances)),
                    gr.update(value=class_stats_rows if class_stats_rows else None, visible=bool(class_stats_rows)),
                    gr.update(value=download_file, visible=bool(download_file))
                )
                
            except Exception as e:
                logger.error(f"Vision analysis error: {str(e)}", exc_info=True)
                return f"❌ 分析失败: {str(e)}", [], gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
        
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
            fn=lambda x: (','.join(['0'] * len([c.strip() for c in x.split('\n') if c.strip()])),) * 2,
            inputs=[semantic_classes],
            outputs=[semantic_countability, openness_list]
        )
        
        fill_ones_btn.click(
            fn=lambda x: (','.join(['1'] * len([c.strip() for c in x.split('\n') if c.strip()])),) * 2,
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
        
        save_preset_btn.click(
            fn=save_current_config,
            inputs=[semantic_classes, semantic_countability, openness_list],
            outputs=[download_link, download_status]
        )
        
        check_api_btn.click(
            fn=check_api_status,
            outputs=[api_health_status]
        )
        
        analyze_btn.click(
            fn=run_vision_analysis_enhanced,
            inputs=[
                semantic_classes, 
                semantic_countability, 
                openness_list,
                segmentation_mode,
                encoder_type,
                detection_threshold,
                min_object_area_ratio,
                enable_hole_filling,
                enable_zip_download
            ],
            outputs=[
                analysis_status, 
                result_images, 
                analysis_stats,
                instance_info,
                class_statistics,
                download_link
            ]
        )
        
        # 初始检查API状态
        api_health_status.value = check_api_status()
        
        return {
            'semantic_classes': semantic_classes,
            'semantic_countability': semantic_countability,
            'openness_list': openness_list,
            'segmentation_mode': segmentation_mode,
            'detection_threshold': detection_threshold,
            'min_object_area_ratio': min_object_area_ratio,
            'enable_hole_filling': enable_hole_filling,
            'analyze_btn': analyze_btn,
            'analysis_status': analysis_status,
            'result_images': result_images,
            'instance_info': instance_info,
            'class_statistics': class_statistics
        }
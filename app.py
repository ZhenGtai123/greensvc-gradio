"""
城市绿地空间视觉分析系统
主应用程序
"""

import gradio as gr
import pandas as pd
import json
import os
import tempfile
from typing import List, Dict, Tuple, Optional
import numpy as np

# 导入各个模块
from modules.api_clients import VisionModelClient, MetricsRecommenderClient
from modules.image_processor import ImageProcessor
from modules.metrics_manager import MetricsManager
from modules.metrics_calculator import MetricsCalculator
from modules.report_generator import ReportGenerator

# 配置
CONFIG = {
    'vision_api_url': 'http://127.0.0.1:8000',  # 本地视觉模型API
    'metrics_api_url': 'http://localhost:8001',  # 本地运行的metrics推荐API
    'metrics_library_path': 'data/library_metrics.xlsx',
    'metrics_code_dir': 'data/metrics_code/',
    'output_dir': 'outputs/',
    'temp_dir': 'temp/'
}

# 确保必要的目录存在
for dir_path in ['data', 'data/metrics_code', 'outputs', 'temp']:
    os.makedirs(dir_path, exist_ok=True)

# 全局变量存储应用状态
app_state = {
    'processed_images': {},
    'selected_metrics': [],
    'vision_results': {},
    'metrics_results': {},
    'gps_data': {}
}

def initialize_app():
    """初始化应用程序"""
    # 初始化各个组件
    vision_client = VisionModelClient(CONFIG['vision_api_url'])
    metrics_client = MetricsRecommenderClient(CONFIG['metrics_api_url'])
    image_processor = ImageProcessor()
    metrics_manager = MetricsManager(CONFIG['metrics_library_path'], CONFIG['metrics_code_dir'])
    metrics_calculator = MetricsCalculator(CONFIG['metrics_code_dir'])
    report_generator = ReportGenerator(CONFIG['output_dir'])
    
    return {
        'vision_client': vision_client,
        'metrics_client': metrics_client,
        'image_processor': image_processor,
        'metrics_manager': metrics_manager,
        'metrics_calculator': metrics_calculator,
        'report_generator': report_generator
    }

# 初始化组件
components = initialize_app()

# === 界面功能函数 ===

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

def load_metrics_library() -> pd.DataFrame:
    """加载指标库"""
    try:
        return components['metrics_manager'].load_metrics()
    except Exception as e:
        return pd.DataFrame()

def update_metrics_library(file) -> str:
    """更新指标库文件"""
    try:
        if file is None:
            return "请选择文件"
        
        # 保存上传的文件
        df = pd.read_excel(file.name)
        df.to_excel(CONFIG['metrics_library_path'], index=False)
        
        # 重新加载
        components['metrics_manager'].reload_metrics()
        
        return "指标库已更新"
    except Exception as e:
        return f"更新失败: {str(e)}"

def upload_metric_code(metric_name: str, code_file) -> str:
    """上传指标计算代码"""
    try:
        if not metric_name or code_file is None:
            return "请输入指标名称并选择代码文件"
        
        # Gradio File组件返回的是文件路径，不是文件对象
        file_path = code_file.name if hasattr(code_file, 'name') else code_file
        
        # 读取代码文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            code_content = f.read()
            
        components['metrics_manager'].save_metric_code(metric_name, code_content)
        
        return f"指标 '{metric_name}' 的代码已上传"
    except Exception as e:
        return f"上传失败: {str(e)}"

def process_images(files) -> Tuple[str, pd.DataFrame, bool]:
    """处理上传的图片"""
    global app_state
    
    try:
        if not files:
            return "请上传图片", pd.DataFrame(), False
        
        # 处理图片
        results = components['image_processor'].batch_process_images([f.name for f in files])
        app_state['processed_images'] = results
        
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
        app_state['gps_data'] = {
            'all_have_gps': all_have_gps,
            'locations': [(info['gps'][0], info['gps'][1]) 
                         for info in results.values() if info['has_gps']]
        }
        
        status = f"已处理 {len(files)} 张图片"
        if all_have_gps:
            status += "\n所有图片都包含GPS信息，可以生成空间热力图"
        
        return status, df_gps, all_have_gps
        
    except Exception as e:
        return f"处理失败: {str(e)}", pd.DataFrame(), False

def run_vision_analysis(semantic_classes: str, semantic_countability: str, 
                       openness_list: str) -> Tuple[str, List]:
    """运行视觉分析"""
    global app_state
    
    try:
        if not app_state['processed_images']:
            return "请先上传图片", []
        
        # 准备语义类别和参数
        classes = [c.strip() for c in semantic_classes.split('\n') if c.strip()]
        countability = [int(x) for x in semantic_countability.split(',')]
        openness = [int(x) for x in openness_list.split(',')]
        
        # 对每张图片进行分析
        results = []
        sample_images = []
        
        for path, info in app_state['processed_images'].items():
            if info['status'] == 'success':
                result = components['vision_client'].analyze_image(
                    info['processed_path'],
                    classes,
                    countability,
                    openness
                )
                results.append(result)
                app_state['vision_results'][path] = result
                
                # 从结果中提取并保存示例图片
                if result['status'] == 'success' and 'images' in result:
                    # 创建该图片的结果目录
                    img_name = os.path.splitext(os.path.basename(path))[0]
                    result_dir = os.path.join(CONFIG['temp_dir'], f'vision_results_{img_name}')
                    os.makedirs(result_dir, exist_ok=True)
                    
                    # 保存各种分析图片
                    image_types = ['semantic_map', 'depth_map', 'fmb_map', 'openness_map']
                    
                    for img_type in image_types:
                        if img_type in result['images']:
                            img_data = result['images'][img_type]
                            if isinstance(img_data, bytes):
                                # 保存图片
                                img_path = os.path.join(result_dir, f'{img_type}.png')
                                with open(img_path, 'wb') as f:
                                    f.write(img_data)
                                
                                # 添加到示例图片列表（只显示前几张图片的结果）
                                if len(sample_images) < 12:  # 最多显示12张
                                    sample_images.append((img_path, f'{img_name} - {img_type}'))
        
        if not results:
            return "没有成功分析的图片", []
        
        success_count = sum(1 for r in results if r['status'] == 'success')
        status_msg = f"已完成 {len(results)} 张图片的视觉分析\n"
        status_msg += f"成功: {success_count}, 失败: {len(results) - success_count}"
        
        return status_msg, sample_images
        
    except Exception as e:
        return f"分析失败: {str(e)}", []

def calculate_metrics() -> Tuple[str, pd.DataFrame]:
    """计算选定的指标"""
    global app_state
    
    try:
        if not app_state['vision_results']:
            return "请先进行视觉分析", pd.DataFrame()
        
        if not app_state['selected_metrics']:
            return "请先选择指标", pd.DataFrame()
        
        # 准备指标信息（包含required_images）
        metrics_info = {}
        for metric in app_state['selected_metrics']:
            metric_name = metric['metric name']
            metrics_info[metric_name] = {
                'required_images': metric.get('required_images', ''),
                'category': metric.get('Primary Category', ''),
                'unit': metric.get('Unit', '')
            }
        
        # 计算每个指标
        results = []
        errors = []
        
        for img_path, vision_result in app_state['vision_results'].items():
            img_metrics = {'图片': os.path.basename(img_path)}
            
            for metric in app_state['selected_metrics']:
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
        app_state['metrics_results'] = df_results
        
        status = "指标计算完成"
        if errors:
            status += f"\n\n注意：\n" + "\n".join(errors[:5])  # 显示前5个错误
            if len(errors) > 5:
                status += f"\n... 还有 {len(errors)-5} 个错误"
        
        return status, df_results
        
    except Exception as e:
        return f"计算失败: {str(e)}", pd.DataFrame()

def validate_metrics_compatibility(selected_metrics, vision_results):
    """验证选定的指标是否与视觉分析结果兼容"""
    
    compatibility_report = []
    
    # 获取第一个视觉结果作为样本
    sample_result = next(iter(vision_results.values())) if vision_results else None
    if not sample_result or 'images' not in sample_result:
        return "无法验证：没有视觉分析结果"
    
    available_images = set(sample_result['images'].keys())
    
    for metric in selected_metrics:
        metric_name = metric['metric name']
        required_str = metric.get('required_images', '')
        
        if required_str:
            # 解析required_images
            required_images = [img.strip() for img in required_str.split(',')]
            missing = []
            
            for req_img in required_images:
                # 处理可选图像（用|分隔）
                if '|' in req_img:
                    alternatives = [alt.strip() for alt in req_img.split('|')]
                    if not any(alt in available_images for alt in alternatives):
                        missing.append(req_img)
                elif req_img not in available_images:
                    missing.append(req_img)
            
            if missing:
                compatibility_report.append(
                    f"⚠️ {metric_name}: 缺少 {', '.join(missing)}"
                )
            else:
                compatibility_report.append(
                    f"✓ {metric_name}: 所有图像可用"
                )
        else:
            compatibility_report.append(
                f"❓ {metric_name}: 未配置所需图像"
            )
    
    return "\n".join(compatibility_report)



def generate_report(include_heatmap: bool, openai_key: str) -> Tuple[str, str]:
    """生成分析报告"""
    global app_state
    
    try:
        if not app_state['metrics_results'].empty:
            report_path = components['report_generator'].generate_report(
                app_state['metrics_results'],
                app_state['selected_metrics'],
                app_state['vision_results'],
                app_state['gps_data'] if include_heatmap else None,
                openai_key
            )
            
            return "报告生成成功", report_path
        else:
            return "没有可用的分析结果", ""
            
    except Exception as e:
        return f"生成失败: {str(e)}", ""

# === 创建Gradio界面 ===

def create_interface():
    """创建Gradio界面"""
    
    with gr.Blocks(title="城市绿地空间视觉分析系统") as app:
        gr.Markdown("# 城市绿地空间视觉分析系统")
        gr.Markdown("通过AI与空间视觉指标相结合，为城市绿地等空间的专业分析、科学决策与设计优化提供数据驱动工具")
        
        with gr.Tabs():
            # Tab 1: 指标推荐与选择
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
            
            # Tab 2: 指标库管理
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
            
            # Tab 3: 图片上传与处理
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
            
            # Tab 4: 视觉分析
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
                        choices=[
                            "默认配置（8类）",
                            "简单配置（3类）",
                            "详细配置（15类）",
                            "建筑分析（10类）",
                            "自然景观（12类）"
                        ],
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
                
            # === 添加事件处理函数 ===

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

            def apply_preset_config(preset_name):
                """应用预设配置"""
                if preset_name in PRESET_CONFIGS:
                    config = PRESET_CONFIGS[preset_name]
                    return config["classes"], config["countability"], config["openness"]
                return "", "", ""

            # 修改原来的 run_vision_analysis 函数，添加参数验证
            def run_vision_analysis_with_validation(semantic_classes, semantic_countability, openness_list,
                                                segmentation_mode, detection_threshold, 
                                                min_object_area_ratio, enable_hole_filling):
                """带参数验证的视觉分析"""
                global app_state
                
                # 首先验证参数
                validation_result = validate_semantic_params(semantic_classes, semantic_countability, openness_list)
                if not validation_result.startswith("✅"):
                    return validation_result, []
                
                try:
                    if not app_state['processed_images']:
                        return "请先上传图片", []
                    
                    # 准备语义类别和参数
                    classes = [c.strip() for c in semantic_classes.split('\n') if c.strip()]
                    countability = [int(x) for x in semantic_countability.split(',')]
                    openness = [int(x) for x in openness_list.split(',')]
                    
                    # 对每张图片进行分析
                    results = []
                    sample_images = []
                    
                    for path, info in app_state['processed_images'].items():
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
                            app_state['vision_results'][path] = result
                            
                            # 处理结果图片（保持原有逻辑）
                            if result['status'] == 'success' and 'images' in result:
                                img_name = os.path.splitext(os.path.basename(path))[0]
                                result_dir = os.path.join(CONFIG['temp_dir'], f'vision_results_{img_name}')
                                os.makedirs(result_dir, exist_ok=True)
                                
                                # 更新图片类型列表，包括新API返回的所有图片
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

            # === 绑定事件 ===

            # 参数验证
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

            # 快速填充按钮
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

            # 预设配置
            apply_preset_btn.click(
                fn=apply_preset_config,
                inputs=[preset_configs],
                outputs=[semantic_classes, semantic_countability, openness_list]
            )

            # 视觉分析 - 使用新的带验证的函数
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
                        
            # Tab 5: 指标计算与报告
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
           
           

        
        # === 事件绑定 ===
        
        # 指标推荐
        recommend_btn.click(
            fn=recommend_metrics,
            inputs=[user_input, openai_key_1],
            outputs=[recommendation_text, recommended_metrics]
        )
        
        # 刷新指标库
        def refresh_metrics():
            df = load_metrics_library()
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
        
        # 选择指标
        def select_metrics_wrapper(selected_items: List[str]) -> str:
            """选择要使用的指标"""
            global app_state
            
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
                
                app_state['selected_metrics'] = selected_metrics
                
                status = f"已选择 {len(selected_metrics)} 个指标"
                if skipped_metrics:
                    status += f"\n跳过了 {len(skipped_metrics)} 个没有代码的指标: {', '.join(skipped_metrics[:3])}"
                    if len(skipped_metrics) > 3:
                        status += f" 等"
                
                return status
            except Exception as e:
                return f"选择失败: {str(e)}"
        
        select_btn.click(
            fn=select_metrics_wrapper,
            inputs=[selected_indices],
            outputs=[selection_status]
        )
        
        # 更新指标库
        update_lib_btn.click(
            fn=update_metrics_library,
            inputs=[metrics_file],
            outputs=[update_status]
        )
        
        # 更新指标下拉列表的函数
        def update_metric_dropdown():
            metrics = components['metrics_manager'].get_all_metrics()
            choices = []
            for metric in metrics:
                metric_name = metric['metric name']
                has_code = components['metrics_manager'].has_metric_code(metric_name)
                status = "✓" if has_code else "✗"
                choices.append(f"{status} {metric_name}")
            return gr.update(choices=choices)
        
        # 获取指标代码状态
        def get_metrics_code_status():
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
        
        # 修改上传代码函数以使用下拉选择
        def upload_metric_code_from_dropdown(metric_selection: str, code_file) -> str:
            if not metric_selection:
                return "请选择一个指标"
            
            # 从选择中提取指标名称（去掉状态标记）
            metric_name = metric_selection[2:] if metric_selection.startswith(('✓ ', '✗ ')) else metric_selection
            
            return upload_metric_code(metric_name, code_file)
        
        # 上传代码 - 更新绑定
        upload_code_btn.click(
            fn=upload_metric_code_from_dropdown,
            inputs=[metric_name_dropdown, code_file],
            outputs=[upload_status]
        ).then(
            fn=get_metrics_code_status,
            outputs=[metrics_code_status]
        ).then(
            fn=update_metric_dropdown,
            outputs=[metric_name_dropdown]
        ).then(
            fn=refresh_metrics,
            outputs=[metrics_library, selected_indices]
        )
        
        # 处理图片
        process_btn.click(
            fn=process_images,
            inputs=[image_files],
            outputs=[process_status, gps_info, enable_heatmap]
        )
        
        # 视觉分析
        analyze_btn.click(
            fn=run_vision_analysis,
            inputs=[semantic_classes, semantic_countability, openness_list],
            outputs=[analysis_status, result_images]
        )
        
        # 计算指标
        calc_btn.click(
            fn=calculate_metrics,
            outputs=[calc_status, metrics_results]
        )
        
        # 生成报告
        def generate_final_report(include_heatmap, key):
            status, path = generate_report(include_heatmap, key)
            if path and os.path.exists(path):
                return status, path
            return status, None
        
        generate_btn.click(
            fn=generate_final_report,
            inputs=[include_heatmap_final, openai_key_2],
            outputs=[report_status, report_file]
        )
        
        # 初始加载
        def initial_load():
            df_metrics, choices_update = refresh_metrics()
            dropdown_update = update_metric_dropdown()
            code_status = get_metrics_code_status()
            return df_metrics, choices_update, dropdown_update, code_status
        
        app.load(
            fn=initial_load, 
            outputs=[metrics_library, selected_indices, metric_name_dropdown, metrics_code_status]
        )
    
    return app

# 启动应用
if __name__ == "__main__":
    # 创建初始指标库文件（如果不存在）
    if not os.path.exists(CONFIG['metrics_library_path']):
        # 从JSON创建Excel
        import json
        json_path = 'library_metrics.json'
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                metrics_data = json.load(f)
            df = pd.DataFrame(metrics_data)
            df.to_excel(CONFIG['metrics_library_path'], index=False)
    
    # 创建并启动应用
    app = create_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )
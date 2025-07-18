"""
城市绿地空间视觉分析系统
主应用程序（模块化版本）
"""

import os
import logging
from ui import create_main_interface, AppState
from modules.api_clients import VisionModelClient, MetricsRecommenderClient
from modules.image_processor import ImageProcessor
from modules.metrics_manager import MetricsManager
from modules.metrics_calculator import MetricsCalculator
from modules.report_generator import ReportGenerator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 配置
CONFIG = {
    'vision_api_url': 'http://127.0.0.1:8000',  # 本地视觉模型API
    'metrics_api_url': 'http://localhost:8001',  # 本地运行的metrics推荐API
    'metrics_library_path': 'data/library_metrics.xlsx',
    'metrics_code_dir': 'data/metrics_code/',
    'output_dir': 'outputs/',
    'temp_dir': 'temp/'
}

def ensure_directories():
    """确保必要的目录存在"""
    directories = [
        'data', 
        'data/metrics_code', 
        'outputs', 
        'temp',
        'ui',
        'ui/tabs'
    ]
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"确保目录存在: {dir_path}")

def initialize_components():
    """初始化各个组件"""
    logger.info("正在初始化组件...")
    
    try:
        # 初始化API客户端
        vision_client = VisionModelClient(CONFIG['vision_api_url'])
        metrics_client = MetricsRecommenderClient(CONFIG['metrics_api_url'])
        
        # 初始化处理器
        image_processor = ImageProcessor()
        
        # 初始化管理器
        metrics_manager = MetricsManager(
            CONFIG['metrics_library_path'], 
            CONFIG['metrics_code_dir']
        )
        
        # 初始化计算器（传入metrics_manager以获取配置）
        metrics_calculator = MetricsCalculator(
            CONFIG['metrics_code_dir'],
            metrics_manager
        )
        
        # 初始化报告生成器
        report_generator = ReportGenerator(CONFIG['output_dir'])
        
        components = {
            'vision_client': vision_client,
            'metrics_client': metrics_client,
            'image_processor': image_processor,
            'metrics_manager': metrics_manager,
            'metrics_calculator': metrics_calculator,
            'report_generator': report_generator
        }
        
        logger.info("所有组件初始化成功")
        return components
        
    except Exception as e:
        logger.error(f"组件初始化失败: {e}")
        raise

def create_initial_metrics_library():
    """创建初始指标库文件（如果不存在）"""
    if not os.path.exists(CONFIG['metrics_library_path']):
        logger.info("创建初始指标库...")
        # 尝试从JSON创建
        import json
        import pandas as pd
        
        json_path = 'library_metrics.json'
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    metrics_data = json.load(f)
                df = pd.DataFrame(metrics_data)
                df.to_excel(CONFIG['metrics_library_path'], index=False)
                logger.info(f"从JSON文件创建了指标库: {CONFIG['metrics_library_path']}")
            except Exception as e:
                logger.warning(f"无法从JSON创建指标库: {e}")
        else:
            # 创建空的指标库
            import pandas as pd
            df = pd.DataFrame(columns=[
                'metric name', 'Primary Category', 'Secondary Attribute',
                'Standard Range', 'Unit', 'Parameter Definition',
                'Data Input', 'Calculation Method', 'Professional Interpretation',
                'required_images'
            ])
            df.to_excel(CONFIG['metrics_library_path'], index=False)
            logger.info("创建了空的指标库")

def main():
    """主函数"""
    try:
        # 确保目录存在
        ensure_directories()
        
        # 创建初始指标库
        create_initial_metrics_library()
        
        # 初始化组件
        components = initialize_components()
        
        # 创建状态管理器
        app_state = AppState()
        
        # 创建界面
        logger.info("正在创建用户界面...")
        app = create_main_interface(components, CONFIG, app_state)
        
        # 启动应用
        logger.info("正在启动应用...")
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True
        )
        
    except Exception as e:
        logger.error(f"应用启动失败: {e}")
        raise

if __name__ == "__main__":
    main()
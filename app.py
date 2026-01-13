"""
GreenSVC-AI 城市绿地空间视觉分析系统 v2.0
主入口
"""

import logging
from config import Config
from ui import create_main_interface, AppState
from modules import (
    VisionModelClient, 
    ImageProcessor, 
    MetricsManager, 
    MetricsCalculator, 
    ReportGenerator
)

# 配置日志
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def init_components():
    """初始化组件"""
    logger.info("初始化组件...")
    
    Config.ensure_directories()
    
    components = {
        'vision_client': VisionModelClient(Config.VISION_API_URL),
        'image_processor': ImageProcessor(),
        'metrics_manager': MetricsManager(
            str(Config.METRICS_LIBRARY_PATH),
            str(Config.METRICS_CODE_DIR)
        ),
        'metrics_calculator': MetricsCalculator(str(Config.METRICS_CODE_DIR)),
        'report_generator': ReportGenerator(str(Config.OUTPUT_DIR))
    }
    
    logger.info("✅ 组件初始化完成")
    return components


def main():
    """主函数"""
    try:
        components = init_components()
        app_state = AppState()
        
        logger.info("创建界面...")
        app = create_main_interface(components, Config, app_state)
        
        logger.info(f"启动服务 (port={Config.GRADIO_SERVER_PORT}, share={Config.GRADIO_SHARE})")
        app.launch(
            server_name="0.0.0.0",
            server_port=Config.GRADIO_SERVER_PORT,
            share=Config.GRADIO_SHARE
        )
        
    except Exception as e:
        logger.error(f"启动失败: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

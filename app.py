"""
GreenSVC-AI: Urban Green Space Visual Analysis System v2.0
Main Entry Point
"""

import logging
from config import Config
from ui import create_main_interface, AppState
from modules import VisionModelClient, MetricsManager, MetricsCalculator

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def init_components():
    """Initialize application components"""
    logger.info("Initializing components...")
    
    Config.ensure_directories()
    
    components = {
        'vision_client': VisionModelClient(Config.VISION_API_URL),
        'metrics_manager': MetricsManager(
            str(Config.METRICS_LIBRARY_PATH),
            str(Config.METRICS_CODE_DIR)
        ),
        'metrics_calculator': MetricsCalculator(str(Config.METRICS_CODE_DIR)),
    }
    
    logger.info("✅ Components initialized")
    return components


def main():
    """Main function"""
    try:
        components = init_components()
        app_state = AppState()
        
        logger.info("Creating interface...")
        app = create_main_interface(components, Config, app_state)
        
        logger.info(f"Starting server (port={Config.GRADIO_SERVER_PORT}, share={Config.GRADIO_SHARE})")
        app.launch(
            server_name="0.0.0.0",
            server_port=Config.GRADIO_SERVER_PORT,
            share=Config.GRADIO_SHARE
        )
        
    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

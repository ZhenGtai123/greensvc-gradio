"""
配置管理模块
从.env文件加载配置
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 加载.env文件
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    # 尝试当前目录
    load_dotenv()


class Config:
    """应用配置"""
    
    # API配置
    GOOGLE_API_KEY: str = os.getenv('GOOGLE_API_KEY', '')
    OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY', '')  # 保留，用于其他功能
    VISION_API_URL: str = os.getenv('VISION_API_URL', 'http://127.0.0.1:8000')
    METRICS_API_URL: str = os.getenv('METRICS_API_URL', 'http://localhost:8001')
    
    # 服务器配置
    GRADIO_SERVER_PORT: int = int(os.getenv('GRADIO_SERVER_PORT', '7860'))
    GRADIO_SHARE: bool = os.getenv('GRADIO_SHARE', 'false').lower() == 'true'
    
    # 日志配置
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    
    # 路径配置
    BASE_DIR: Path = Path(__file__).parent
    DATA_DIR: Path = BASE_DIR / 'data'
    METRICS_LIBRARY_PATH: Path = DATA_DIR / 'library_metrics.xlsx'
    METRICS_CODE_DIR: Path = DATA_DIR / 'metrics_code'
    KNOWLEDGE_BASE_DIR: Path = DATA_DIR / 'knowledge_base'
    OUTPUT_DIR: Path = BASE_DIR / 'outputs'
    TEMP_DIR: Path = BASE_DIR / 'temp'
    
    @classmethod
    def ensure_directories(cls):
        """确保所有目录存在"""
        for dir_path in [cls.DATA_DIR, cls.METRICS_CODE_DIR, 
                         cls.KNOWLEDGE_BASE_DIR, cls.OUTPUT_DIR, cls.TEMP_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def to_dict(cls) -> dict:
        """转换为字典"""
        return {
            'vision_api_url': cls.VISION_API_URL,
            'metrics_api_url': cls.METRICS_API_URL,
            'metrics_library_path': str(cls.METRICS_LIBRARY_PATH),
            'metrics_code_dir': str(cls.METRICS_CODE_DIR),
            'knowledge_base_dir': str(cls.KNOWLEDGE_BASE_DIR),
            'output_dir': str(cls.OUTPUT_DIR),
            'temp_dir': str(cls.TEMP_DIR),
        }


# 创建配置实例
config = Config()

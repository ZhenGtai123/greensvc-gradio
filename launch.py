"""
系统启动器
简化启动流程
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path

def check_requirements():
    """检查系统要求"""
    print("检查系统要求...")
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        print("❌ Python版本过低，需要3.8或更高版本")
        return False
    
    # 检查必要的文件
    required_files = [
        'app.py',
        'requirements.txt',
        'modules/api_clients.py',
        'modules/image_processor.py',
        'modules/metrics_manager.py',
        'modules/metrics_calculator.py',
        'modules/report_generator.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ 缺少必要文件: {', '.join(missing_files)}")
        print("请运行 python setup.py 初始化项目")
        return False
    
    # 检查环境变量文件
    if not os.path.exists('.env'):
        if os.path.exists('.env.template'):
            print("⚠️  未找到.env文件，将使用模板创建...")
            import shutil
            shutil.copy('.env.template', '.env')
            print("✓ 已创建.env文件，请编辑并填入API密钥")
        else:
            print("❌ 未找到环境配置文件")
            return False
    
    # 检查指标库和代码目录
    if not os.path.exists('data'):
        os.makedirs('data', exist_ok=True)
        print("✓ 创建数据目录")
    
    if not os.path.exists('data/metrics_code'):
        os.makedirs('data/metrics_code', exist_ok=True)
        print("✓ 创建指标代码目录")
    
    # 检查是否有示例指标代码
    metrics_code_files = os.listdir('data/metrics_code') if os.path.exists('data/metrics_code') else []
    if not metrics_code_files:
        print("⚠️  未找到指标代码文件")
        print("  提示：运行 python setup.py 创建示例代码")
    else:
        print(f"✓ 找到 {len(metrics_code_files)} 个指标代码文件")
    
    print("✓ 系统要求检查通过")
    return True

def check_dependencies():
    """检查Python依赖"""
    print("\n检查Python依赖...")
    
    try:
        import gradio
        import pandas
        import numpy
        import cv2
        import requests
        print("✓ 核心依赖已安装")
        return True
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请运行: pip install -r requirements.txt")
        return False

def start_metrics_api(port=8001):
    """启动指标推荐API"""
    print(f"\n启动指标推荐API (端口 {port})...")
    
    # 检查是否已有metrics_recommender.py
    if not os.path.exists('metrics_recommender.py'):
        print("⚠️  未找到metrics_recommender.py，将在嵌入模式下运行")
        return None
    
    # 启动API
    process = subprocess.Popen(
        [sys.executable, 'metrics_recommender.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # 等待启动
    time.sleep(3)
    
    # 检查是否成功启动
    if process.poll() is None:
        print(f"✓ 指标推荐API已启动 (PID: {process.pid})")
        return process
    else:
        print("⚠️  指标推荐API启动失败，将使用本地推荐器")
        return None

def start_main_app(share=False, port=7860):
    """启动主应用"""
    print(f"\n启动主应用 (端口 {port})...")
    
    env = os.environ.copy()
    env['GRADIO_SERVER_PORT'] = str(port)
    
    if share:
        env['GRADIO_SHARE'] = 'true'
    
    try:
        subprocess.run([sys.executable, 'app.py'], env=env)
    except KeyboardInterrupt:
        print("\n\n正在关闭应用...")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='城市绿地空间视觉分析系统启动器')
    parser.add_argument('--share', action='store_true', help='创建公共分享链接')
    parser.add_argument('--port', type=int, default=7860, help='主应用端口')
    parser.add_argument('--metrics-port', type=int, default=8001, help='指标API端口')
    parser.add_argument('--skip-metrics-api', action='store_true', help='跳过指标API启动')
    parser.add_argument('--setup', action='store_true', help='运行设置脚本')
    
    args = parser.parse_args()
    
    print("=== 城市绿地空间视觉分析系统 ===\n")
    
    # 如果指定了setup参数，运行设置脚本
    if args.setup:
        subprocess.run([sys.executable, 'setup.py'])
        return
    
    # 检查系统要求
    if not check_requirements():
        print("\n启动失败，请先解决上述问题")
        return
    
    # 检查依赖
    if not check_dependencies():
        print("\n启动失败，请先安装依赖")
        return
    
    # 启动服务
    metrics_process = None
    
    try:
        # 启动指标推荐API
        if not args.skip_metrics_api:
            metrics_process = start_metrics_api(args.metrics_port)
        
        # 启动主应用
        print(f"\n{'='*50}")
        print(f"系统即将启动...")
        print(f"主应用地址: http://localhost:{args.port}")
        if args.share:
            print("将创建公共分享链接...")
        print(f"{'='*50}\n")
        
        time.sleep(2)
        start_main_app(share=args.share, port=args.port)
        
    except Exception as e:
        print(f"\n发生错误: {e}")
    finally:
        # 清理进程
        if metrics_process and metrics_process.poll() is None:
            print("\n关闭指标推荐API...")
            metrics_process.terminate()
            metrics_process.wait()
        
        print("\n系统已关闭")

if __name__ == "__main__":
    main()
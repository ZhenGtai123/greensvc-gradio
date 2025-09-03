"""
API配置标签页
用于动态配置Vision API URL
"""

import gradio as gr
import logging

logger = logging.getLogger(__name__)

def create_api_config_tab(components, app_state):
    """创建API配置标签页"""
    
    vision_client = components['vision_client']
    
    def test_and_update_url(url):
        """测试并更新URL"""
        if not url or not url.strip():
            return "❌ 请输入URL", vision_client.base_url
        
        url = url.strip()
        old_url = vision_client.base_url
        
        # 尝试更新URL
        vision_client.base_url = url.rstrip('/')
        
        # 测试连接
        if vision_client.check_health():
            app_state.vision_api_url = url  # 保存到状态
            return f"✅ 连接成功！API已更新到: {url}", url
        else:
            vision_client.base_url = old_url  # 恢复原URL
            return f"❌ 无法连接到: {url}\n请检查Colab API是否正在运行", old_url
    
    def get_current_status():
        """获取当前状态"""
        current_url = vision_client.base_url
        if vision_client.check_health():
            return f"✅ API在线: {current_url}"
        else:
            return f"❌ API离线: {current_url}"
    
    with gr.Tab("⚙️ API配置"):
        gr.Markdown("""
        ### 🔧 配置Vision API
        1. 在Google Colab运行API notebook
        2. 复制ngrok URL (例如: https://xxxx.ngrok-free.app)
        3. 粘贴到下方并点击连接
        """)
        
        with gr.Row():
            url_input = gr.Textbox(
                label="API URL",
                placeholder="https://xxxx.ngrok-free.app",
                value=vision_client.base_url,
                scale=3
            )
            connect_btn = gr.Button("🔌 连接", variant="primary", scale=1)
        
        status_text = gr.Textbox(
            label="状态",
            value=get_current_status(),
            interactive=False
        )
        
        # 事件绑定
        connect_btn.click(
            fn=test_and_update_url,
            inputs=url_input,
            outputs=[status_text, url_input]
        )
        
        # 添加刷新按钮
        refresh_btn = gr.Button("🔄 刷新状态", variant="secondary", size="sm")
        refresh_btn.click(
            fn=get_current_status,
            outputs=status_text
        )
    
    return {'url_input': url_input, 'status_text': status_text}
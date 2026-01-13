"""
Tab 0: API Configuration
"""

import gradio as gr


def create_api_config_tab(components: dict, app_state, config):
    """Create API Configuration Tab"""
    
    with gr.Tab("0. API Config"):
        gr.Markdown("## ⚙️ API Configuration")
        
        with gr.Group():
            gr.Markdown("### Vision API")
            gr.Markdown("*Vision model API for semantic segmentation (local or remote)*")
            
            with gr.Row():
                vision_url = gr.Textbox(
                    label="Vision API URL",
                    value=config.VISION_API_URL,
                    placeholder="http://127.0.0.1:8000"
                )
                test_btn = gr.Button("Test Connection")
            
            api_status = gr.Textbox(label="Status", interactive=False)
        
        with gr.Group():
            gr.Markdown("### Google API (Gemini)")
            gr.Markdown("*For AI-powered indicator recommendation*")
            
            google_key = gr.Textbox(
                label="Google API Key",
                value=config.GOOGLE_API_KEY[:20] + "..." if config.GOOGLE_API_KEY else "",
                type="password",
                placeholder="AIza..."
            )
            
            if config.GOOGLE_API_KEY:
                gr.Markdown("*✅ API Key loaded from .env file*")
            else:
                gr.Markdown("*Configure in .env file or enter manually*")
        
        with gr.Group():
            gr.Markdown("### OpenAI API (Optional)")
            gr.Markdown("*For AI-powered report analysis*")
            
            openai_key = gr.Textbox(
                label="OpenAI API Key",
                value=config.OPENAI_API_KEY[:20] + "..." if config.OPENAI_API_KEY else "",
                type="password",
                placeholder="sk-..."
            )
        
        def test_vision_api(url):
            try:
                vc = components.get('vision_client')
                if vc:
                    vc.base_url = url
                    if vc.check_health():
                        return f"✅ Connected: {url}"
                return "❌ Connection failed"
            except Exception as e:
                return f"❌ Error: {e}"
        
        def update_vision_url(url):
            if components.get('vision_client'):
                components['vision_client'].base_url = url
            return f"Updated: {url}"
        
        test_btn.click(test_vision_api, [vision_url], [api_status])
        vision_url.change(update_vision_url, [vision_url], [api_status])
        
        return {'vision_url': vision_url, 'api_status': api_status}

"""
Tab 4: Vision Analysis
Semantic segmentation, depth estimation, FMB segmentation
"""

import gradio as gr
import os
import json
import requests
import logging
from typing import Dict, List
from pathlib import Path

logger = logging.getLogger(__name__)


def load_semantic_config(config_path: str) -> Dict:
    """Load semantic configuration from JSON file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        classes = [item['name'] for item in config]
        colors = {item['name']: item['color'] for item in config}
        countability = [item.get('countable', 0) for item in config]
        openness = [item.get('openness', 0) for item in config]
        
        return {
            'classes': classes,
            'colors': colors,
            'countability': countability,
            'openness': openness,
            'raw': config
        }
    except Exception as e:
        logger.error(f"Failed to load semantic config: {e}")
        return None


def hex_to_rgb(hex_color: str, bgr: bool = True) -> List[int]:
    """Convert hex color to RGB."""
    h = hex_color.lstrip('#')
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    return [b, g, r] if bgr else [r, g, b]


def create_vision_analysis_tab(components: dict, app_state, config):
    """Create Vision Analysis Tab"""
    
    # Load default semantic configuration
    default_config_path = Path(config.DATA_DIR) / 'Semantic_configuration.json'
    default_semantic = load_semantic_config(str(default_config_path)) if default_config_path.exists() else None
    
    with gr.Tab("4. Vision Analysis"):
        gr.Markdown("""
        ## 🎯 Vision Analysis
        Semantic segmentation, depth estimation, foreground-middleground-background segmentation
        """)
        
        # API Status
        with gr.Row():
            api_status = gr.Textbox(label="API Status", value="Not checked", interactive=False)
            check_btn = gr.Button("Check API")
        
        # Image List
        with gr.Group():
            gr.Markdown("### 📁 Images to Analyze")
            refresh_btn = gr.Button("🔄 Refresh Image List")
            with gr.Row():
                img_count = gr.Textbox(label="Image Count", value="0", interactive=False)
                zone_info = gr.Textbox(label="Zone Distribution", interactive=False)
            preview_gallery = gr.Gallery(label="Preview", columns=6, height="auto")
        
        # Configuration
        with gr.Group():
            gr.Markdown("### ⚙️ Analysis Configuration")
            
            config_source = gr.Radio(
                label="Configuration Source",
                choices=["Default Config (150 classes)", "Upload Config File", "Manual Input"],
                value="Default Config (150 classes)"
            )
            
            # Upload config file
            with gr.Row(visible=False) as upload_row:
                config_file = gr.File(label="Semantic Config (.json)", file_types=[".json"])
                load_config_btn = gr.Button("Load")
            
            # Config preview
            with gr.Accordion("Configuration Preview", open=False):
                config_info = gr.Textbox(
                    label="Current Config",
                    value=f"Loaded: {len(default_semantic['classes'])} classes" if default_semantic else "Not loaded",
                    interactive=False
                )
                
                classes_text = gr.Textbox(
                    label="Semantic Classes",
                    value="\n".join(default_semantic['classes'][:20]) + "\n..." if default_semantic else "",
                    lines=6
                )
                
                with gr.Row():
                    countability_text = gr.Textbox(
                        label="Countability (0/1)",
                        value=",".join(map(str, default_semantic['countability'][:20])) + ",..." if default_semantic else ""
                    )
                    openness_text = gr.Textbox(
                        label="Openness (0/1)",
                        value=",".join(map(str, default_semantic['openness'][:20])) + ",..." if default_semantic else ""
                    )
            
            # Model settings
            with gr.Row():
                encoder = gr.Radio(label="Encoder", choices=["vitb", "vits"], value="vitb")
                hole_fill = gr.Checkbox(label="Hole Filling", value=True)
                blur = gr.Checkbox(label="Median Blur", value=True)
        
        # Analyze button
        analyze_btn = gr.Button("🚀 Start Analysis", variant="primary", size="lg")
        status = gr.Textbox(label="Status", interactive=False)
        
        # Results
        result_gallery = gr.Gallery(label="Analysis Results", columns=5, height="auto")
        stats = gr.Textbox(label="Statistics", lines=2, interactive=False)
        
        # State
        current_config = gr.State(default_semantic)
        
        # ========== Event Handlers ==========
        
        def toggle_config_source(source):
            if source == "Upload Config File":
                return gr.update(visible=True)
            return gr.update(visible=False)
        
        def load_uploaded_config(file):
            if not file:
                return None, "Please select a file", "", "", ""
            
            cfg = load_semantic_config(file.name)
            if cfg:
                return (
                    cfg,
                    f"Loaded: {len(cfg['classes'])} classes",
                    "\n".join(cfg['classes'][:20]) + "\n...",
                    ",".join(map(str, cfg['countability'][:20])) + ",...",
                    ",".join(map(str, cfg['openness'][:20])) + ",..."
                )
            return None, "Load failed", "", "", ""
        
        def check_api():
            try:
                vc = components.get('vision_client')
                if vc and vc.check_health():
                    return f"✅ Connected ({vc.base_url})"
                return "❌ Not connected"
            except:
                return "❌ Cannot connect"
        
        def refresh_images():
            images = app_state.project_query.uploaded_images
            if not images:
                return "0", "None", []
            
            zones = {}
            ungrouped = 0
            gallery = []
            
            for img in images:
                gallery.append((img.filepath, img.filename))
                if img.zone_id:
                    z = app_state.project_query.get_zone(img.zone_id)
                    name = z.zone_name if z else img.zone_id
                    zones[name] = zones.get(name, 0) + 1
                else:
                    ungrouped += 1
            
            info = ", ".join([f"{k}:{v}" for k, v in zones.items()])
            if ungrouped:
                info += f", Ungrouped:{ungrouped}"
            
            return str(len(images)), info or "None", gallery
        
        def run_analysis(cfg, classes, count_str, open_str, enc, fill, blur_v):
            try:
                vc = components.get('vision_client')
                if not vc:
                    return "❌ API not configured", [], ""
                
                images = app_state.project_query.uploaded_images
                if not images:
                    return "❌ Please upload images first", [], ""
                
                # Parse config
                if cfg:
                    cls_list = cfg['classes']
                    cnt_list = cfg['countability']
                    op_list = cfg['openness']
                    colors = cfg['colors']
                else:
                    cls_list = [c.strip() for c in classes.split('\n') if c.strip()]
                    cnt_list = [int(x.strip()) for x in count_str.split(',') if x.strip()]
                    op_list = [int(x.strip()) for x in open_str.split(',') if x.strip()]
                    colors = {}
                
                # Build color mapping
                sem_colors = {"0": [0, 0, 0]}
                for i, c in enumerate(cls_list):
                    color = colors.get(c, "#808080")
                    sem_colors[str(i+1)] = hex_to_rgb(color)
                
                results = []
                success = 0
                
                type_names = {
                    'semantic_map': 'Semantic', 'depth_map': 'Depth',
                    'fmb_map': 'FMB', 'openness_map': 'Openness',
                    'foreground_map': 'Foreground', 'middleground_map': 'Middleground', 
                    'background_map': 'Background',
                }
                
                for img in images:
                    path = img.filepath
                    name = os.path.basename(path).split('.')[0]
                    
                    req = {
                        "image_id": f"img_{name}",
                        "semantic_classes": cls_list,
                        "semantic_countability": cnt_list,
                        "openness_list": op_list,
                        "semantic_colors": sem_colors,
                        "encoder": enc,
                        "enable_hole_filling": fill,
                        "enable_median_blur": blur_v
                    }
                    
                    try:
                        with open(path, 'rb') as f:
                            files = {'file': (os.path.basename(path), f, 'image/jpeg')}
                            data = {'request_data': json.dumps(req)}
                            resp = requests.post(
                                f"{vc.base_url}/analyze", 
                                files=files, data=data, timeout=600
                            )
                        
                        if resp.status_code == 200:
                            result = resp.json()
                            if result.get('status') == 'success':
                                success += 1
                                app_state.add_vision_result(path, result)
                                
                                if 'images' in result:
                                    save_dir = os.path.join(str(config.TEMP_DIR), f'vision_{name}')
                                    os.makedirs(save_dir, exist_ok=True)
                                    
                                    for img_type, hex_data in result['images'].items():
                                        if hex_data and isinstance(hex_data, str):
                                            img_path = os.path.join(save_dir, f'{img_type}.png')
                                            with open(img_path, 'wb') as f:
                                                f.write(bytes.fromhex(hex_data))
                                            label = type_names.get(img_type, img_type)
                                            results.append((img_path, f"{name}-{label}"))
                    except Exception as e:
                        logger.error(f"Analysis failed {path}: {e}")
                
                summary = f"Processed: {success}/{len(images)}\nClasses: {len(cls_list)}"
                return f"✅ Complete, {success} images processed", results, summary
                
            except Exception as e:
                logger.error(f"Analysis error: {e}")
                return f"❌ Error: {e}", [], ""
        
        # ===== Bind Events =====
        config_source.change(toggle_config_source, [config_source], [upload_row])
        load_config_btn.click(
            load_uploaded_config, 
            [config_file], 
            [current_config, config_info, classes_text, countability_text, openness_text]
        )
        check_btn.click(check_api, outputs=[api_status])
        refresh_btn.click(refresh_images, outputs=[img_count, zone_info, preview_gallery])
        
        analyze_btn.click(
            run_analysis,
            [current_config, classes_text, countability_text, openness_text, encoder, hole_fill, blur],
            [status, result_gallery, stats]
        )
        
        return {'result_gallery': result_gallery}

"""
Tab 5: Metrics Calculation
Uses MetricsCalculator and MetricsManager from modules
Based on GreenSVC Stage 2.5 methodology
"""

import gradio as gr
import pandas as pd
import numpy as np
import os
import json
import logging
from typing import Dict, List
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


def calculate_statistics(values: List[float]) -> Dict:
    """Calculate descriptive statistics"""
    if not values:
        return {'N': 0}
    
    arr = np.array(values)
    q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))
    
    return {
        'N': len(values),
        'Mean': round(mean_val, 3),
        'Std': round(std_val, 3),
        'Min': round(float(np.min(arr)), 3),
        'Q1': round(float(q1), 3),
        'Median': round(float(np.median(arr)), 3),
        'Q3': round(float(q3), 3),
        'Max': round(float(np.max(arr)), 3),
        'CV(%)': round(float(std_val / mean_val * 100), 2) if mean_val != 0 else 0
    }


def create_metrics_calculation_tab(components: dict, app_state, config):
    """Create Metrics Calculation Tab"""
    
    # Get components
    metrics_calculator = components.get('metrics_calculator')
    metrics_manager = components.get('metrics_manager')
    
    with gr.Tab("5. Metrics Calculation"):
        gr.Markdown("""
        ## 📊 Indicator Calculation
        Calculate indicators from semantic segmentation masks using Stage 2.5 methodology
        """)
        
        # ===== Configuration =====
        with gr.Group():
            gr.Markdown("### ⚙️ Semantic Configuration")
            
            with gr.Row():
                semantic_file = gr.File(
                    label="Semantic Configuration (.json)",
                    file_types=[".json"]
                )
                load_semantic_btn = gr.Button("Load Config")
            
            semantic_status = gr.Textbox(label="Status", interactive=False)
            
            # Auto-load default config
            default_config = Path(config.DATA_DIR) / 'Semantic_configuration.json'
            if default_config.exists():
                gr.Markdown(f"*Default config available: {default_config.name}*")
        
        # ===== Calculator Selection =====
        with gr.Group():
            gr.Markdown("### 📐 Select Calculators")
            
            refresh_calcs_btn = gr.Button("🔄 Refresh List")
            
            available_calcs = gr.CheckboxGroup(
                label="Available Calculators (from Metrics Library)",
                choices=[]
            )
            
            calc_info = gr.Textbox(label="Calculator Info", lines=3, interactive=False)
        
        # ===== Image Source =====
        with gr.Group():
            gr.Markdown("### 🖼️ Image Source")
            
            image_source = gr.Radio(
                label="Select Image Source",
                choices=[
                    "From Vision Analysis Results",
                    "Upload Mask Images"
                ],
                value="From Vision Analysis Results"
            )
            
            with gr.Row(visible=False) as upload_row:
                mask_upload = gr.File(
                    label="Upload Mask Images",
                    file_count="multiple",
                    file_types=["image"]
                )
            
            image_info = gr.Textbox(label="Images", interactive=False)
            refresh_images_btn = gr.Button("Refresh Images")
        
        # ===== Calculation =====
        calculate_btn = gr.Button("🚀 Calculate Indicators", variant="primary", size="lg")
        calc_status = gr.Textbox(label="Calculation Status", interactive=False)
        
        # ===== Results =====
        with gr.Group(visible=False) as results_group:
            gr.Markdown("### 📈 Results")
            
            with gr.Row():
                total_images = gr.Textbox(label="Images Processed", interactive=False)
                total_indicators = gr.Textbox(label="Indicators", interactive=False)
            
            results_df = gr.Dataframe(
                label="Calculation Results",
                interactive=False,
                wrap=True
            )
            
            with gr.Accordion("Statistics Summary", open=False):
                stats_df = gr.Dataframe(
                    label="Descriptive Statistics",
                    interactive=False
                )
            
            with gr.Accordion("Export Results", open=False):
                export_btn = gr.Button("Export to JSON")
                export_file = gr.File(label="Download", visible=False)
        
        # ========== Event Handlers ==========
        
        def load_semantic_config(file):
            if not file:
                # Try default config
                default_path = Path(config.DATA_DIR) / 'Semantic_configuration.json'
                if default_path.exists():
                    if metrics_calculator:
                        if metrics_calculator.load_semantic_colors(str(default_path)):
                            return f"✅ Loaded default config ({len(metrics_calculator.semantic_colors)} classes)"
                return "Please select a file or use default config"
            
            if metrics_calculator:
                if metrics_calculator.load_semantic_colors(file.name):
                    return f"✅ Loaded {len(metrics_calculator.semantic_colors)} semantic classes"
            return "❌ Failed to load configuration"
        
        def toggle_upload(source):
            return gr.update(visible=source == "Upload Mask Images")
        
        def refresh_calculators():
            """Get available calculators from MetricsManager"""
            choices = []
            
            if metrics_manager:
                metrics_manager.scan_calculators()
                for calc in metrics_manager.get_all_calculators():
                    choices.append(f"{calc.get('id', '')}: {calc.get('name', '')}")
            
            # Also check selected metrics from recommendation
            selected = app_state.get_selected_metrics()
            if selected:
                for m in selected:
                    code = m.get('indicator_code', '')
                    if code and code not in [c.split(":")[0] for c in choices]:
                        choices.append(f"{code}: (from recommendation)")
            
            info = f"Found {len(choices)} calculators"
            return gr.update(choices=choices), info
        
        def refresh_images(source, uploaded):
            if source == "From Vision Analysis Results":
                vision_results = app_state.get_vision_results()
                if vision_results:
                    return f"✅ {len(vision_results)} images from vision analysis"
                return "No vision analysis results. Run Vision Analysis first."
            else:
                if uploaded:
                    return f"✅ {len(uploaded)} uploaded images"
                return "No images uploaded"
        
        def run_calculation(selected_calcs, source, uploaded_files):
            try:
                if not selected_calcs:
                    return ("❌ Please select at least one calculator", "", "", 
                            pd.DataFrame(), pd.DataFrame(), 
                            gr.update(visible=False), gr.update(visible=False))
                
                if not metrics_calculator:
                    return ("❌ MetricsCalculator not initialized", "", "",
                            pd.DataFrame(), pd.DataFrame(),
                            gr.update(visible=False), gr.update(visible=False))
                
                # Load semantic colors if not already loaded
                if not metrics_calculator.semantic_colors:
                    default_path = Path(config.DATA_DIR) / 'Semantic_configuration.json'
                    if default_path.exists():
                        metrics_calculator.load_semantic_colors(str(default_path))
                    else:
                        return ("❌ Please load semantic configuration first", "", "",
                                pd.DataFrame(), pd.DataFrame(),
                                gr.update(visible=False), gr.update(visible=False))
                
                # Get image paths
                if source == "From Vision Analysis Results":
                    vision_results = app_state.get_vision_results()
                    if not vision_results:
                        return ("❌ No vision analysis results", "", "",
                                pd.DataFrame(), pd.DataFrame(),
                                gr.update(visible=False), gr.update(visible=False))
                    
                    # Get semantic map paths from vision results
                    image_paths = []
                    for path, result in vision_results.items():
                        name = os.path.basename(path).split('.')[0]
                        sem_path = os.path.join(str(config.TEMP_DIR), f'vision_{name}', 'semantic_map.png')
                        if os.path.exists(sem_path):
                            image_paths.append(sem_path)
                        else:
                            image_paths.append(path)
                else:
                    if not uploaded_files:
                        return ("❌ No images uploaded", "", "",
                                pd.DataFrame(), pd.DataFrame(),
                                gr.update(visible=False), gr.update(visible=False))
                    image_paths = [f.name for f in uploaded_files]
                
                # Extract indicator IDs
                indicator_ids = [sel.split(":")[0].strip() for sel in selected_calcs]
                
                # Calculate using MetricsCalculator
                results_df_data = metrics_calculator.calculate_batch(indicator_ids, image_paths)
                
                if results_df_data.empty:
                    return ("❌ No results calculated", "", "",
                            pd.DataFrame(), pd.DataFrame(),
                            gr.update(visible=False), gr.update(visible=False))
                
                # Calculate statistics per indicator
                all_stats = []
                for ind_id in indicator_ids:
                    ind_data = results_df_data[
                        (results_df_data['Indicator'] == ind_id) & 
                        (results_df_data['Success'] == True)
                    ]
                    if not ind_data.empty and ind_data['Value'].notna().any():
                        values = ind_data['Value'].dropna().tolist()
                        if values:
                            stats = calculate_statistics(values)
                            stats['Indicator'] = ind_id
                            stats['Name'] = ind_data['Name'].iloc[0] if len(ind_data) > 0 else ''
                            stats['Unit'] = ind_data['Unit'].iloc[0] if len(ind_data) > 0 else ''
                            all_stats.append(stats)
                
                stats_df_data = pd.DataFrame(all_stats) if all_stats else pd.DataFrame()
                
                # Store in app state
                app_state.set_metrics_results(results_df_data)
                
                success_count = results_df_data['Success'].sum()
                
                return (
                    f"✅ Calculation complete ({success_count} successful)",
                    str(len(image_paths)),
                    str(len(indicator_ids)),
                    results_df_data,
                    stats_df_data,
                    gr.update(visible=True),
                    gr.update(visible=False)
                )
                
            except Exception as e:
                logger.error(f"Calculation error: {e}", exc_info=True)
                return (f"❌ Error: {e}", "", "",
                        pd.DataFrame(), pd.DataFrame(),
                        gr.update(visible=False), gr.update(visible=False))
        
        def export_results():
            try:
                results = app_state.get_metrics_results()
                if results.empty:
                    return gr.update(visible=False)
                
                output_dir = Path(config.OUTPUT_DIR)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                output_path = output_dir / f"calculation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                
                output = {
                    'metadata': {
                        'generated_at': datetime.now().isoformat(),
                        'system': 'GreenSVC-AI Stage 2.5'
                    },
                    'results': results.to_dict(orient='records')
                }
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(output, f, ensure_ascii=False, indent=2, default=str)
                
                return gr.update(value=str(output_path), visible=True)
            except Exception as e:
                logger.error(f"Export error: {e}")
                return gr.update(visible=False)
        
        # ===== Bind Events =====
        load_semantic_btn.click(load_semantic_config, [semantic_file], [semantic_status])
        image_source.change(toggle_upload, [image_source], [upload_row])
        refresh_calcs_btn.click(refresh_calculators, outputs=[available_calcs, calc_info])
        refresh_images_btn.click(refresh_images, [image_source, mask_upload], [image_info])
        
        calculate_btn.click(
            run_calculation,
            [available_calcs, image_source, mask_upload],
            [calc_status, total_images, total_indicators, results_df, stats_df, results_group, export_file]
        )
        
        export_btn.click(export_results, outputs=[export_file])
        
        return {'results_df': results_df, 'stats_df': stats_df}

"""
Tab 5: Metrics Calculation
Calculate indicators from semantic segmentation masks

Supports:
- Folder structure: mask/zone_id/layer/image.png
- Layers: full, foreground, middleground, background
- Individual image upload
"""

import gradio as gr
import pandas as pd
import numpy as np
import os
import json
import glob
import logging
from typing import Dict, List, Any, Tuple
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

# Stage 2.5 Configuration
LAYERS = ["full", "foreground", "middleground", "background"]


def calculate_statistics(values: List[float]) -> Dict:
    """
    Calculate descriptive statistics (from processing_layer.py)
    """
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
        'Range': round(float(np.max(arr) - np.min(arr)), 3),
        'IQR': round(float(q3 - q1), 3),
        'Variance': round(float(np.var(arr)), 3),
        'CV(%)': round(float(std_val / mean_val * 100), 2) if mean_val != 0 else 0
    }


def scan_zone_images(base_path: str, zone_id: str, layers: List[str]) -> Dict[str, List[str]]:
    """
    Scan mask folder for images (from input_layer.py)
    
    Returns: {layer: [filepath1, filepath2, ...]}
    """
    zone_images = {}
    
    for layer in layers:
        layer_path = os.path.join(base_path, zone_id, layer)
        
        if os.path.exists(layer_path):
            png_files = glob.glob(os.path.join(layer_path, "*.png"))
            jpg_files = glob.glob(os.path.join(layer_path, "*.jpg"))
            jpeg_files = glob.glob(os.path.join(layer_path, "*.jpeg"))
            
            all_files = sorted(png_files + jpg_files + jpeg_files)
            zone_images[layer] = all_files
        else:
            zone_images[layer] = []
    
    return zone_images


def process_zone(zone: Dict, zone_images: Dict[str, List[str]], 
                 calculator_func, indicator_info: Dict) -> Dict:
    """
    Process all images in a zone (from processing_layer.py)
    """
    zone_id = zone.get('id', zone.get('zone_id', ''))
    
    results = {
        'zone_id': zone_id,
        'zone_name': zone.get('name', zone.get('zone_name', '')),
        'area_sqm': zone.get('area_sqm', 0),
        'status': zone.get('status', 'unknown'),
        'layers': {},
        'all_values': [],
        'values_by_layer': {},
        'images_processed': 0,
        'images_failed': 0,
        'images_no_data': 0
    }
    
    for layer, image_paths in zone_images.items():
        layer_results = {
            'images': [],
            'values': [],
            'statistics': {}
        }
        
        for image_path in image_paths:
            result = calculator_func(image_path)
            
            if result.get('success', False):
                image_data = {
                    'filename': os.path.basename(image_path),
                    'value': result.get('value')
                }
                
                # Add extra fields
                for key, val in result.items():
                    if key not in ['success', 'value', 'error']:
                        image_data[key] = val
                
                layer_results['images'].append(image_data)
                
                if result.get('value') is not None:
                    layer_results['values'].append(result['value'])
                    results['all_values'].append(result['value'])
                else:
                    results['images_no_data'] += 1
                
                results['images_processed'] += 1
            else:
                results['images_failed'] += 1
        
        # Layer statistics
        if layer_results['values']:
            layer_results['statistics'] = calculate_statistics(layer_results['values'])
        
        results['layers'][layer] = layer_results
        results['values_by_layer'][layer] = layer_results['values']
    
    return results


def build_output_json(indicator: Dict, all_zone_results: List[Dict], 
                      all_values: List[float], all_values_by_layer: Dict,
                      query_data: Dict = None, semantic_config_path: str = None) -> Dict:
    """
    Build Stage 2.5 output JSON structure (from output_layer.py)
    """
    # Overall statistics
    descriptive_stats = calculate_statistics(all_values)
    
    # Layer statistics
    layer_overall_stats = {}
    for layer in LAYERS:
        if all_values_by_layer.get(layer):
            layer_overall_stats[layer] = calculate_statistics(all_values_by_layer[layer])
        else:
            layer_overall_stats[layer] = {'N': 0, 'Mean': None, 'note': 'No images found'}
    
    # Zone statistics table
    zone_statistics = []
    for zr in all_zone_results:
        if zr['all_values']:
            zone_stat = {
                'Zone': zr['zone_name'],
                'Area_ID': zr['zone_id'],
                'Area_sqm': zr['area_sqm'],
                'Status': zr['status'],
                'Indicator': indicator['id'],
                'N_total': len(zr['all_values']),
                'Mean_overall': round(float(np.mean(zr['all_values'])), 3),
                'Std_overall': round(float(np.std(zr['all_values'])), 3),
                'Min_overall': round(float(min(zr['all_values'])), 3),
                'Max_overall': round(float(max(zr['all_values'])), 3)
            }
            
            # Layer statistics
            for layer in LAYERS:
                layer_stats = zr['layers'].get(layer, {}).get('statistics', {})
                zone_stat[f'{layer}_N'] = layer_stats.get('N', 0)
                zone_stat[f'{layer}_Mean'] = layer_stats.get('Mean', None)
                zone_stat[f'{layer}_Std'] = layer_stats.get('Std', None)
            
            zone_statistics.append(zone_stat)
    
    # Build output
    output = {
        'computation_metadata': {
            'version': '2.5',
            'generated_at': datetime.now().isoformat(),
            'system': 'GreenSVC-AI Stage 2.5',
            'indicator_id': indicator['id'],
            'semantic_config': os.path.basename(semantic_config_path) if semantic_config_path else None,
            'color_matching': 'exact',
            'note': 'Images processed from mask folders, all layers'
        },
        'indicator_definition': {
            'id': indicator['id'],
            'name': indicator.get('name', ''),
            'definition': indicator.get('definition', ''),
            'unit': indicator.get('unit', ''),
            'formula': indicator.get('formula', ''),
            'target_direction': indicator.get('target_direction', ''),
            'category': indicator.get('category', ''),
            'calc_type': indicator.get('calc_type', 'ratio'),
            'semantic_classes': indicator.get('target_classes', [])
        },
        'computation_summary': {
            'total_zones': len(all_zone_results),
            'total_images_analyzed': sum(r['images_processed'] for r in all_zone_results),
            'images_failed': sum(r['images_failed'] for r in all_zone_results),
            'images_no_data': sum(r.get('images_no_data', 0) for r in all_zone_results),
            'layers_processed': LAYERS,
            'images_per_layer': {layer: len(all_values_by_layer.get(layer, [])) for layer in LAYERS}
        },
        'descriptive_statistics_overall': {
            'Indicator': indicator['id'],
            'Name': indicator.get('name', ''),
            'Unit': indicator.get('unit', ''),
            **descriptive_stats
        },
        'descriptive_statistics_by_layer': layer_overall_stats,
        'zone_statistics': zone_statistics,
        'layer_results': {zr['zone_id']: zr['layers'] for zr in all_zone_results}
    }
    
    return output


def create_metrics_calculation_tab(components: dict, app_state, config):
    """Create Metrics Calculation Tab (Stage 2.5)"""
    
    metrics_calculator = components.get('metrics_calculator')
    metrics_manager = components.get('metrics_manager')
    
    with gr.Tab("5. Metrics Calculation"):
        gr.Markdown("""
        ## 📊 Indicator Calculation
        
        Calculate indicators from semantic segmentation masks.
        
        **Layers**: full, foreground, middleground, background
        """)
        
        # ===== Configuration =====
        with gr.Group():
            gr.Markdown("### ⚙️ Configuration")
            
            with gr.Row():
                semantic_file = gr.File(label="Semantic Config (.json or .xlsx)", file_types=[".json", ".xlsx"])
                load_config_btn = gr.Button("Load Config")
            
            config_status = gr.Textbox(label="Semantic Status", interactive=False,
                                       value="Click 'Load Config' to load default or upload custom")
        
        # ===== Calculator Selection =====
        with gr.Group():
            gr.Markdown("### 📐 Select Calculator")
            
            refresh_btn = gr.Button("🔄 Refresh Calculators")
            
            calculator_dropdown = gr.Dropdown(
                label="Select Indicator Calculator",
                choices=[],
                info="Select one calculator_layer_IND_*.py"
            )
            
            calc_details = gr.JSON(label="Calculator Info", visible=False)
        
        # ===== Image Source =====
        with gr.Group():
            gr.Markdown("### 🖼️ Mask Images")
            
            gr.Markdown("""
            **Option A**: Use folder structure:
            ```
            mask_folder/zone_id/layer/image.png
            ```
            
            **Option B**: Upload individual mask images (treated as 'full' layer)
            """)
            
            input_mode = gr.Radio(
                label="Input Mode",
                choices=["Folder Structure", "Upload Images"],
                value="Upload Images"
            )
            
            mask_folder_input = gr.Textbox(
                label="Mask Folder Path",
                placeholder="/path/to/mask/",
                visible=False
            )
            
            mask_upload = gr.File(
                label="Upload Mask Images",
                file_count="multiple",
                file_types=["image"]
            )
            
            scan_btn = gr.Button("📂 Scan Folder", visible=False)
            img_status = gr.Textbox(label="Image Status", interactive=False)
        
        # ===== Calculate =====
        calculate_btn = gr.Button("🚀 Calculate", variant="primary", size="lg")
        progress_text = gr.Textbox(label="Progress", interactive=False)
        
        # ===== Results =====
        with gr.Group(visible=False) as results_group:
            gr.Markdown("### 📈 Results")
            
            with gr.Row():
                result_indicator = gr.Textbox(label="Indicator", interactive=False)
                result_total = gr.Textbox(label="Total Images", interactive=False)
                result_mean = gr.Textbox(label="Overall Mean", interactive=False)
            
            # Zone statistics table
            zone_stats_table = gr.Dataframe(
                label="Zone Statistics",
                interactive=False,
                wrap=True
            )
            
            # Layer breakdown
            with gr.Accordion("Statistics by Layer", open=True):
                layer_stats_table = gr.Dataframe(
                    label="Layer Statistics",
                    interactive=False
                )
            
            # Raw results
            with gr.Accordion("Detailed Results", open=False):
                raw_results_table = gr.Dataframe(
                    label="All Image Results",
                    interactive=False
                )
        
        # ===== Export =====
        with gr.Group(visible=False) as export_group:
            gr.Markdown("### 💾 Export")
            
            export_btn = gr.Button("📥 Export JSON")
            export_file = gr.File(label="Download", visible=False)
        
        # State
        output_json_state = gr.State({})
        
        # ========== EVENT HANDLERS ==========
        
        def load_semantic(file):
            if not metrics_calculator:
                return "❌ MetricsCalculator not initialized"
            
            # Use uploaded or default
            if file:
                path = file.name
            else:
                path = str(Path(config.DATA_DIR) / 'Semantic_configuration.json')
            
            if not os.path.exists(path):
                return f"❌ File not found: {path}"
            
            if metrics_calculator.load_semantic_colors(path):
                return f"✅ Loaded {len(metrics_calculator.semantic_colors)} classes from {os.path.basename(path)}"
            return "❌ Failed to load"
        
        def toggle_input_mode(mode):
            is_folder = mode == "Folder Structure"
            return gr.update(visible=is_folder), gr.update(visible=is_folder)
        
        def refresh_calcs():
            if not metrics_manager:
                return gr.update(choices=[])
            
            metrics_manager.scan_calculators()
            choices = []
            for calc in metrics_manager.get_all_calculators():
                choices.append(f"{calc['id']}: {calc['name']}")
            
            return gr.update(choices=choices)
        
        def show_calc_info(selection):
            if not selection or not metrics_calculator:
                return gr.update(visible=False)
            
            ind_id = selection.split(":")[0].strip()
            info = metrics_calculator.get_calculator_info(ind_id)
            
            if info:
                return gr.update(value=info, visible=True)
            return gr.update(visible=False)
        
        def run_calculation(calc_sel, mode, folder_path, uploaded_files):
            """Run Stage 2.5 calculation"""
            try:
                if not calc_sel:
                    return ("❌ Select a calculator", "", "", "",
                            [], [], [], gr.update(visible=False), gr.update(visible=False), {})
                
                if not metrics_calculator:
                    return ("❌ MetricsCalculator not initialized", "", "", "",
                            [], [], [], gr.update(visible=False), gr.update(visible=False), {})
                
                # Load semantic colors if not loaded
                if not metrics_calculator.semantic_colors:
                    default_path = Path(config.DATA_DIR) / 'Semantic_configuration.json'
                    if default_path.exists():
                        metrics_calculator.load_semantic_colors(str(default_path))
                
                ind_id = calc_sel.split(":")[0].strip()
                module = metrics_calculator.load_calculator_module(ind_id)
                
                if not module:
                    return (f"❌ Could not load calculator: {ind_id}", "", "", "",
                            [], [], [], gr.update(visible=False), gr.update(visible=False), {})
                
                indicator = module.INDICATOR
                calc_func = module.calculate_indicator
                
                # Get zones from app state or create default
                query_zones = []
                for z in app_state.project_query.spatial_zones:
                    query_zones.append({
                        'id': z.zone_id,
                        'name': z.zone_name,
                        'area_sqm': 0,
                        'status': 'active'
                    })
                
                # If no zones defined, create a default one
                if not query_zones:
                    query_zones = [{'id': 'zone_1', 'name': 'Default Zone', 'area_sqm': 0, 'status': 'active'}]
                
                all_zone_results = []
                all_values = []
                all_values_by_layer = {layer: [] for layer in LAYERS}
                all_raw_results = []
                
                if mode == "Folder Structure" and folder_path:
                    # Folder structure
                    for zone in query_zones:
                        zone_images = scan_zone_images(folder_path, zone['id'], LAYERS)
                        total_zone_images = sum(len(f) for f in zone_images.values())
                        
                        if total_zone_images == 0:
                            continue
                        
                        result = process_zone(zone, zone_images, calc_func, indicator)
                        all_zone_results.append(result)
                        all_values.extend(result['all_values'])
                        
                        for layer in LAYERS:
                            all_values_by_layer[layer].extend(result['values_by_layer'].get(layer, []))
                        
                        # Collect raw results
                        for layer, layer_data in result['layers'].items():
                            for img_data in layer_data.get('images', []):
                                all_raw_results.append({
                                    'Zone': zone['name'],
                                    'Layer': layer,
                                    'Image': img_data.get('filename', ''),
                                    'Value': img_data.get('value'),
                                    'Unit': indicator.get('unit', '')
                                })
                
                else:
                    # Simple upload mode - treat as single zone, full layer
                    if not uploaded_files:
                        return ("❌ Upload images or specify folder", "", "", "",
                                [], [], [], gr.update(visible=False), gr.update(visible=False), {})
                    
                    zone = query_zones[0]
                    zone_images = {'full': [f.name for f in uploaded_files]}
                    
                    # Also add empty lists for other layers
                    for layer in LAYERS:
                        if layer not in zone_images:
                            zone_images[layer] = []
                    
                    result = process_zone(zone, zone_images, calc_func, indicator)
                    all_zone_results.append(result)
                    all_values.extend(result['all_values'])
                    
                    for layer in LAYERS:
                        all_values_by_layer[layer].extend(result['values_by_layer'].get(layer, []))
                    
                    # Collect raw results
                    for layer, layer_data in result['layers'].items():
                        for img_data in layer_data.get('images', []):
                            all_raw_results.append({
                                'Zone': zone['name'],
                                'Layer': layer,
                                'Image': img_data.get('filename', ''),
                                'Value': img_data.get('value'),
                                'Unit': indicator.get('unit', '')
                            })
                
                if not all_values:
                    return ("❌ No values computed", "", "", "",
                            [], [], [], gr.update(visible=False), gr.update(visible=False), {})
                
                # Build output JSON
                output = build_output_json(
                    indicator, all_zone_results, all_values, all_values_by_layer,
                    semantic_config_path=str(Path(config.DATA_DIR) / 'Semantic_configuration.json')
                )
                
                # Store in app state
                app_state.set_metrics_results(pd.DataFrame(all_raw_results))
                
                # Prepare display data
                overall_stats = output['descriptive_statistics_overall']
                zone_stats = output['zone_statistics']
                layer_stats = output['descriptive_statistics_by_layer']
                
                # Zone stats table
                zone_df = pd.DataFrame(zone_stats) if zone_stats else pd.DataFrame()
                
                # Layer stats table
                layer_data = []
                for layer, stats in layer_stats.items():
                    if stats.get('N', 0) > 0:
                        layer_data.append({
                            'Layer': layer,
                            'N': stats.get('N', 0),
                            'Mean': stats.get('Mean'),
                            'Std': stats.get('Std'),
                            'Min': stats.get('Min'),
                            'Max': stats.get('Max')
                        })
                layer_df = pd.DataFrame(layer_data) if layer_data else pd.DataFrame()
                
                # Raw results table
                raw_df = pd.DataFrame(all_raw_results) if all_raw_results else pd.DataFrame()
                
                return (
                    f"✅ Computed {indicator['id']}",
                    f"{indicator['id']}: {indicator['name']}",
                    str(overall_stats.get('N', 0)),
                    f"{overall_stats.get('Mean', 'N/A')} {indicator.get('unit', '')}",
                    zone_df,
                    layer_df,
                    raw_df,
                    gr.update(visible=True),
                    gr.update(visible=True),
                    output
                )
                
            except Exception as e:
                logger.error(f"Calculation error: {e}", exc_info=True)
                return (f"❌ Error: {e}", "", "", "",
                        [], [], [], gr.update(visible=False), gr.update(visible=False), {})
        
        def export_json(output_data):
            if not output_data:
                return gr.update(visible=False)
            
            try:
                output_dir = Path(config.OUTPUT_DIR)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                ind_id = output_data.get('indicator_definition', {}).get('id', 'IND')
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                # Save timestamped and latest versions (like output_layer.py)
                for filename in [f"{ind_id}_{timestamp}.json", f"{ind_id}_latest.json"]:
                    filepath = output_dir / filename
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(output_data, f, ensure_ascii=False, indent=2, default=str)
                
                return gr.update(value=str(output_dir / f"{ind_id}_{timestamp}.json"), visible=True)
            except Exception as e:
                logger.error(f"Export error: {e}")
                return gr.update(visible=False)
        
        # ===== BIND EVENTS =====
        load_config_btn.click(load_semantic, [semantic_file], [config_status])
        input_mode.change(toggle_input_mode, [input_mode], [mask_folder_input, scan_btn])
        refresh_btn.click(refresh_calcs, outputs=[calculator_dropdown])
        calculator_dropdown.change(show_calc_info, [calculator_dropdown], [calc_details])
        
        calculate_btn.click(
            run_calculation,
            [calculator_dropdown, input_mode, mask_folder_input, mask_upload],
            [progress_text, result_indicator, result_total, result_mean,
             zone_stats_table, layer_stats_table, raw_results_table,
             results_group, export_group, output_json_state]
        )
        
        export_btn.click(export_json, [output_json_state], [export_file])
        
        return {'zone_stats_table': zone_stats_table}

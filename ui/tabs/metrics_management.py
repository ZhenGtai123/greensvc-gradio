"""
Tab 3: Metrics Library Management
Uses MetricsManager from modules for business logic
"""

import gradio as gr
import pandas as pd
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def create_metrics_management_tab(components: dict, app_state, config):
    """Create Metrics Library Management Tab"""
    
    # Get MetricsManager from components
    metrics_manager = components.get('metrics_manager')
    
    with gr.Tab("3. Metrics Library"):
        gr.Markdown("""
        ## 📚 Metrics Library Management
        Upload and manage indicator calculators (calculator_layer_IND_XXX.py format)
        """)
        
        # ===== Upload Calculator =====
        with gr.Group():
            gr.Markdown("### 📤 Upload Calculator")
            gr.Markdown("""
            Upload calculator files:
            - Filename: `calculator_layer_IND_XXX.py`
            - Must contain: `INDICATOR` dict and `calculate_indicator(image_path)` function
            """)
            
            with gr.Row():
                calc_file = gr.File(
                    label="Calculator File (.py)",
                    file_types=[".py"]
                )
                upload_btn = gr.Button("Upload", variant="primary")
            
            upload_status = gr.Textbox(label="Status", interactive=False)
        
        # ===== Calculator Library =====
        with gr.Group():
            gr.Markdown("### 📋 Installed Calculators")
            
            refresh_btn = gr.Button("🔄 Refresh")
            
            calc_table = gr.Dataframe(
                headers=["ID", "Name", "Unit", "Type", "Direction", "Category"],
                label="Available Calculators",
                interactive=False,
                wrap=True
            )
        
        # ===== Calculator Details =====
        with gr.Accordion("Calculator Details", open=False):
            select_calc = gr.Dropdown(label="Select Calculator", choices=[])
            
            with gr.Row():
                calc_id = gr.Textbox(label="Indicator ID", interactive=False)
                calc_name = gr.Textbox(label="Name", interactive=False)
            
            with gr.Row():
                calc_unit = gr.Textbox(label="Unit", interactive=False)
                calc_type = gr.Textbox(label="Calc Type", interactive=False)
            
            calc_formula = gr.Textbox(label="Formula", lines=2, interactive=False)
            calc_definition = gr.Textbox(label="Definition", lines=2, interactive=False)
            
            with gr.Row():
                view_btn = gr.Button("View Code")
                delete_btn = gr.Button("Delete", variant="stop")
            
            code_display = gr.Code(label="Source Code", language="python", visible=False)
        
        # ===== Legacy Indicators Excel =====
        with gr.Accordion("Import from Excel (Legacy Format)", open=False):
            gr.Markdown("Import indicator definitions from A_indicators.xlsx")
            
            excel_file = gr.File(label="Indicators Excel (.xlsx)", file_types=[".xlsx"])
            import_btn = gr.Button("Import")
            import_status = gr.Textbox(label="Import Status", interactive=False)
        
        # ========== Event Handlers ==========
        
        def upload_calculator(file):
            if not file:
                return "Please select a file"
            
            if not metrics_manager:
                return "❌ MetricsManager not initialized"
            
            indicator_id = metrics_manager.add_calculator(file.name)
            if indicator_id:
                calc = metrics_manager.get_calculator(indicator_id)
                return f"✅ Uploaded: {calc.get('filename', '')}\n   ID: {indicator_id}\n   Name: {calc.get('name', '')}"
            return "❌ Upload failed. Check file format."
        
        def refresh_library():
            if not metrics_manager:
                return pd.DataFrame(), gr.update(choices=[])
            
            # Rescan calculators
            metrics_manager.scan_calculators()
            
            # Build table data
            data = []
            choices = []
            
            for calc in metrics_manager.get_all_calculators():
                data.append([
                    calc.get('id', ''),
                    calc.get('name', ''),
                    calc.get('unit', ''),
                    calc.get('calc_type', ''),
                    calc.get('target_direction', ''),
                    calc.get('category', '')
                ])
                choices.append(f"{calc.get('id', '')}: {calc.get('name', '')}")
            
            df = pd.DataFrame(data, columns=["ID", "Name", "Unit", "Type", "Direction", "Category"]) if data else pd.DataFrame()
            
            return df, gr.update(choices=choices)
        
        def show_calc_details(selection):
            if not selection or not metrics_manager:
                return "", "", "", "", "", "", gr.update(visible=False)
            
            indicator_id = selection.split(":")[0].strip()
            calc = metrics_manager.get_calculator(indicator_id)
            
            if not calc:
                return indicator_id, "Not found", "", "", "", "", gr.update(visible=False)
            
            return (
                calc.get('id', ''),
                calc.get('name', ''),
                calc.get('unit', ''),
                calc.get('calc_type', ''),
                calc.get('formula', ''),
                calc.get('definition', ''),
                gr.update(visible=False)
            )
        
        def view_code(selection):
            if not selection or not metrics_manager:
                return gr.update(visible=False)
            
            indicator_id = selection.split(":")[0].strip()
            code = metrics_manager.get_calculator_code(indicator_id)
            
            if code:
                return gr.update(value=code, visible=True)
            return gr.update(visible=False)
        
        def delete_calculator(selection):
            if not selection or not metrics_manager:
                return "Please select a calculator"
            
            indicator_id = selection.split(":")[0].strip()
            
            if metrics_manager.remove_calculator(indicator_id):
                return f"✅ Deleted: {indicator_id}"
            return "❌ Delete failed"
        
        def import_from_excel(file):
            if not file or not metrics_manager:
                return "Please select an Excel file"
            
            try:
                success = metrics_manager.import_metrics(file.name)
                if success:
                    return f"✅ Imported indicator definitions"
                return "❌ Import failed"
            except Exception as e:
                return f"❌ Import failed: {e}"
        
        # ===== Bind Events =====
        upload_btn.click(upload_calculator, [calc_file], [upload_status])
        refresh_btn.click(refresh_library, outputs=[calc_table, select_calc])
        select_calc.change(
            show_calc_details, 
            [select_calc], 
            [calc_id, calc_name, calc_unit, calc_type, calc_formula, calc_definition, code_display]
        )
        view_btn.click(view_code, [select_calc], [code_display])
        delete_btn.click(delete_calculator, [select_calc], [upload_status])
        import_btn.click(import_from_excel, [excel_file], [import_status])
        
        return {'calc_table': calc_table}

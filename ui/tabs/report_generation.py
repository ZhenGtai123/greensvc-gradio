"""
Tab 6: Report Generation
"""

import gradio as gr
import pandas as pd
import os
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


def create_report_generation_tab(components: dict, app_state, config):
    """Create Report Generation Tab"""
    
    with gr.Tab("6. Report Generation"):
        gr.Markdown("""
        ## 📄 Report Generation
        Generate analysis reports from calculation results
        """)
        
        # ===== Data Source =====
        with gr.Group():
            gr.Markdown("### 📊 Data Source")
            
            refresh_data_btn = gr.Button("Refresh Data")
            
            with gr.Row():
                data_status = gr.Textbox(label="Calculation Results", interactive=False)
                zones_status = gr.Textbox(label="Spatial Zones", interactive=False)
            
            preview_df = gr.Dataframe(
                label="Results Preview",
                interactive=False
            )
        
        # ===== Report Options =====
        with gr.Group():
            gr.Markdown("### ⚙️ Report Options")
            
            report_format = gr.Radio(
                label="Report Format",
                choices=["JSON", "CSV", "Excel"],
                value="JSON"
            )
            
            with gr.Row():
                include_stats = gr.Checkbox(label="Include Statistics", value=True)
                include_metadata = gr.Checkbox(label="Include Project Metadata", value=True)
            
            with gr.Accordion("Advanced Options", open=False):
                group_by_zone = gr.Checkbox(label="Group Results by Zone", value=False)
                include_raw = gr.Checkbox(label="Include Raw Values", value=True)
        
        # ===== Generate =====
        generate_btn = gr.Button("📄 Generate Report", variant="primary", size="lg")
        gen_status = gr.Textbox(label="Status", interactive=False)
        
        # ===== Download =====
        with gr.Group(visible=False) as download_group:
            gr.Markdown("### 📥 Download Report")
            report_file = gr.File(label="Report File")
            
            report_summary = gr.Textbox(
                label="Report Summary",
                lines=5,
                interactive=False
            )
        
        # ========== Event Handlers ==========
        
        def refresh_data():
            results = app_state.get_metrics_results()
            zones = app_state.project_query.spatial_zones
            
            if results.empty:
                return "No calculation results", f"{len(zones)} zones defined", pd.DataFrame()
            
            return (
                f"✅ {len(results)} records",
                f"{len(zones)} zones defined",
                results.head(10)
            )
        
        def generate_report(fmt, stats, metadata, by_zone, raw):
            try:
                results = app_state.get_metrics_results()
                if results.empty:
                    return ("❌ No calculation results. Run Metrics Calculation first.",
                            gr.update(visible=False), None, "")
                
                # Build report data
                report_data = {
                    'metadata': {},
                    'results': [],
                    'statistics': {}
                }
                
                # Add metadata
                if metadata:
                    q = app_state.project_query
                    report_data['metadata'] = {
                        'generated_at': datetime.now().isoformat(),
                        'system': 'GreenSVC-AI v2.0',
                        'project': {
                            'name': q.project_name,
                            'location': q.project_location,
                            'scale': q.site_scale,
                            'phase': q.project_phase
                        },
                        'context': {
                            'koppen_zone': q.koppen_zone_id,
                            'space_type': q.space_type_id,
                            'country': q.country_id
                        },
                        'zones': [{'id': z.zone_id, 'name': z.zone_name} 
                                  for z in q.spatial_zones]
                    }
                
                # Add results
                if raw:
                    report_data['results'] = results.to_dict(orient='records')
                
                # Add statistics
                if stats:
                    for indicator in results['Indicator'].unique():
                        ind_data = results[results['Indicator'] == indicator]['Value']
                        report_data['statistics'][indicator] = {
                            'N': len(ind_data),
                            'Mean': round(float(ind_data.mean()), 3),
                            'Std': round(float(ind_data.std()), 3),
                            'Min': round(float(ind_data.min()), 3),
                            'Max': round(float(ind_data.max()), 3)
                        }
                
                # Generate output file
                output_dir = Path(config.OUTPUT_DIR)
                output_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                if fmt == "JSON":
                    import json
                    filepath = output_dir / f"report_{timestamp}.json"
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(report_data, f, ensure_ascii=False, indent=2)
                
                elif fmt == "CSV":
                    filepath = output_dir / f"report_{timestamp}.csv"
                    results.to_csv(filepath, index=False)
                
                elif fmt == "Excel":
                    filepath = output_dir / f"report_{timestamp}.xlsx"
                    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                        results.to_excel(writer, sheet_name='Results', index=False)
                        if stats:
                            stats_df = pd.DataFrame(report_data['statistics']).T
                            stats_df.to_excel(writer, sheet_name='Statistics')
                
                # Summary
                summary = f"""Report Generated Successfully
                
Format: {fmt}
File: {filepath.name}
Records: {len(results)}
Indicators: {results['Indicator'].nunique()}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
                
                return (
                    "✅ Report generated successfully",
                    gr.update(visible=True),
                    str(filepath),
                    summary
                )
                
            except Exception as e:
                logger.error(f"Report generation error: {e}", exc_info=True)
                return (f"❌ Error: {e}", gr.update(visible=False), None, "")
        
        # ===== Bind Events =====
        refresh_data_btn.click(
            refresh_data,
            outputs=[data_status, zones_status, preview_df]
        )
        
        generate_btn.click(
            generate_report,
            [report_format, include_stats, include_metadata, group_by_zone, include_raw],
            [gen_status, download_group, report_file, report_summary]
        )
        
        return {'report_file': report_file}

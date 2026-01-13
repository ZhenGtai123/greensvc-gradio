"""
Tab 6: Report Generation (Stage 3)
Based on GreenSVC_Stage3_Colab.ipynb - LLM Diagnosis + IOM Matching

Stage 3 Pipeline:
1. Load indicator results (from Stage 2.5)
2. LLM Diagnosis: Analyze zones and identify issues
3. IOM Matching: Match issues to Intervention-Operation-Measure recommendations
4. Generate design strategy report
"""

import gradio as gr
import pandas as pd
import json
import os
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

# Gemini API
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

logger = logging.getLogger(__name__)


# ========== Stage 3 Prompt Template ==========
DIAGNOSIS_PROMPT = """
# GreenSVC Stage 3: Zone Diagnosis

You are a landscape architect analyzing indicator results for urban green spaces.

## Indicator Results
```json
{indicator_data}
```

## Task
For each zone, analyze the indicator values and:
1. Identify zones with poor performance (values significantly below/above targets)
2. Determine the likely causes based on indicator patterns
3. Suggest which landscape interventions might improve the indicators

## Output Format (JSON)
```json
{{
  "zone_diagnoses": [
    {{
      "zone_id": "zone_1",
      "zone_name": "Main Plaza",
      "overall_assessment": "Needs Improvement / Satisfactory / Good / Excellent",
      "priority_issues": [
        {{
          "indicator_id": "IND_GVI",
          "indicator_name": "Green View Index",
          "current_value": 15.2,
          "assessment": "Below target",
          "likely_causes": ["Low tree coverage", "Hard surface dominance"],
          "suggested_interventions": ["Add street trees", "Install green walls"]
        }}
      ],
      "strengths": ["Good walkability", "Adequate seating"],
      "recommendations": ["Increase vegetation cover", "Add shade structures"]
    }}
  ],
  "cross_zone_patterns": [
    "All zones show low GVI values",
    "Thermal comfort varies significantly between zones"
  ],
  "priority_actions": [
    {{
      "action": "Increase tree planting",
      "zones_affected": ["zone_1", "zone_2"],
      "expected_improvement": ["GVI +20%", "Thermal comfort +15%"]
    }}
  ]
}}
```

Analyze the data and provide your diagnosis.
"""


IOM_MATCHING_PROMPT = """
# GreenSVC Stage 3: IOM Matching

Based on the zone diagnosis, match appropriate Intervention-Operation-Measures (IOMs).

## Zone Diagnosis
```json
{diagnosis}
```

## Available IOMs
```json
{iom_data}
```

## Task
For each priority issue in each zone:
1. Find matching IOMs that address the issue
2. Rank IOMs by relevance and feasibility
3. Provide implementation guidance

## Output Format (JSON)
```json
{{
  "iom_recommendations": [
    {{
      "zone_id": "zone_1",
      "zone_name": "Main Plaza",
      "issue": "Low Green View Index",
      "matched_ioms": [
        {{
          "iom_id": "IOM_001",
          "intervention": "Add vegetation",
          "operation": "Plant trees",
          "measure": "Install 10-15 trees per hectare",
          "relevance_score": 0.95,
          "implementation_notes": "Consider native species",
          "expected_indicator_change": "GVI +15-25%"
        }}
      ]
    }}
  ],
  "implementation_priority": [
    {{
      "rank": 1,
      "iom_id": "IOM_001",
      "zones": ["zone_1", "zone_2"],
      "urgency": "High",
      "cost_estimate": "Medium",
      "timeline": "1-2 seasons"
    }}
  ]
}}
```

Provide IOM recommendations.
"""


class Stage3Processor:
    """Stage 3 LLM Diagnosis and IOM Matching"""
    
    def __init__(self, api_key: str = None, model: str = "gemini-2.0-flash"):
        self.api_key = api_key
        self.model_name = model
        self.model = None
        
        if api_key and GEMINI_AVAILABLE:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model)
    
    def load_indicator_results(self, filepath: str) -> Dict:
        """Load indicator results from Stage 2.5 output"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def run_diagnosis(self, indicator_data: Dict) -> Dict:
        """Run LLM diagnosis on indicator results"""
        if not self.model:
            return self._fallback_diagnosis(indicator_data)
        
        try:
            prompt = DIAGNOSIS_PROMPT.format(
                indicator_data=json.dumps(indicator_data, indent=2, default=str)[:30000]
            )
            
            response = self.model.generate_content(prompt)
            return self._parse_json_response(response.text)
        except Exception as e:
            logger.error(f"LLM diagnosis failed: {e}")
            return self._fallback_diagnosis(indicator_data)
    
    def run_iom_matching(self, diagnosis: Dict, iom_data: List = None) -> Dict:
        """Match IOMs to diagnosed issues"""
        if not self.model:
            return self._fallback_iom_matching(diagnosis)
        
        try:
            prompt = IOM_MATCHING_PROMPT.format(
                diagnosis=json.dumps(diagnosis, indent=2, default=str),
                iom_data=json.dumps(iom_data[:50] if iom_data else [], indent=2)
            )
            
            response = self.model.generate_content(prompt)
            return self._parse_json_response(response.text)
        except Exception as e:
            logger.error(f"IOM matching failed: {e}")
            return self._fallback_iom_matching(diagnosis)
    
    def _fallback_diagnosis(self, indicator_data: Dict) -> Dict:
        """Simple rule-based diagnosis when LLM unavailable"""
        diagnoses = []
        
        zone_stats = indicator_data.get('zone_statistics', [])
        overall = indicator_data.get('descriptive_statistics_overall', {})
        indicator_id = indicator_data.get('indicator_definition', {}).get('id', 'Unknown')
        indicator_name = indicator_data.get('indicator_definition', {}).get('name', 'Unknown')
        target_dir = indicator_data.get('indicator_definition', {}).get('target_direction', 'INCREASE')
        overall_mean = overall.get('Mean', 0)
        overall_std = overall.get('Std', 1)
        
        for zone in zone_stats:
            zone_mean = zone.get('Mean_overall', 0)
            
            # Calculate z-score
            z_score = (zone_mean - overall_mean) / overall_std if overall_std > 0 else 0
            
            # Determine assessment
            if target_dir == 'INCREASE':
                if z_score < -1:
                    assessment = "Needs Improvement"
                elif z_score < 0:
                    assessment = "Below Average"
                elif z_score < 1:
                    assessment = "Good"
                else:
                    assessment = "Excellent"
            else:
                if z_score > 1:
                    assessment = "Needs Improvement"
                elif z_score > 0:
                    assessment = "Above Average (issue)"
                elif z_score > -1:
                    assessment = "Good"
                else:
                    assessment = "Excellent"
            
            diagnoses.append({
                'zone_id': zone.get('Area_ID', ''),
                'zone_name': zone.get('Zone', ''),
                'overall_assessment': assessment,
                'indicator_value': zone_mean,
                'z_score': round(z_score, 2),
                'priority_issues': [{
                    'indicator_id': indicator_id,
                    'indicator_name': indicator_name,
                    'current_value': zone_mean,
                    'assessment': assessment
                }] if assessment in ['Needs Improvement', 'Below Average', 'Above Average (issue)'] else []
            })
        
        return {
            'zone_diagnoses': diagnoses,
            'cross_zone_patterns': [],
            'priority_actions': [],
            '_method': 'fallback_rule_based'
        }
    
    def _fallback_iom_matching(self, diagnosis: Dict) -> Dict:
        """Simple IOM suggestions when LLM unavailable"""
        recommendations = []
        
        # Generic IOMs based on common indicators
        generic_ioms = {
            'IND_GVI': [
                {'intervention': 'Add vegetation', 'operation': 'Plant trees', 'measure': '10-15 trees/ha'},
                {'intervention': 'Add vegetation', 'operation': 'Install green walls', 'measure': '50-100 sqm'}
            ],
            'IND_SVF': [
                {'intervention': 'Modify enclosure', 'operation': 'Add canopy', 'measure': 'Target 40-60% SVF'},
            ],
            'IND_ASV': [
                {'intervention': 'Reduce hard surfaces', 'operation': 'Add permeable paving', 'measure': '30% reduction'},
            ]
        }
        
        for zone in diagnosis.get('zone_diagnoses', []):
            for issue in zone.get('priority_issues', []):
                ind_id = issue.get('indicator_id', '')
                ioms = generic_ioms.get(ind_id, [{'intervention': 'Consult specialist', 'operation': 'Site assessment', 'measure': 'TBD'}])
                
                recommendations.append({
                    'zone_id': zone.get('zone_id'),
                    'zone_name': zone.get('zone_name'),
                    'issue': f"Low {issue.get('indicator_name', 'performance')}",
                    'matched_ioms': ioms
                })
        
        return {
            'iom_recommendations': recommendations,
            'implementation_priority': [],
            '_method': 'fallback_generic'
        }
    
    def _parse_json_response(self, text: str) -> Dict:
        """Parse JSON from LLM response"""
        text = text.strip()
        
        # Remove markdown code blocks
        if text.startswith('```'):
            text = text.split('```')[1]
            if text.startswith('json'):
                text = text[4:]
        if text.endswith('```'):
            text = text[:-3]
        
        try:
            return json.loads(text.strip())
        except:
            return {'error': 'Failed to parse response', 'raw': text[:500]}


def create_report_generation_tab(components: dict, app_state, config):
    """Create Report Generation Tab (Stage 3)"""
    
    with gr.Tab("6. Report Generation"):
        gr.Markdown("""
        ## 📄 Stage 3: Diagnosis & Recommendations
        
        Generate design recommendations based on indicator analysis results.
        
        **Pipeline**: Load Results → LLM Diagnosis → IOM Matching → Report
        """)
        
        # ===== Data Source =====
        with gr.Group():
            gr.Markdown("### 📥 Load Indicator Results")
            
            with gr.Row():
                result_file = gr.File(label="Stage 2.5 Output (.json)", file_types=[".json"])
                load_btn = gr.Button("Load Results")
            
            # Or use from current session
            use_session_btn = gr.Button("Use Current Session Results")
            
            data_status = gr.Textbox(label="Data Status", interactive=False)
            
            with gr.Accordion("Preview Data", open=False):
                data_preview = gr.JSON(label="Loaded Data")
        
        # ===== AI Configuration =====
        with gr.Accordion("⚙️ AI Configuration (Optional)", open=False):
            gr.Markdown("Configure Gemini API for intelligent diagnosis. Without API, uses rule-based fallback.")
            
            with gr.Row():
                api_key = gr.Textbox(label="Google API Key", type="password", scale=2)
                model_select = gr.Dropdown(
                    label="Model",
                    choices=["gemini-2.0-flash", "gemini-1.5-pro"],
                    value="gemini-2.0-flash",
                    scale=1
                )
            
            if hasattr(config, 'GOOGLE_API_KEY') and config.GOOGLE_API_KEY:
                gr.Markdown("*✅ API Key available from .env*")
        
        # ===== Generate =====
        generate_btn = gr.Button("🚀 Generate Diagnosis & Recommendations", variant="primary", size="lg")
        gen_status = gr.Textbox(label="Status", interactive=False)
        
        # ===== Results =====
        with gr.Group(visible=False) as results_group:
            gr.Markdown("### 📊 Diagnosis Results")
            
            # Zone assessments
            zone_assessment_table = gr.Dataframe(
                label="Zone Assessments",
                interactive=False
            )
            
            # Recommendations
            gr.Markdown("### 💡 Recommendations")
            recommendations_table = gr.Dataframe(
                label="IOM Recommendations",
                interactive=False
            )
            
            # Full report
            with gr.Accordion("Full Report (JSON)", open=False):
                full_report = gr.JSON(label="Complete Report")
        
        # ===== Export =====
        with gr.Group(visible=False) as export_group:
            gr.Markdown("### 💾 Export Report")
            
            with gr.Row():
                export_json_btn = gr.Button("Export JSON")
                export_md_btn = gr.Button("Export Markdown")
            
            export_file = gr.File(label="Download", visible=False)
        
        # State
        loaded_data = gr.State({})
        report_data = gr.State({})
        
        # ========== EVENT HANDLERS ==========
        
        def load_results_file(file):
            if not file:
                return {}, "No file selected", {}
            
            try:
                with open(file.name, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                ind_id = data.get('indicator_definition', {}).get('id', 'Unknown')
                total = data.get('computation_summary', {}).get('total_images_analyzed', 0)
                
                return data, f"✅ Loaded: {ind_id} ({total} images)", data
            except Exception as e:
                return {}, f"❌ Error: {e}", {}
        
        def use_session_data():
            results = app_state.get_metrics_results()
            
            if results.empty:
                return {}, "No results in current session", {}
            
            # Convert to Stage 2.5 format
            data = {
                'indicator_definition': {'id': 'Session', 'name': 'Current Session'},
                'zone_statistics': results.to_dict(orient='records'),
                'descriptive_statistics_overall': {
                    'N': len(results),
                    'Mean': results['Value'].mean() if 'Value' in results.columns else 0
                }
            }
            
            return data, f"✅ Using session data ({len(results)} records)", data
        
        def generate_report(data, api_key_val, model):
            if not data:
                return ("❌ Load data first", [], [], {}, 
                        gr.update(visible=False), gr.update(visible=False), {})
            
            try:
                # Use API key from input or config
                key = api_key_val or (config.GOOGLE_API_KEY if hasattr(config, 'GOOGLE_API_KEY') else None)
                
                processor = Stage3Processor(api_key=key, model=model)
                
                # Run diagnosis
                diagnosis = processor.run_diagnosis(data)
                
                # Run IOM matching
                iom_results = processor.run_iom_matching(diagnosis)
                
                # Combine into report
                report = {
                    'metadata': {
                        'generated_at': datetime.now().isoformat(),
                        'system': 'GreenSVC-AI Stage 3',
                        'model': model if key else 'rule_based_fallback'
                    },
                    'input_summary': {
                        'indicator': data.get('indicator_definition', {}).get('id'),
                        'zones': len(data.get('zone_statistics', []))
                    },
                    'diagnosis': diagnosis,
                    'iom_recommendations': iom_results
                }
                
                # Prepare display tables
                zone_data = []
                for zd in diagnosis.get('zone_diagnoses', []):
                    zone_data.append({
                        'Zone': zd.get('zone_name', ''),
                        'Assessment': zd.get('overall_assessment', ''),
                        'Value': zd.get('indicator_value', ''),
                        'Z-Score': zd.get('z_score', ''),
                        'Issues': len(zd.get('priority_issues', []))
                    })
                zone_df = pd.DataFrame(zone_data) if zone_data else pd.DataFrame()
                
                rec_data = []
                for rec in iom_results.get('iom_recommendations', []):
                    for iom in rec.get('matched_ioms', []):
                        rec_data.append({
                            'Zone': rec.get('zone_name', ''),
                            'Issue': rec.get('issue', ''),
                            'Intervention': iom.get('intervention', ''),
                            'Operation': iom.get('operation', ''),
                            'Measure': iom.get('measure', '')
                        })
                rec_df = pd.DataFrame(rec_data) if rec_data else pd.DataFrame()
                
                method = diagnosis.get('_method', 'llm')
                status = f"✅ Report generated ({method})"
                
                return (status, zone_df, rec_df, report,
                        gr.update(visible=True), gr.update(visible=True), report)
                
            except Exception as e:
                logger.error(f"Report generation error: {e}", exc_info=True)
                return (f"❌ Error: {e}", [], [], {},
                        gr.update(visible=False), gr.update(visible=False), {})
        
        def export_json_report(report):
            if not report:
                return gr.update(visible=False)
            
            try:
                output_dir = Path(config.OUTPUT_DIR)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filepath = output_dir / f"stage3_report_{timestamp}.json"
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(report, f, ensure_ascii=False, indent=2, default=str)
                
                return gr.update(value=str(filepath), visible=True)
            except Exception as e:
                logger.error(f"Export error: {e}")
                return gr.update(visible=False)
        
        def export_markdown_report(report):
            if not report:
                return gr.update(visible=False)
            
            try:
                output_dir = Path(config.OUTPUT_DIR)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filepath = output_dir / f"stage3_report_{timestamp}.md"
                
                # Generate markdown
                md = f"""# GreenSVC Stage 3 Report
Generated: {report['metadata']['generated_at']}

## Summary
- Indicator: {report['input_summary']['indicator']}
- Zones Analyzed: {report['input_summary']['zones']}

## Zone Diagnoses
"""
                for zd in report['diagnosis'].get('zone_diagnoses', []):
                    md += f"\n### {zd.get('zone_name', 'Zone')}\n"
                    md += f"- **Assessment**: {zd.get('overall_assessment', 'N/A')}\n"
                    md += f"- **Value**: {zd.get('indicator_value', 'N/A')}\n"
                    if zd.get('priority_issues'):
                        md += f"- **Issues**: {len(zd['priority_issues'])}\n"

                md += "\n## Recommendations\n"
                for rec in report['iom_recommendations'].get('iom_recommendations', []):
                    md += f"\n### {rec.get('zone_name', 'Zone')}: {rec.get('issue', '')}\n"
                    for iom in rec.get('matched_ioms', []):
                        md += f"- {iom.get('intervention', '')} → {iom.get('operation', '')} ({iom.get('measure', '')})\n"
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(md)
                
                return gr.update(value=str(filepath), visible=True)
            except Exception as e:
                logger.error(f"Export error: {e}")
                return gr.update(visible=False)
        
        # ===== BIND EVENTS =====
        load_btn.click(load_results_file, [result_file], [loaded_data, data_status, data_preview])
        use_session_btn.click(use_session_data, outputs=[loaded_data, data_status, data_preview])
        
        generate_btn.click(
            generate_report,
            [loaded_data, api_key, model_select],
            [gen_status, zone_assessment_table, recommendations_table, full_report,
             results_group, export_group, report_data]
        )
        
        export_json_btn.click(export_json_report, [report_data], [export_file])
        export_md_btn.click(export_markdown_report, [report_data], [export_file])
        
        return {'full_report': full_report}

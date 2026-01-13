"""
Tab 2: Indicator Recommendation (Evidence-Based Indicator Matching)
Based on GreenSVC_Stage1_Colab.ipynb - Using Gemini API
"""

import gradio as gr
import pandas as pd
import json
import logging
import os
from typing import Dict, List, Optional
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Gemini API
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

logger = logging.getLogger(__name__)


# ========== Stage 1 Prompt Template ==========
PROMPT_TEMPLATE = """
# GreenSVC-AI Stage 1: Evidence-Based Indicator Matching

## System Role
You are an evidence-based landscape design consultant. Your task is to recommend environmental indicators for landscape projects based on scientific evidence.

## Core Principles
1. **Evidence-Based**: Every recommendation must cite evidence_id
2. **Code Expansion**: Use Codebook to expand all codes to code + name + definition
3. **No Fabrication**: Only use evidence that exists in the input

---

## Input Data

### User Query
```json
{query}
```

### Codebook (Code Definitions)
```json
{codebook}
```

### Evidence ({evd_count} records)
```json
{evidence}
```

---

## Output Format

Output a JSON object with this structure:

```json
{{
  "metadata": {{
    "project_name": "from query.project.name",
    "target_dimensions": [{{"code": "PRF_BEH", "name": "...", "definition": "..."}}],
    "target_subdimensions": [{{"code": "PRS_WLK", "name": "...", "definition": "..."}}],
    "evidence_used": 10
  }},

  "recommended_indicators": [
    {{
      "rank": 1,
      "indicator": {{
        "code": "IND_XXX",
        "name": "from Codebook",
        "definition": "from Codebook",
        "formula": "from Evidence",
        "category": {{"code": "CAT_XXX", "name": "...", "definition": "..."}}
      }},
      "performance": {{
        "dimension": {{"code": "PRF_BEH", "name": "...", "definition": "..."}},
        "subdimension": {{"code": "PRS_WLK", "name": "...", "definition": "..."}},
        "outcome_measure": "from Evidence",
        "outcome_type": {{"code": "OUT_XXX", "name": "...", "definition": "..."}}
      }},
      "evidence": [
        {{
          "evidence_id": "EVD_xxx",
          "citation": "Author (Year). Title.",
          "year": 2024,
          "relationship": {{
            "direction": {{"code": "DIR_POS", "name": "Positive"}},
            "effect_size": "0.45",
            "p_value": "<0.001"
          }},
          "study": {{
            "design": {{"code": "DES_CRS", "name": "..."}},
            "sample_size": 100,
            "setting": {{"code": "SET_PRK", "name": "..."}},
            "country": {{"code": "CNT_NLD", "name": "..."}}
          }},
          "quality": {{
            "tier": {{"code": "TIR_T2", "name": "..."}},
            "confidence": {{"code": "CON_HIG", "name": "..."}}
          }}
        }}
      ],
      "measurement": {{
        "method": {{"code": "MTH_XXX", "name": "..."}},
        "unit": {{"code": "UNT_XXX", "name": "..."}},
        "data_source": {{"code": "SRC_XXX", "name": "..."}}
      }},
      "target_direction": {{
        "direction": "INCREASE or DECREASE",
        "derivation": "Based on evidence direction and performance goal"
      }},
      "rationale": "Why this indicator is recommended"
    }}
  ],

  "indicator_relationships": [
    {{
      "indicators": [{{"code": "IND_A", "name": "..."}}, {{"code": "IND_B", "name": "..."}}],
      "type": "SYNERGISTIC/INVERSE/INDEPENDENT",
      "explanation": "How they interact"
    }}
  ],

  "summary": {{
    "total_indicators": 5,
    "total_evidence": 12,
    "key_findings": ["Finding 1", "Finding 2"],
    "evidence_gaps": ["Gap 1"]
  }}
}}
```

## Rules
1. Recommend 5-8 indicators relevant to target dimensions/subdimensions
2. Each indicator MUST have evidence_id from the provided Evidence
3. Expand ALL codes using Codebook
4. Do NOT output numerical target values
5. Output valid JSON directly, no markdown code blocks
"""


class KnowledgeBase:
    """Evidence Knowledge Base"""
    
    def __init__(self, evidence_path: str = None):
        self.evidence = []
        self.perf_idx = defaultdict(list)
        self.subdim_idx = defaultdict(list)
        self.indicator_idx = defaultdict(list)
        
        if evidence_path and Path(evidence_path).exists():
            self.load(evidence_path)
    
    def load(self, path: str) -> bool:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                self.evidence = json.load(f)
            self._build_index()
            logger.info(f"📚 Evidence loaded: {len(self.evidence)} records")
            return True
        except Exception as e:
            logger.error(f"Failed to load evidence: {e}")
            return False
    
    def _build_index(self):
        self.perf_idx.clear()
        self.subdim_idx.clear()
        self.indicator_idx.clear()
        
        for e in self.evidence:
            perf = e.get('performance', {})
            dim = perf.get('dimension_id')
            subdim = perf.get('subdimension_id')
            
            if dim:
                self.perf_idx[dim].append(e)
            if subdim and subdim != 'PRS_NA':
                self.subdim_idx[subdim].append(e)
            
            indicator = e.get('indicator', {})
            ind_id = indicator.get('indicator_id')
            if ind_id:
                self.indicator_idx[ind_id].append(e)
    
    def retrieve(self, dimensions: List[str], subdimensions: List[str] = None) -> List[Dict]:
        """Retrieve relevant Evidence"""
        evds = []
        
        for d in dimensions:
            for e in self.perf_idx.get(d, []):
                if e not in evds:
                    evds.append(e)
        
        if subdimensions:
            for sd in subdimensions:
                for e in self.subdim_idx.get(sd, []):
                    if e not in evds:
                        evds.append(e)
        
        logger.info(f"🔍 Retrieved: {len(evds)} Evidence records")
        return evds
    
    def get_summary(self) -> Dict:
        return {
            "total_evidence": len(self.evidence),
            "dimensions": len(self.perf_idx),
            "subdimensions": len(self.subdim_idx),
            "indicators": len(self.indicator_idx)
        }


class Codebook:
    """Appendix Codebook"""
    
    def __init__(self, path: str = None):
        self.data = {}
        if path and Path(path).exists():
            self.load(path)
    
    def load(self, path: str) -> bool:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            logger.info(f"📖 Codebook loaded: {len(self.data)} tables")
            return True
        except Exception as e:
            logger.error(f"Failed to load codebook: {e}")
            return False
    
    def subset(self, max_chars: int = 40000) -> Dict:
        """Extract subset for prompt"""
        priority = [
            'A_indicators', 'A_categories',
            'C_performance', 'C_subdimensions', 'C_outcome_types',
            'B_methods', 'B_units', 'B_data_sources',
            'D_directions', 'D_significance',
            'E_settings', 'E_countries',
            'K_climate', 'F_quality'
        ]
        
        out, sz = {}, 0
        for n in priority:
            if n in self.data:
                simplified = {}
                for code, entry in self.data[n].items():
                    simplified[code] = {
                        "name": entry.get('name', code),
                        "definition": entry.get('definition', '')[:200]
                    }
                s = len(json.dumps(simplified, ensure_ascii=False))
                if sz + s < max_chars:
                    out[n] = simplified
                    sz += s
        return out


class GeminiRunner:
    """Gemini AI Model Runner (Stage 1 Notebook compatible)"""
    
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        if not GEMINI_AVAILABLE:
            raise ImportError("Please install google-generativeai: pip install google-generativeai")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.model_name = model
        logger.info(f"✅ Gemini Model: {model}")
    
    def run(self, prompt: str) -> Dict:
        """Call Gemini API and parse result"""
        try:
            logger.info(f"🚀 Calling Gemini API (~{len(prompt)//4:,} tokens)")
            
            config = genai.GenerationConfig(
                temperature=0.2,
                max_output_tokens=32768
            )
            
            response = self.model.generate_content(prompt, generation_config=config)
            text = response.text
            
            logger.info(f"✅ Response: {len(text):,} chars")
            
            if not text.rstrip().endswith('}'):
                logger.warning("⚠️ Response may be truncated")
            
            return self._parse(text)
            
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            return {"error": str(e)}
    
    def _parse(self, text: str) -> Dict:
        """Parse JSON response"""
        text = text.strip()
        
        # Clean markdown code blocks
        for p in ['```json', '```']:
            if text.startswith(p):
                text = text[len(p):]
        if text.endswith('```'):
            text = text[:-3]
        text = text.strip()
        
        try:
            return json.loads(text)
        except:
            pass
        
        # Try to repair truncated JSON
        if not text.endswith('}'):
            r = text.rstrip().rstrip(',: \n\t"\'')
            r += ']' * (text.count('[') - text.count(']'))
            r += '}' * (text.count('{') - text.count('}'))
            try:
                res = json.loads(r)
                res['_warning'] = 'Auto-repaired JSON'
                return res
            except:
                pass
        
        return {'raw': text, 'error': 'JSON parse failed'}


def build_prompt(query: Dict, evidence: List[Dict], codebook: Dict) -> str:
    """Build recommendation prompt"""
    return PROMPT_TEMPLATE.format(
        query=json.dumps(query, ensure_ascii=False, indent=2),
        codebook=json.dumps(codebook, ensure_ascii=False, indent=2),
        evidence=json.dumps(evidence[:60], ensure_ascii=False, indent=2),
        evd_count=len(evidence)
    )


def create_indicator_recommendation_tab(components: dict, app_state, config):
    """Create Indicator Recommendation Tab"""
    
    # Auto-load knowledge base
    kb = KnowledgeBase()
    codebook = Codebook()
    
    kb_dir = Path(config.DATA_DIR) / 'knowledge_base'
    evidence_path = kb_dir / 'Evidence_final_v5_2_fixed.json'
    appendix_path = kb_dir / 'Appendix_final_v5_2_fixed.json'
    
    auto_loaded = False
    if evidence_path.exists() and appendix_path.exists():
        kb.load(str(evidence_path))
        codebook.load(str(appendix_path))
        auto_loaded = True
    
    with gr.Tab("2. Indicator Recommendation"):
        gr.Markdown("""
        ## 🌿 Evidence-Based Indicator Matching
        Recommend SVC indicators based on research evidence matching project questionnaire
        """)
        
        # ===== Knowledge Base Status =====
        with gr.Group():
            gr.Markdown("### 📚 Knowledge Base")
            
            if auto_loaded:
                kb_info = gr.Markdown(f"""
                ✅ **Knowledge base auto-loaded**
                - Evidence: {kb.get_summary()['total_evidence']} records, {kb.get_summary()['indicators']} indicators
                - Codebook: {len(codebook.data)} tables
                """)
            else:
                kb_info = gr.Markdown("⚠️ Knowledge base not found, please upload manually")
            
            with gr.Accordion("Manual Upload", open=not auto_loaded):
                with gr.Row():
                    evidence_file = gr.File(label="Evidence (.json)", file_types=[".json"])
                    codebook_file = gr.File(label="Codebook (.json)", file_types=[".json"])
                load_btn = gr.Button("Load", variant="secondary")
                load_status = gr.Textbox(label="Status", interactive=False)
        
        # ===== Query Preview =====
        with gr.Accordion("📋 Project Query Preview", open=False):
            refresh_btn = gr.Button("Refresh", size="sm")
            with gr.Row():
                dims_display = gr.Textbox(label="Performance Dimensions", interactive=False)
                zones_display = gr.Textbox(label="Spatial Zones", interactive=False)
                imgs_display = gr.Textbox(label="Images", interactive=False)
        
        # ===== AI Configuration =====
        with gr.Accordion("⚙️ AI Configuration", open=False):
            with gr.Row():
                api_key_input = gr.Textbox(
                    label="Google API Key",
                    type="password",
                    value=config.GOOGLE_API_KEY if hasattr(config, 'GOOGLE_API_KEY') else "",
                    placeholder="AIza..."
                )
                model_select = gr.Dropdown(
                    label="Gemini Model",
                    choices=["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
                    value="gemini-2.0-flash"
                )
            
            if hasattr(config, 'GOOGLE_API_KEY') and config.GOOGLE_API_KEY:
                gr.Markdown("*✅ Google API Key loaded from .env*")
            else:
                gr.Markdown("*Get API Key from: https://aistudio.google.com/apikey*")
        
        # ===== Recommend Button =====
        recommend_btn = gr.Button("🚀 Run Recommendation", variant="primary", size="lg")
        status_text = gr.Textbox(label="Status", interactive=False)
        
        # ===== Results =====
        with gr.Group(visible=False) as results_group:
            gr.Markdown("### 📊 Recommendation Results")
            
            with gr.Row():
                total_ind = gr.Textbox(label="Recommended Indicators", interactive=False)
                total_evd = gr.Textbox(label="Evidence Used", interactive=False)
            
            indicators_df = gr.Dataframe(
                label="Recommended Indicators",
                headers=["Rank", "Code", "Name", "Category", "Direction", "Evidence", "Rationale"],
                interactive=False,
                wrap=True
            )
            
            with gr.Accordion("Full JSON Result", open=False):
                full_result = gr.JSON(label="JSON")
            
            with gr.Accordion("Key Findings & Evidence Gaps", open=False):
                findings_text = gr.Textbox(label="Key Findings", lines=3, interactive=False)
                gaps_text = gr.Textbox(label="Evidence Gaps", lines=2, interactive=False)
        
        # ===== Indicator Selection =====
        with gr.Group(visible=False) as selection_group:
            gr.Markdown("### ✅ Select Indicators to Use")
            indicator_checks = gr.CheckboxGroup(label="Check indicators", choices=[])
            confirm_btn = gr.Button("Confirm Selection", variant="primary")
            select_status = gr.Textbox(label="Status", interactive=False)
        
        # ========== Event Handlers ==========
        
        def load_knowledge_base(evd_file, cb_file):
            msgs = []
            
            if evd_file:
                if kb.load(evd_file.name):
                    s = kb.get_summary()
                    msgs.append(f"✅ Evidence: {s['total_evidence']} records, {s['indicators']} indicators")
                else:
                    msgs.append("❌ Evidence load failed")
            
            if cb_file:
                if codebook.load(cb_file.name):
                    msgs.append(f"✅ Codebook: {len(codebook.data)} tables")
                else:
                    msgs.append("❌ Codebook load failed")
            
            return "\n".join(msgs) if msgs else "Please select files"
        
        def refresh_query():
            q = app_state.project_query
            dims = ", ".join(q.performance_dimensions) if q.performance_dimensions else "Not selected"
            zones = ", ".join([z.zone_name for z in q.spatial_zones]) if q.spatial_zones else "Not defined"
            imgs = str(len(q.uploaded_images))
            return dims, zones, imgs
        
        def run_recommendation(api_key, model):
            try:
                if not kb.evidence:
                    return ("❌ Please load Evidence knowledge base first", "", "", [], {}, "", "",
                            gr.update(visible=False), gr.update(visible=False), gr.update(choices=[]))
                
                q = app_state.project_query
                if not q.performance_dimensions:
                    return ("❌ Please select performance dimensions in questionnaire first", "", "", [], {}, "", "",
                            gr.update(visible=False), gr.update(visible=False), gr.update(choices=[]))
                
                if not api_key:
                    return ("❌ Please enter Google API Key", "", "", [], {}, "", "",
                            gr.update(visible=False), gr.update(visible=False), gr.update(choices=[]))
                
                if not GEMINI_AVAILABLE:
                    return ("❌ Please install google-generativeai: pip install google-generativeai", "", "", [], {}, "", "",
                            gr.update(visible=False), gr.update(visible=False), gr.update(choices=[]))
                
                evidence = kb.retrieve(q.performance_dimensions, q.subdimensions)
                if not evidence:
                    return ("⚠️ No matching evidence found", "", "", [], {}, "", "",
                            gr.update(visible=False), gr.update(visible=False), gr.update(choices=[]))
                
                cb_subset = codebook.subset() if codebook.data else {}
                prompt = build_prompt(q.to_dict(), evidence, cb_subset)
                
                runner = GeminiRunner(api_key, model)
                result = runner.run(prompt)
                
                if 'error' in result and 'raw' not in result:
                    return (f"❌ API Error: {result['error']}", "", "", [], {}, "", "",
                            gr.update(visible=False), gr.update(visible=False), gr.update(choices=[]))
                
                indicators = result.get('recommended_indicators', [])
                summary = result.get('summary', {})
                
                app_state.set_recommended_indicators(indicators)
                
                table_data = []
                choices = []
                for ind in indicators:
                    ind_info = ind.get('indicator', {})
                    code = ind_info.get('code', '')
                    name = ind_info.get('name', '')
                    rationale = ind.get('rationale', '')[:80] + '...' if len(ind.get('rationale', '')) > 80 else ind.get('rationale', '')
                    
                    table_data.append([
                        ind.get('rank', ''),
                        code,
                        name,
                        ind_info.get('category', {}).get('name', ''),
                        ind.get('target_direction', {}).get('direction', ''),
                        len(ind.get('evidence', [])),
                        rationale
                    ])
                    choices.append(f"{code}: {name}")
                
                findings = "\n".join([f"• {f}" for f in summary.get('key_findings', [])])
                gaps = "\n".join([f"• {g}" for g in summary.get('evidence_gaps', [])])
                
                warning = f" ⚠️ {result['_warning']}" if '_warning' in result else ""
                
                return (
                    f"✅ Recommendation complete! {len(indicators)} indicators{warning}",
                    str(len(indicators)),
                    str(summary.get('total_evidence', len(evidence))),
                    table_data,
                    result,
                    findings,
                    gaps,
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(choices=choices, value=[])
                )
                
            except Exception as e:
                logger.error(f"Recommendation error: {e}", exc_info=True)
                return (f"❌ Error: {e}", "", "", [], {}, "", "",
                        gr.update(visible=False), gr.update(visible=False), gr.update(choices=[]))
        
        def confirm_selection(selections):
            if not selections:
                return "❌ Please select at least one indicator"
            
            selected = []
            for sel in selections:
                code = sel.split(":")[0].strip()
                for ind in app_state.get_recommended_indicators():
                    if ind.get('indicator', {}).get('code') == code:
                        selected.append({
                            'metric name': ind['indicator'].get('name', ''),
                            'indicator_code': code,
                            'category': ind['indicator'].get('category', {}).get('name', ''),
                            'target_direction': ind.get('target_direction', {}).get('direction', ''),
                            'rationale': ind.get('rationale', '')
                        })
                        break
            
            app_state.set_selected_metrics(selected)
            return f"✅ Selected {len(selected)} indicators, ready for vision analysis"
        
        # ===== Bind Events =====
        load_btn.click(load_knowledge_base, [evidence_file, codebook_file], [load_status])
        refresh_btn.click(refresh_query, outputs=[dims_display, zones_display, imgs_display])
        
        recommend_btn.click(
            run_recommendation,
            [api_key_input, model_select],
            [status_text, total_ind, total_evd, indicators_df, full_result,
             findings_text, gaps_text, results_group, selection_group, indicator_checks]
        )
        
        confirm_btn.click(confirm_selection, [indicator_checks], [select_status])
        
        return {
            'evidence_file': evidence_file,
            'codebook_file': codebook_file,
            'indicators_df': indicators_df,
            'indicator_checks': indicator_checks
        }

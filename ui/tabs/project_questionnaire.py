"""
Tab 1: Project Questionnaire (Design Goal Definition)
"""

import gradio as gr
import os
import json
from typing import List, Tuple, Optional
from datetime import datetime
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import logging

logger = logging.getLogger(__name__)

# ========== Options Data ==========
OPTIONS = {
    "koppen": [
        ("", "Select *"),
        ("KPN_AF", "Tropical Rainforest (Af)"),
        ("KPN_CFA", "Humid Subtropical (Cfa)"),
        ("KPN_CFB", "Oceanic (Cfb)"),
        ("KPN_CWA", "Monsoon Subtropical (Cwa)"),
        ("KPN_DFA", "Hot-summer Continental (Dfa)"),
        ("KPN_DWA", "Monsoon Continental (Dwa)"),
    ],
    "country": [
        ("", "Select"),
        ("CNT_CHN", "China"), ("CNT_HKG", "Hong Kong"), ("CNT_JPN", "Japan"),
        ("CNT_KOR", "South Korea"), ("CNT_SGP", "Singapore"), ("CNT_USA", "United States"),
        ("CNT_GBR", "United Kingdom"), ("CNT_NLD", "Netherlands"),
    ],
    "space_type": [
        ("", "Select *"),
        ("SET_PRK", "Park"), ("SET_RES", "Residential"), ("SET_STR", "Street"),
        ("SET_CAM", "Campus"), ("SET_CBD", "Business District"), ("SET_GWY", "Greenway"),
        ("SET_POS", "Public Open Space"),
    ],
    "lcz": [
        ("", "Select"),
        ("LCZ_1", "Compact High-rise"), ("LCZ_2", "Compact Mid-rise"), ("LCZ_3", "Compact Low-rise"),
        ("LCZ_4", "Open High-rise"), ("LCZ_A", "Dense Trees"), ("LCZ_B", "Scattered Trees"),
    ],
    "age": [
        ("", "Select"),
        ("AGE_ALL", "All Ages"), ("AGE_ADL", "Adults"), ("AGE_ELD", "Elderly"),
    ],
    "scale": [
        ("", "Select"),
        ("XS", "<0.5 ha"), ("S", "0.5-2 ha"), ("M", "2-10 ha"),
        ("L", "10-50 ha"), ("XL", ">50 ha"),
    ],
    "phase": [
        ("", "Select"),
        ("conceptual", "Conceptual Design"), ("schematic", "Schematic Design"),
        ("detailed", "Detailed Design"), ("renovation", "Renovation"),
    ],
}

PERFORMANCE_DIMS = [
    ("PRF_AES", "👁️ Visual Spatial Quality", "Aesthetic quality and spatial legibility"),
    ("PRF_BEH", "🚶 Use & Safety", "Usage patterns, accessibility, safety"),
    ("PRF_COM", "😌 Comfort", "Thermal comfort and overall comfort"),
    ("PRF_ENV", "🌿 Environmental Quality", "Air quality and environmental indicators"),
    ("PRF_HLT", "🧠 Health & Wellbeing", "Physical and mental health, stress relief"),
    ("PRF_SOC", "🤝 Social Outcomes", "Social interaction and equity"),
]

SUBDIMS = {
    "PRF_AES": ["PRS_AES: Aesthetic Appreciation", "PRS_APP: Attractiveness", "PRS_DIV: Diversity"],
    "PRF_BEH": ["PRS_WLK: Walkability", "PRS_CYC: Bikeability", "PRS_SAF_VIS: Visual Safety"],
    "PRF_COM": ["PRS_COL: Cooling Effect"],
    "PRF_ENV": ["PRS_PM2: PM2.5", "PRS_QUA: Environmental Quality"],
    "PRF_HLT": ["PRS_STR: Stress Relief", "PRS_SAT: Satisfaction", "PRS_ATT: Attention"],
    "PRF_SOC": ["PRS_SOC: Social Interaction"],
}

ZONE_TYPES = [
    "Entrance/Gateway", "Plaza", "Lawn", "Playground", "Senior Activity Area",
    "Fitness Area", "Waterfront", "Woodland", "Garden", "Path", "Rest Area"
]


def extract_gps(path: str) -> Tuple[bool, Optional[float], Optional[float]]:
    """Extract GPS information from image EXIF"""
    try:
        img = Image.open(path)
        exif = img._getexif()
        if not exif:
            return False, None, None
        
        gps_info = {}
        for tag_id, value in exif.items():
            if TAGS.get(tag_id) == 'GPSInfo':
                for gps_tag_id, gps_value in value.items():
                    gps_info[GPSTAGS.get(gps_tag_id)] = gps_value
        
        if 'GPSLatitude' in gps_info and 'GPSLongitude' in gps_info:
            def to_deg(v):
                return float(v[0]) + float(v[1])/60 + float(v[2])/3600
            lat = to_deg(gps_info['GPSLatitude'])
            lon = to_deg(gps_info['GPSLongitude'])
            if gps_info.get('GPSLatitudeRef') == 'S': lat = -lat
            if gps_info.get('GPSLongitudeRef') == 'W': lon = -lon
            return True, lat, lon
        return False, None, None
    except:
        return False, None, None


def create_project_questionnaire_tab(components: dict, app_state, config):
    """Create Project Questionnaire Tab"""
    
    with gr.Tab("1. Project Questionnaire"):
        gr.Markdown("## 📋 Design Goal Definition")
        
        # ===== Project Information =====
        with gr.Group():
            gr.Markdown("### 📌 Project Information")
            with gr.Row():
                project_name = gr.Textbox(label="Project Name *", placeholder="e.g., Central Park Renovation")
                project_loc = gr.Textbox(label="Location", placeholder="e.g., New York, USA")
            with gr.Row():
                site_scale = gr.Dropdown(label="Scale", choices=[o[1] for o in OPTIONS["scale"]])
                project_phase = gr.Dropdown(label="Phase", choices=[o[1] for o in OPTIONS["phase"]])
        
        # ===== Site Context =====
        with gr.Group():
            gr.Markdown("### 🌍 Site Context")
            with gr.Row():
                koppen = gr.Dropdown(label="Climate Zone *", choices=[o[1] for o in OPTIONS["koppen"]])
                country = gr.Dropdown(label="Country/Region", choices=[o[1] for o in OPTIONS["country"]])
            with gr.Row():
                space_type = gr.Dropdown(label="Space Type *", choices=[o[1] for o in OPTIONS["space_type"]])
                lcz = gr.Dropdown(label="LCZ", choices=[o[1] for o in OPTIONS["lcz"]])
            age_group = gr.Dropdown(label="Target Users", choices=[o[1] for o in OPTIONS["age"]])
        
        # ===== Performance Goals =====
        with gr.Group():
            gr.Markdown("### 🎯 Performance Goals")
            design_brief = gr.Textbox(
                label="Design Brief",
                placeholder="Describe performance goals, constraints, priorities...",
                lines=3
            )
            
            perf_dims = gr.CheckboxGroup(
                label="Performance Dimensions *",
                choices=[f"{p[1]}" for p in PERFORMANCE_DIMS]
            )
            
            with gr.Accordion("Subdimensions (Optional)", open=False):
                subdims = gr.CheckboxGroup(label="", choices=[])
        
        # ===== Spatial Zones =====
        with gr.Group():
            gr.Markdown("### 🗺️ Spatial Zones")
            
            zones_state = gr.State([])
            
            with gr.Row():
                zone_name = gr.Textbox(label="Zone Name", placeholder="e.g., Main Plaza", scale=2)
                zone_types = gr.Dropdown(label="Type", choices=ZONE_TYPES, multiselect=True, scale=2)
                add_zone_btn = gr.Button("➕ Add", scale=1)
            
            zones_table = gr.Dataframe(
                headers=["ID", "Name", "Types", "Images"],
                interactive=False
            )
            
            with gr.Row():
                del_zone_id = gr.Textbox(label="Delete ID", placeholder="zone_1", scale=2)
                del_zone_btn = gr.Button("🗑️ Delete", scale=1)
        
        # ===== Image Upload =====
        with gr.Group():
            gr.Markdown("### 📸 Image Upload")
            
            image_upload = gr.File(
                label="Select Images (batch supported)",
                file_count="multiple",
                file_types=["image"]
            )
            
            with gr.Row():
                total_imgs = gr.Textbox(label="Total", value="0", interactive=False)
                grouped_imgs = gr.Textbox(label="Grouped", value="0", interactive=False)
            
            with gr.Accordion("GPS Information", open=False):
                gps_table = gr.Dataframe(headers=["File", "GPS", "Latitude", "Longitude"])
            
            ungrouped_gallery = gr.Gallery(label="Ungrouped Images", columns=6, height="auto")
            
            with gr.Row():
                img_id = gr.Textbox(label="Image ID", scale=2)
                assign_zone = gr.Dropdown(label="Assign to Zone", choices=[], scale=2)
                assign_btn = gr.Button("Assign", scale=1)
        
        # ===== Action Buttons =====
        with gr.Row():
            reset_btn = gr.Button("🔄 Reset")
            validate_btn = gr.Button("✅ Validate")
            generate_btn = gr.Button("📄 Generate Query", variant="primary")
        
        status = gr.Textbox(label="Status", interactive=False)
        
        with gr.Accordion("Generated Query JSON", open=False, visible=False) as output_panel:
            query_json = gr.JSON()
            download_file = gr.File(label="Download", visible=False)
        
        # ========== Event Handlers ==========
        
        def update_subdims(selected):
            all_subdims = []
            for sel in selected:
                for p_id, p_name, _ in PERFORMANCE_DIMS:
                    if p_name in sel:
                        all_subdims.extend(SUBDIMS.get(p_id, []))
            return gr.update(choices=all_subdims)
        
        def add_zone(zones, name, types):
            if not name.strip():
                return zones, zones, "Please enter a name"
            
            zone_id = f"zone_{len(zones)+1}"
            zones.append([zone_id, name, ", ".join(types or []), 0])
            
            app_state.project_query.add_zone(name, types, "")
            
            return zones, zones, f"✅ Added: {name}"
        
        def delete_zone(zones, zone_id):
            if not zone_id:
                return zones, zones, "Please enter ID"
            
            new_zones = [z for z in zones if z[0] != zone_id]
            if len(new_zones) == len(zones):
                return zones, zones, f"Not found: {zone_id}"
            
            app_state.project_query.remove_zone(zone_id)
            return new_zones, new_zones, f"✅ Deleted"
        
        def process_images(files, zones):
            if not files:
                return [], "0", "0", [], gr.update(choices=[])
            
            gps_data = []
            gallery = []
            
            for f in files:
                path = f.name
                name = os.path.basename(path)
                has_gps, lat, lon = extract_gps(path)
                
                img_id = app_state.project_query.add_image(name, path, has_gps, lat, lon)
                
                gps_data.append([name, "Yes" if has_gps else "No", 
                                f"{lat:.4f}" if lat else "", f"{lon:.4f}" if lon else ""])
                gallery.append((path, f"{img_id}: {name}"))
            
            total = len(app_state.project_query.uploaded_images)
            grouped = len([i for i in app_state.project_query.uploaded_images if i.zone_id])
            
            zone_choices = [f"{z[0]}: {z[1]}" for z in zones]
            
            return gps_data, str(total), str(grouped), gallery, gr.update(choices=zone_choices)
        
        def assign_image(img_id, zone_sel, zones):
            if not img_id or not zone_sel:
                return "Select image and zone", "", ""
            
            zone_id = zone_sel.split(":")[0].strip()
            
            for img in app_state.project_query.uploaded_images:
                if img.image_id == img_id or img_id in img.filename:
                    app_state.project_query.assign_image(img.image_id, zone_id)
                    break
            
            total = len(app_state.project_query.uploaded_images)
            grouped = len([i for i in app_state.project_query.uploaded_images if i.zone_id])
            
            return f"✅ Assigned", str(total), str(grouped)
        
        def validate(name, koppen, space):
            errors = []
            if not name: errors.append("Project Name")
            if not koppen or koppen == "Select *": errors.append("Climate Zone")
            if not space or space == "Select *": errors.append("Space Type")
            
            if errors:
                return f"❌ Please fill in: {', '.join(errors)}", gr.update(visible=False)
            return "✅ Validation passed", gr.update(visible=True)
        
        def get_id(options, label):
            for o in options:
                if o[1] == label:
                    return o[0]
            return ""
        
        def generate(name, loc, scale, phase, kop, ctry, space, lcz_v, age,
                    brief, dims, subdims_v, zones):
            q = app_state.project_query
            q.project_name = name
            q.project_location = loc
            q.site_scale = get_id(OPTIONS["scale"], scale)
            q.project_phase = get_id(OPTIONS["phase"], phase)
            q.koppen_zone_id = get_id(OPTIONS["koppen"], kop)
            q.country_id = get_id(OPTIONS["country"], ctry)
            q.space_type_id = get_id(OPTIONS["space_type"], space)
            q.lcz_type_id = get_id(OPTIONS["lcz"], lcz_v)
            q.age_group_id = get_id(OPTIONS["age"], age)
            q.design_brief = brief
            
            q.performance_dimensions = []
            for d in dims:
                for p_id, p_name, _ in PERFORMANCE_DIMS:
                    if p_name in d:
                        q.performance_dimensions.append(p_id)
            
            q.subdimensions = [s.split(":")[0].strip() for s in subdims_v] if subdims_v else []
            
            return q.to_dict(), "✅ Generated", gr.update(visible=True)
        
        def reset():
            app_state.project_query.reset()
            return ("", "", None, None, None, None, None, None, None,
                    "", [], [], [], "0", "0", [], [], "Reset complete")
        
        # ===== Bind Events =====
        perf_dims.change(update_subdims, [perf_dims], [subdims])
        
        add_zone_btn.click(add_zone, [zones_state, zone_name, zone_types],
                          [zones_state, zones_table, status])
        
        del_zone_btn.click(delete_zone, [zones_state, del_zone_id],
                          [zones_state, zones_table, status])
        
        image_upload.change(process_images, [image_upload, zones_state],
                           [gps_table, total_imgs, grouped_imgs, ungrouped_gallery, assign_zone])
        
        assign_btn.click(assign_image, [img_id, assign_zone, zones_state],
                        [status, total_imgs, grouped_imgs])
        
        validate_btn.click(validate, [project_name, koppen, space_type],
                          [status, output_panel])
        
        generate_btn.click(
            generate,
            [project_name, project_loc, site_scale, project_phase,
             koppen, country, space_type, lcz, age_group,
             design_brief, perf_dims, subdims, zones_state],
            [query_json, status, output_panel]
        )
        
        reset_btn.click(
            reset,
            outputs=[project_name, project_loc, site_scale, project_phase,
                    koppen, country, space_type, lcz, age_group,
                    design_brief, perf_dims, subdims, zones_state,
                    total_imgs, grouped_imgs, gps_table, ungrouped_gallery, status]
        )
        
        return {'project_name': project_name, 'query_json': query_json}

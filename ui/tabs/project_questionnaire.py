"""
Tab 1: Project Questionnaire
Clean UI for image upload and zone assignment
"""

import gradio as gr
import os
import json
from typing import List, Tuple, Optional, Dict
from datetime import datetime
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import logging

logger = logging.getLogger(__name__)

# ========== Options ==========
KOPPEN_OPTIONS = [
    "", "Tropical Rainforest (Af)", "Tropical Monsoon (Am)", "Humid Subtropical (Cfa)",
    "Oceanic (Cfb)", "Monsoon Subtropical (Cwa)", "Hot-summer Continental (Dfa)"
]
KOPPEN_IDS = ["", "KPN_AF", "KPN_AM", "KPN_CFA", "KPN_CFB", "KPN_CWA", "KPN_DFA"]

SPACE_OPTIONS = ["", "Park", "Residential", "Street", "Campus", "Business District", "Greenway", "Waterfront"]
SPACE_IDS = ["", "SET_PRK", "SET_RES", "SET_STR", "SET_CAM", "SET_CBD", "SET_GWY", "SET_WAT"]

COUNTRY_OPTIONS = ["", "China", "Hong Kong", "Japan", "South Korea", "Singapore", "USA", "UK", "Netherlands"]
COUNTRY_IDS = ["", "CNT_CHN", "CNT_HKG", "CNT_JPN", "CNT_KOR", "CNT_SGP", "CNT_USA", "CNT_GBR", "CNT_NLD"]

LCZ_OPTIONS = ["", "Compact High-rise", "Compact Mid-rise", "Open High-rise", "Dense Trees", "Scattered Trees"]
LCZ_IDS = ["", "LCZ_1", "LCZ_2", "LCZ_4", "LCZ_A", "LCZ_B"]

AGE_OPTIONS = ["", "All Ages", "Children", "Adults", "Elderly"]
AGE_IDS = ["", "AGE_ALL", "AGE_CHD", "AGE_ADL", "AGE_ELD"]

SCALE_OPTIONS = ["", "< 0.5 ha", "0.5-2 ha", "2-10 ha", "10-50 ha", "> 50 ha"]
SCALE_IDS = ["", "XS", "S", "M", "L", "XL"]

PHASE_OPTIONS = ["", "Conceptual", "Schematic", "Detailed", "Renovation", "Post-Occupancy"]
PHASE_IDS = ["", "conceptual", "schematic", "detailed", "renovation", "post_occupancy"]

PERFORMANCE_DIMS = {
    "👁️ Visual Quality": "PRF_AES",
    "🚶 Use & Safety": "PRF_BEH",
    "😌 Comfort": "PRF_COM",
    "🌿 Environment": "PRF_ENV",
    "🧠 Health": "PRF_HLT",
    "🤝 Social": "PRF_SOC"
}

ZONE_TYPES = ["Entrance", "Plaza", "Lawn", "Playground", "Fitness Area", "Waterfront", "Woodland", "Path", "Rest Area", "Sports Field"]


def extract_gps(filepath: str) -> Tuple[bool, Optional[float], Optional[float]]:
    """Extract GPS coordinates from image EXIF"""
    try:
        img = Image.open(filepath)
        exif = img._getexif()
        if not exif:
            return False, None, None
        
        gps_info = {}
        for tag_id, value in exif.items():
            tag = TAGS.get(tag_id)
            if tag == 'GPSInfo':
                for gps_tag_id in value:
                    gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                    gps_info[gps_tag] = value[gps_tag_id]
        
        if 'GPSLatitude' in gps_info and 'GPSLongitude' in gps_info:
            def convert_to_degrees(value):
                d, m, s = value
                return float(d) + float(m)/60 + float(s)/3600
            
            lat = convert_to_degrees(gps_info['GPSLatitude'])
            lon = convert_to_degrees(gps_info['GPSLongitude'])
            
            if gps_info.get('GPSLatitudeRef') == 'S':
                lat = -lat
            if gps_info.get('GPSLongitudeRef') == 'W':
                lon = -lon
            
            return True, lat, lon
    except Exception as e:
        logger.debug(f"GPS extraction failed: {e}")
    
    return False, None, None


def get_id_from_label(options: List[str], ids: List[str], label: str) -> str:
    """Get ID from label"""
    try:
        idx = options.index(label)
        return ids[idx]
    except:
        return ""


def create_project_questionnaire_tab(components: dict, app_state, config):
    """Create Project Questionnaire Tab"""
    

    with gr.Tab("1. Project Questionnaire"):
        # Internal state for images
        images_state = gr.State({})  # {image_id: {filename, filepath, zone_id, has_gps, lat, lon}}
        zones_state = gr.State({})   # {zone_id: {name, type}}
    
        gr.Markdown("## 📋 Design Goal Definition")
        
        with gr.Row():
            # ===== LEFT COLUMN =====
            with gr.Column(scale=1):
                # Project Info
                gr.Markdown("### 📌 Project Information")
                project_name = gr.Textbox(label="Project Name *", placeholder="e.g., Central Park Renovation")
                project_loc = gr.Textbox(label="Location", placeholder="City, Country")
                
                with gr.Row():
                    site_scale = gr.Dropdown(label="Scale", choices=SCALE_OPTIONS)
                    project_phase = gr.Dropdown(label="Phase", choices=PHASE_OPTIONS)
                
                # Site Context
                gr.Markdown("### 🌍 Site Context")
                koppen = gr.Dropdown(label="Climate Zone *", choices=KOPPEN_OPTIONS)
                space_type = gr.Dropdown(label="Space Type *", choices=SPACE_OPTIONS)
                
                with gr.Row():
                    country = gr.Dropdown(label="Country", choices=COUNTRY_OPTIONS)
                    lcz = gr.Dropdown(label="LCZ", choices=LCZ_OPTIONS)
                
                age_group = gr.Dropdown(label="Target Users", choices=AGE_OPTIONS)
                
                # Performance Goals
                gr.Markdown("### 🎯 Performance Goals")
                design_brief = gr.Textbox(label="Design Brief", placeholder="Describe your design goals...", lines=3)
                perf_dims = gr.CheckboxGroup(
                    label="Performance Dimensions *",
                    choices=list(PERFORMANCE_DIMS.keys())
                )
            
            # ===== RIGHT COLUMN =====
            with gr.Column(scale=1):
                # Zones Section
                gr.Markdown("### 🗺️ Spatial Zones")
                
                with gr.Row():
                    zone_name_input = gr.Textbox(label="Zone Name", placeholder="e.g., Main Entrance", scale=2)
                    zone_type_input = gr.Dropdown(label="Type", choices=ZONE_TYPES, scale=1)
                    add_zone_btn = gr.Button("➕ Add", variant="primary", scale=1)
                
                zone_list = gr.Dataframe(
                    headers=["ID", "Name", "Type", "Images"],
                    datatype=["str", "str", "str", "number"],
                    interactive=False,
                    row_count=4,
                    col_count=(4, "fixed"),
                    label="Zones"
                )
                
                with gr.Row():
                    del_zone_id = gr.Textbox(label="Delete Zone ID", placeholder="zone_1", scale=2)
                    del_zone_btn = gr.Button("🗑️ Delete", variant="stop", scale=1)
                
                # Images Section
                gr.Markdown("### 📸 Site Photos")
                
                image_upload = gr.File(
                    label="Upload Images (JPG, PNG)",
                    file_count="multiple",
                    file_types=["image"]
                )
                
                with gr.Row():
                    total_imgs = gr.Textbox(label="Total", value="0", interactive=False, scale=1)
                    assigned_imgs = gr.Textbox(label="Assigned", value="0", interactive=False, scale=1)
                    unassigned_imgs = gr.Textbox(label="Unassigned", value="0", interactive=False, scale=1)
                
                # Assignment UI
                gr.Markdown("#### Assign Images to Zones")
                
                with gr.Row():
                    img_select = gr.Dropdown(label="Image", choices=[], scale=2)
                    zone_select = gr.Dropdown(label="Zone", choices=[], scale=2)
                    assign_btn = gr.Button("✓", variant="primary", scale=1)
                
                # Image-Zone Table
                img_zone_table = gr.Dataframe(
                    headers=["Image", "Zone", "GPS"],
                    datatype=["str", "str", "str"],
                    interactive=False,
                    row_count=6,
                    col_count=(3, "fixed"),
                    label="Assignments"
                )
                
                # Batch assign
                with gr.Row():
                    batch_zone = gr.Dropdown(label="Batch assign unassigned to:", choices=[], scale=2)
                    batch_btn = gr.Button("Assign All", scale=1)
        
        # ===== Actions =====
        gr.Markdown("---")
        with gr.Row():
            validate_btn = gr.Button("✅ Validate")
            generate_btn = gr.Button("📄 Generate Query", variant="primary")
            reset_btn = gr.Button("🔄 Reset")
        
        status_msg = gr.Textbox(label="Status", interactive=False)
        
        with gr.Accordion("Query JSON Output", open=False):
            query_json = gr.JSON(label="Generated Query")
        
        # ========== FUNCTIONS ==========
        
        def add_zone(zones, name, ztype):
            if not name or not name.strip():
                return zones, update_zone_table(zones), "Enter zone name", gr.update(), gr.update()
            
            zone_id = f"zone_{len(zones) + 1}"
            zones[zone_id] = {"name": name.strip(), "type": ztype or "", "images": 0}
            
            # Update app state
            app_state.project_query.add_zone(name.strip(), [ztype] if ztype else [], "")
            
            zone_choices = [f"{zid}: {z['name']}" for zid, z in zones.items()]
            return zones, update_zone_table(zones), f"✅ Added {name}", gr.update(choices=zone_choices), gr.update(choices=zone_choices)
        
        def delete_zone(zones, images, zone_id):
            if not zone_id or zone_id.strip() not in zones:
                return zones, images, update_zone_table(zones), update_img_table(images, zones), "Zone not found", gr.update(), gr.update()
            
            zid = zone_id.strip()
            del zones[zid]
            
            # Unassign images from this zone
            for img_id in images:
                if images[img_id].get("zone_id") == zid:
                    images[img_id]["zone_id"] = None
            
            app_state.project_query.remove_zone(zid)
            
            zone_choices = [f"{z}: {zones[z]['name']}" for z in zones]
            return zones, images, update_zone_table(zones), update_img_table(images, zones), f"✅ Deleted {zid}", gr.update(choices=zone_choices), gr.update(choices=zone_choices)
        
        def update_zone_table(zones):
            data = []
            for zid, z in zones.items():
                data.append([zid, z["name"], z["type"], z.get("images", 0)])
            return data if data else []
        
        def update_img_table(images, zones):
            data = []
            for img_id, img in images.items():
                zone_name = "Unassigned"
                if img.get("zone_id") and img["zone_id"] in zones:
                    zone_name = zones[img["zone_id"]]["name"]
                gps = "Yes" if img.get("has_gps") else "No"
                data.append([img["filename"], zone_name, gps])
            return data if data else []
        
        def get_counts(images):
            total = len(images)
            assigned = len([i for i in images.values() if i.get("zone_id")])
            return str(total), str(assigned), str(total - assigned)
        
        def process_uploads(files, images, zones):
            if not files:
                return images, update_img_table(images, zones), *get_counts(images), gr.update()
            
            for f in files:
                filepath = f.name
                filename = os.path.basename(filepath)
                has_gps, lat, lon = extract_gps(filepath)
                
                img_id = f"img_{len(images) + 1}"
                images[img_id] = {
                    "filename": filename,
                    "filepath": filepath,
                    "zone_id": None,
                    "has_gps": has_gps,
                    "lat": lat,
                    "lon": lon
                }
                
                # Also add to app state
                app_state.project_query.add_image(filename, filepath, has_gps, lat, lon)
            
            img_choices = [f"{iid}: {images[iid]['filename']}" for iid in images]
            total, assigned, unassigned = get_counts(images)
            
            return images, update_img_table(images, zones), total, assigned, unassigned, gr.update(choices=img_choices)
        
        def assign_single(images, zones, img_sel, zone_sel):
            if not img_sel or not zone_sel:
                return images, zones, update_zone_table(zones), update_img_table(images, zones), *get_counts(images), "Select both"
            
            img_id = img_sel.split(":")[0].strip()
            zone_id = zone_sel.split(":")[0].strip()
            
            if img_id not in images or zone_id not in zones:
                return images, zones, update_zone_table(zones), update_img_table(images, zones), *get_counts(images), "Invalid selection"
            
            # Update old zone count
            old_zone = images[img_id].get("zone_id")
            if old_zone and old_zone in zones:
                zones[old_zone]["images"] = max(0, zones[old_zone].get("images", 1) - 1)
            
            # Assign
            images[img_id]["zone_id"] = zone_id
            zones[zone_id]["images"] = zones[zone_id].get("images", 0) + 1
            
            # Update app state
            app_state.project_query.assign_image(img_id, zone_id)
            
            return images, zones, update_zone_table(zones), update_img_table(images, zones), *get_counts(images), f"✅ Assigned"
        
        def batch_assign(images, zones, zone_sel):
            if not zone_sel:
                return images, zones, update_zone_table(zones), update_img_table(images, zones), *get_counts(images), "Select zone"
            
            zone_id = zone_sel.split(":")[0].strip()
            if zone_id not in zones:
                return images, zones, update_zone_table(zones), update_img_table(images, zones), *get_counts(images), "Invalid zone"
            
            count = 0
            for img_id in images:
                if not images[img_id].get("zone_id"):
                    images[img_id]["zone_id"] = zone_id
                    zones[zone_id]["images"] = zones[zone_id].get("images", 0) + 1
                    app_state.project_query.assign_image(img_id, zone_id)
                    count += 1
            
            return images, zones, update_zone_table(zones), update_img_table(images, zones), *get_counts(images), f"✅ Assigned {count} images"
        
        def validate(name, climate, space):
            errors = []
            if not name: errors.append("Project Name")
            if not climate: errors.append("Climate Zone")
            if not space: errors.append("Space Type")
            
            if errors:
                return f"❌ Missing: {', '.join(errors)}"
            return "✅ Valid"
        
        def generate(name, loc, scale, phase, climate, space, country_v, lcz_v, age_v, brief, dims, zones, images):
            q = app_state.project_query
            q.project_name = name or ""
            q.project_location = loc or ""
            q.site_scale = get_id_from_label(SCALE_OPTIONS, SCALE_IDS, scale)
            q.project_phase = get_id_from_label(PHASE_OPTIONS, PHASE_IDS, phase)
            q.koppen_zone_id = get_id_from_label(KOPPEN_OPTIONS, KOPPEN_IDS, climate)
            q.space_type_id = get_id_from_label(SPACE_OPTIONS, SPACE_IDS, space)
            q.country_id = get_id_from_label(COUNTRY_OPTIONS, COUNTRY_IDS, country_v)
            q.lcz_type_id = get_id_from_label(LCZ_OPTIONS, LCZ_IDS, lcz_v)
            q.age_group_id = get_id_from_label(AGE_OPTIONS, AGE_IDS, age_v)
            q.design_brief = brief or ""
            
            q.performance_dimensions = [PERFORMANCE_DIMS[d] for d in (dims or []) if d in PERFORMANCE_DIMS]
            
            return q.to_dict(), "✅ Generated"
        
        def reset_all():
            app_state.project_query.reset()
            return (
                {}, {},  # states
                [], [],  # tables
                "0", "0", "0",  # counts
                gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[]),  # dropdowns
                "", "", None, None, None, None, None, None, None, "", [],  # form fields
                "Reset complete"
            )
        
        # ========== EVENTS ==========
        
        add_zone_btn.click(
            add_zone,
            [zones_state, zone_name_input, zone_type_input],
            [zones_state, zone_list, status_msg, zone_select, batch_zone]
        )
        
        del_zone_btn.click(
            delete_zone,
            [zones_state, images_state, del_zone_id],
            [zones_state, images_state, zone_list, img_zone_table, status_msg, zone_select, batch_zone]
        )
        
        image_upload.change(
            process_uploads,
            [image_upload, images_state, zones_state],
            [images_state, img_zone_table, total_imgs, assigned_imgs, unassigned_imgs, img_select]
        )
        
        assign_btn.click(
            assign_single,
            [images_state, zones_state, img_select, zone_select],
            [images_state, zones_state, zone_list, img_zone_table, total_imgs, assigned_imgs, unassigned_imgs, status_msg]
        )
        
        batch_btn.click(
            batch_assign,
            [images_state, zones_state, batch_zone],
            [images_state, zones_state, zone_list, img_zone_table, total_imgs, assigned_imgs, unassigned_imgs, status_msg]
        )
        
        validate_btn.click(validate, [project_name, koppen, space_type], [status_msg])
        
        generate_btn.click(
            generate,
            [project_name, project_loc, site_scale, project_phase, koppen, space_type,
             country, lcz, age_group, design_brief, perf_dims, zones_state, images_state],
            [query_json, status_msg]
        )
        
        reset_btn.click(
            reset_all,
            outputs=[
                zones_state, images_state,
                zone_list, img_zone_table,
                total_imgs, assigned_imgs, unassigned_imgs,
                img_select, zone_select, batch_zone,
                project_name, project_loc, site_scale, project_phase, koppen, space_type,
                country, lcz, age_group, design_brief, perf_dims,
                status_msg
            ]
        )
        
        return {"query_json": query_json}

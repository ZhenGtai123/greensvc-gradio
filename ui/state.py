"""
Application State Management Module
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import pandas as pd


@dataclass
class SpatialZone:
    """Spatial Zone"""
    zone_id: str
    zone_name: str
    zone_types: List[str] = field(default_factory=list)
    description: str = ""


@dataclass 
class UploadedImage:
    """Uploaded Image"""
    image_id: str
    filename: str
    filepath: str
    zone_id: Optional[str] = None
    has_gps: bool = False
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class ProjectQuery:
    """Project Questionnaire Data"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        # Project Info
        self.project_name = ""
        self.project_location = ""
        self.site_scale = ""
        self.project_phase = ""
        
        # Site Context
        self.koppen_zone_id = ""
        self.country_id = ""
        self.space_type_id = ""
        self.lcz_type_id = ""
        self.age_group_id = ""
        
        # Performance Goals
        self.design_brief = ""
        self.performance_dimensions: List[str] = []
        self.subdimensions: List[str] = []
        
        # Spatial Zones
        self.spatial_zones: List[SpatialZone] = []
        
        # Images
        self.uploaded_images: List[UploadedImage] = []
    
    def add_zone(self, zone_name: str, zone_types: List[str] = None, description: str = "") -> str:
        """Add spatial zone"""
        zone_id = f"zone_{len(self.spatial_zones) + 1}"
        self.spatial_zones.append(SpatialZone(
            zone_id=zone_id,
            zone_name=zone_name,
            zone_types=zone_types or [],
            description=description
        ))
        return zone_id
    
    def remove_zone(self, zone_id: str):
        """Remove zone"""
        self.spatial_zones = [z for z in self.spatial_zones if z.zone_id != zone_id]
        for img in self.uploaded_images:
            if img.zone_id == zone_id:
                img.zone_id = None
    
    def get_zone(self, zone_id: str) -> Optional[SpatialZone]:
        """Get zone by ID"""
        return next((z for z in self.spatial_zones if z.zone_id == zone_id), None)
    
    def add_image(self, filename: str, filepath: str, 
                  has_gps: bool = False, lat: float = None, lon: float = None) -> str:
        """Add image"""
        image_id = f"img_{len(self.uploaded_images) + 1}"
        self.uploaded_images.append(UploadedImage(
            image_id=image_id,
            filename=filename,
            filepath=filepath,
            has_gps=has_gps,
            latitude=lat,
            longitude=lon
        ))
        return image_id
    
    def assign_image(self, image_id: str, zone_id: Optional[str]):
        """Assign image to zone"""
        for img in self.uploaded_images:
            if img.image_id == image_id:
                img.zone_id = zone_id
                break
    
    def get_zone_images(self, zone_id: str) -> List[UploadedImage]:
        """Get images in zone"""
        return [img for img in self.uploaded_images if img.zone_id == zone_id]
    
    def get_ungrouped_images(self) -> List[UploadedImage]:
        """Get ungrouped images"""
        return [img for img in self.uploaded_images if img.zone_id is None]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary (for API calls and JSON export)"""
        return {
            "query_metadata": {
                "query_version": "2.0",
                "generated_at": datetime.now().isoformat(),
                "system": "GreenSVC-AI"
            },
            "project": {
                "name": self.project_name,
                "location": self.project_location or None,
                "scale": self.site_scale or None,
                "phase": self.project_phase or None
            },
            "context": {
                "climate": {"koppen_zone_id": self.koppen_zone_id},
                "urban_form": {
                    "space_type_id": self.space_type_id,
                    "lcz_type_id": self.lcz_type_id or None
                },
                "user": {"age_group_id": self.age_group_id or None},
                "country_id": self.country_id or None
            },
            "performance_query": {
                "design_brief": self.design_brief or None,
                "dimensions": self.performance_dimensions,
                "subdimensions": self.subdimensions
            },
            "spatial_zones": [
                {
                    "zone_id": z.zone_id,
                    "zone_name": z.zone_name,
                    "zone_types": z.zone_types,
                    "description": z.description
                } for z in self.spatial_zones
            ],
            "site_photos": {
                "total_images": len(self.uploaded_images),
                "grouped_images": len([i for i in self.uploaded_images if i.zone_id]),
                "groups": [
                    {
                        "zone_id": z.zone_id,
                        "zone_name": z.zone_name,
                        "images": [img.filename for img in self.get_zone_images(z.zone_id)]
                    } for z in self.spatial_zones if self.get_zone_images(z.zone_id)
                ]
            }
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        def clean(obj):
            if isinstance(obj, dict):
                return {k: clean(v) for k, v in obj.items() 
                        if v is not None and v != "" and v != []}
            elif isinstance(obj, list):
                return [clean(i) for i in obj if i is not None]
            return obj
        return json.dumps(clean(self.to_dict()), ensure_ascii=False, indent=indent)


class AppState:
    """Application State Manager"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.project_query = ProjectQuery()
        self.selected_metrics: List[Dict] = []
        self.recommended_indicators: List[Dict] = []
        self.vision_results: Dict[str, Dict] = {}
        self.metrics_results = pd.DataFrame()
        self.gps_data: Dict = {}
        self.components = None
    
    def set_components(self, components: Dict):
        self.components = components
    
    # Metrics related
    def set_selected_metrics(self, metrics: List[Dict]):
        self.selected_metrics = metrics
    
    def get_selected_metrics(self) -> List[Dict]:
        return self.selected_metrics
    
    def set_recommended_indicators(self, indicators: List[Dict]):
        self.recommended_indicators = indicators
    
    def get_recommended_indicators(self) -> List[Dict]:
        return self.recommended_indicators
    
    # Vision analysis related
    def add_vision_result(self, path: str, result: Dict):
        self.vision_results[path] = result
    
    def get_vision_results(self) -> Dict:
        return self.vision_results
    
    def has_vision_results(self) -> bool:
        return len(self.vision_results) > 0
    
    # Metrics calculation related
    def set_metrics_results(self, results: pd.DataFrame):
        self.metrics_results = results
    
    def get_metrics_results(self) -> pd.DataFrame:
        return self.metrics_results
    
    def has_metrics_results(self) -> bool:
        return not self.metrics_results.empty
    
    # GPS related
    def set_gps_data(self, data: Dict):
        self.gps_data = data
    
    def get_gps_data(self) -> Dict:
        return self.gps_data
    
    # Convenience methods
    def get_all_image_paths(self) -> List[str]:
        return [img.filepath for img in self.project_query.uploaded_images]

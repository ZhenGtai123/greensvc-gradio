"""
图像处理模块
- GPS信息提取
- 图片尺寸调整（800x600）
- 中心裁剪
"""

import os
import numpy as np
from PIL import Image
import exifread
import cv2
from typing import Tuple, Optional, List, Dict
import tempfile
import logging

logger = logging.getLogger(__name__)

class ImageProcessor:
    """图像处理器类"""
    
    # 定义目标图片尺寸
    TARGET_WIDTH = 800   # 宽度为800像素
    TARGET_HEIGHT = 600  # 高度为600像素
    TARGET_DPI = 300     # DPI设置为300
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp(prefix='image_processor_')
        logger.info(f"Created temp directory: {self.temp_dir}")
    
    def extract_gps_info(self, image_path: str) -> Tuple[Optional[float], Optional[float]]:
        """
        从图片中提取GPS信息
        
        Args:
            image_path: 图片路径
            
        Returns:
            (latitude, longitude) 或 (None, None)
        """
        try:
            with open(image_path, 'rb') as f:
                tags = exifread.process_file(f, details=False)
                
            # 获取GPS信息
            gps_latitude = tags.get('GPS GPSLatitude')
            gps_latitude_ref = tags.get('GPS GPSLatitudeRef')
            gps_longitude = tags.get('GPS GPSLongitude')
            gps_longitude_ref = tags.get('GPS GPSLongitudeRef')
            
            if gps_latitude and gps_longitude:
                # 转换GPS坐标为十进制度数
                lat = self._convert_to_degrees(gps_latitude)
                lon = self._convert_to_degrees(gps_longitude)
                
                # 考虑南北半球和东西半球
                if gps_latitude_ref and str(gps_latitude_ref) == 'S':
                    lat = -lat
                if gps_longitude_ref and str(gps_longitude_ref) == 'W':
                    lon = -lon
                    
                return lat, lon
        except Exception as e:
            logger.warning(f"提取GPS信息失败 {image_path}: {e}")
        
        return None, None
    
    def _convert_to_degrees(self, value) -> float:
        """将GPS坐标转换为十进制度数"""
        try:
            d = float(value.values[0].num) / float(value.values[0].den)
            m = float(value.values[1].num) / float(value.values[1].den)
            s = float(value.values[2].num) / float(value.values[2].den)
            return d + (m / 60.0) + (s / 3600.0)
        except:
            return 0.0
    
    def process_and_resize_image(self, img_path: str, output_path: Optional[str] = None) -> str:
        """
        处理单个图片：调整尺寸为800x600，进行中心裁剪
        
        Args:
            img_path: 输入图片路径
            output_path: 输出路径（可选）
            
        Returns:
            处理后的图片路径
        """
        try:
            # 打开图片
            img = Image.open(img_path)
            
            # 如果是RGBA，转换为RGB
            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 获取原始尺寸
            original_width, original_height = img.size
            
            # 计算缩放比例
            # 确保缩放后的图片能够覆盖目标尺寸
            width_ratio = self.TARGET_WIDTH / original_width
            height_ratio = self.TARGET_HEIGHT / original_height
            
            # 选择较大的缩放比例，确保图片填满目标区域
            scale_ratio = max(width_ratio, height_ratio)
            
            # 计算新的尺寸
            new_width = int(original_width * scale_ratio)
            new_height = int(original_height * scale_ratio)
            
            # 调整图片大小
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 计算裁剪区域（中心裁剪）
            left = (new_width - self.TARGET_WIDTH) // 2
            top = (new_height - self.TARGET_HEIGHT) // 2
            right = left + self.TARGET_WIDTH
            bottom = top + self.TARGET_HEIGHT
            
            # 裁剪图片
            img = img.crop((left, top, right, bottom))
            
            # 确保最终尺寸正确
            if img.size != (self.TARGET_WIDTH, self.TARGET_HEIGHT):
                img = img.resize((self.TARGET_WIDTH, self.TARGET_HEIGHT), Image.Resampling.LANCZOS)
            
            # 如果没有指定输出路径，创建临时文件
            if output_path is None:
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False, 
                    suffix='.png',
                    dir=self.temp_dir
                )
                output_path = temp_file.name
            
            # 保存图片，设置DPI
            img.save(output_path, 'PNG', dpi=(self.TARGET_DPI, self.TARGET_DPI))
            
            logger.info(f"图片已处理并保存到: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"处理图片 {img_path} 时出错: {str(e)}")
            raise
    
    def batch_process_images(self, image_files: List[str]) -> Dict[str, Dict]:
        """
        批量处理图片
        
        Args:
            image_files: 图片文件路径列表
            
        Returns:
            处理结果字典，包含处理后的路径、GPS信息等
        """
        results = {}
        
        for idx, img_file in enumerate(image_files):
            try:
                # 提取GPS信息
                lat, lon = self.extract_gps_info(img_file)
                
                # 处理图片（调整尺寸）
                processed_path = self.process_and_resize_image(img_file)
                
                results[img_file] = {
                    'index': idx,
                    'processed_path': processed_path,
                    'gps': (lat, lon),
                    'has_gps': lat is not None and lon is not None,
                    'original_path': img_file,
                    'status': 'success'
                }
                
                logger.info(f"成功处理图片 {idx+1}/{len(image_files)}: {os.path.basename(img_file)}")
                
            except Exception as e:
                logger.error(f"处理图片失败 {img_file}: {str(e)}")
                results[img_file] = {
                    'index': idx,
                    'processed_path': None,
                    'gps': (None, None),
                    'has_gps': False,
                    'original_path': img_file,
                    'status': 'error',
                    'error': str(e)
                }
        
        return results
    
    def check_all_images_have_gps(self, results: Dict[str, Dict]) -> Tuple[bool, int, int]:
        """
        检查是否所有图片都有GPS信息
        
        Args:
            results: batch_process_images的返回结果
            
        Returns:
            (是否都有GPS, 有GPS的数量, 总数量)
        """
        total = len(results)
        with_gps = sum(1 for r in results.values() if r['has_gps'])
        all_have_gps = (with_gps == total) and (total > 0)
        
        return all_have_gps, with_gps, total
    
    def get_gps_bounds(self, results: Dict[str, Dict]) -> Dict[str, float]:
        """
        获取GPS坐标的边界
        
        Args:
            results: batch_process_images的返回结果
            
        Returns:
            包含min_lat, max_lat, min_lon, max_lon的字典
        """
        valid_gps = []
        for r in results.values():
            if r['has_gps']:
                lat, lon = r['gps']
                if lat is not None and lon is not None:
                    valid_gps.append((lat, lon))
        
        if not valid_gps:
            return {}
        
        lats = [gps[0] for gps in valid_gps]
        lons = [gps[1] for gps in valid_gps]
        
        return {
            'min_lat': min(lats),
            'max_lat': max(lats),
            'min_lon': min(lons),
            'max_lon': max(lons),
            'center_lat': np.mean(lats),
            'center_lon': np.mean(lons),
            'count': len(valid_gps)
        }
    
    def create_thumbnail(self, image_path: str, size: Tuple[int, int] = (150, 150)) -> str:
        """创建缩略图"""
        try:
            img = Image.open(image_path)
            img.thumbnail(size, Image.Resampling.LANCZOS)
            
            thumb_path = os.path.join(
                self.temp_dir,
                f"thumb_{os.path.basename(image_path)}"
            )
            img.save(thumb_path, 'PNG')
            
            return thumb_path
        except Exception as e:
            logger.error(f"创建缩略图失败: {e}")
            return image_path
    
    def cleanup(self):
        """清理临时文件"""
        try:
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info(f"清理临时目录: {self.temp_dir}")
        except Exception as e:
            logger.error(f"清理临时文件失败: {e}")

"""
更新指标库，添加required_images列
"""

import pandas as pd
import os

def update_metrics_library(library_path='data/library_metrics.xlsx'):
    """更新指标库，添加required_images列"""
    
    # 定义每个指标所需的图像
    required_images_mapping = {
        "Shape Edge Regularity Index (S_ERI)": "fmb_map, foreground_map",
        "Shape Edge Contrast Index (S_ECI)": "semantic_map",
        "Shape Patch Similarity Index (S_PSI)": "foreground_map",
        "Shape Axis Proximity Index (S_API)": "fmb_map, foreground_map",
        "Size View Field Ratio (S_VFR)": "foreground_map",
        "Size Area Dispersion Index (S_ADI)": "fmb_map, semantic_map",
        "Position Location Value (P_LV)": "depth_map, fmb_map",
        "Position Location Number (P_LN)": "fmb_map",
        "Texture Element Richness Index (T_ERI)": "semantic_map",
        "Texture Element Splitting Index (T_ESI)": "semantic_map",
        "Texture Interval Solidity Index (T_ISI)": "semantic_map",
        "Texture Interval Variation Index (T_IVI)": "semantic_map",
        "Position Edge Continuity Index (P_ECI)": "fmb_map, foreground_map",
        "Position Location Unique Value (P_LUV)": "depth_map",
        "Mean Patch Edge Regularity (M_PER)": "fmb_map, foreground_map"
    }
    
    try:
        # 读取现有的指标库
        if os.path.exists(library_path):
            df = pd.read_excel(library_path)
            print(f"读取现有指标库: {len(df)} 个指标")
        else:
            print(f"指标库文件不存在: {library_path}")
            return False
        
        # 检查是否已有required_images列
        if 'required_images' not in df.columns:
            df['required_images'] = ''
            print("添加 required_images 列")
        
        # 更新每个指标的required_images
        updated_count = 0
        for idx, row in df.iterrows():
            metric_name = row['metric name']
            if metric_name in required_images_mapping:
                df.at[idx, 'required_images'] = required_images_mapping[metric_name]
                updated_count += 1
            else:
                # 基于Data Input字段推断
                data_input = row.get('Data Input', '').lower()
                if 'segmentation' in data_input:
                    df.at[idx, 'required_images'] = 'semantic_map'
                elif 'depth' in data_input:
                    df.at[idx, 'required_images'] = 'depth_map'
                else:
                    df.at[idx, 'required_images'] = 'semantic_map, depth_map'
        
        print(f"更新了 {updated_count} 个指标的 required_images")
        
        # 保存更新后的文件
        backup_path = library_path.replace('.xlsx', '_backup.xlsx')
        if os.path.exists(library_path):
            # 创建备份
            import shutil
            shutil.copy(library_path, backup_path)
            print(f"创建备份: {backup_path}")
        
        # 保存更新后的文件
        df.to_excel(library_path, index=False)
        print(f"更新后的指标库已保存: {library_path}")
        
        # 显示示例
        print("\n示例（前5个指标）:")
        print(df[['metric name', 'required_images']].head())
        
        return True
        
    except Exception as e:
        print(f"更新失败: {e}")
        return False

def create_example_with_required_images():
    """创建包含required_images的示例代码"""
    
    example_code = '''"""
指标名称: Green Space Coverage Index (GSCI)
功能: 计算绿色空间覆盖率
required_images: ['semantic_map']
"""

import cv2
import numpy as np

def calculate(vision_result):
    """
    计算绿色空间覆盖率
    
    Required: semantic_map
    """
    try:
        # 检查输入
        if 'images' not in vision_result:
            return None
        
        # 获取语义分割图
        semantic_data = vision_result['images'].get('semantic_map')
        if semantic_data is None:
            return None
        
        # 转换为图像
        nparr = np.frombuffer(semantic_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 假设绿色类别的颜色是 [4, 250, 7]（草地）
        green_color = np.array([4, 250, 7])
        
        # 创建掩码
        mask = cv2.inRange(img, green_color - 10, green_color + 10)
        
        # 计算绿色像素比例
        green_pixels = np.sum(mask > 0)
        total_pixels = mask.shape[0] * mask.shape[1]
        
        if total_pixels == 0:
            return 0.0
        
        coverage = green_pixels / total_pixels
        
        return float(coverage)
        
    except Exception as e:
        print(f"计算GSCI时出错: {e}")
        return None
'''
    
    with open('example_metric_with_required.py', 'w', encoding='utf-8') as f:
        f.write(example_code)
    
    print("\n创建了示例代码文件: example_metric_with_required.py")
    print("该文件展示了如何在代码中声明required_images")

def display_image_types():
    """显示所有可用的图像类型"""
    
    print("\n可用的图像类型:")
    print("="*50)
    
    image_types = {
        'semantic_map': '语义分割图 - 每个像素标记了语义类别',
        'depth_map': '深度图 - 显示场景深度信息',
        'fmb_map': '前中后景图 - 0=前景, 1=中景, 2=背景',
        'foreground_map': '前景掩码 - 255=前景, 0=非前景',
        'middleground_map': '中景掩码',
        'background_map': '背景掩码',
        'openness_map': '开放度图 - 显示空间开放程度',
        'original': '调整尺寸后的原始图片'
    }
    
    for img_type, description in image_types.items():
        print(f"- {img_type}: {description}")
    
    print("\n在required_images中可以:")
    print("1. 指定单个图像: 'semantic_map'")
    print("2. 指定多个图像: 'semantic_map, depth_map'")
    print("3. 指定替代选项: 'fmb_map|foreground_map' (使用|分隔)")

if __name__ == "__main__":
    print("指标库required_images更新工具")
    print("="*50)
    
    # 更新指标库
    success = update_metrics_library()
    
    if success:
        # 创建示例代码
        create_example_with_required_images()
        
        # 显示图像类型说明
        display_image_types()
        
        print("\n下一步:")
        print("1. 检查更新后的 data/library_metrics.xlsx")
        print("2. 在指标代码中使用正确的图像类型")
        print("3. 重启应用以使用新的配置")
    else:
        print("\n更新失败，请检查错误信息")

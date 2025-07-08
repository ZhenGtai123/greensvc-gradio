"""
指标库管理模块
管理指标定义和计算代码
"""

import os
import pandas as pd
import json
from typing import List, Dict, Optional, Any
import shutil
import logging

logger = logging.getLogger(__name__)

class MetricsManager:
    """指标管理器"""
    
    def __init__(self, metrics_library_path: str, metrics_code_dir: str):
        self.metrics_library_path = metrics_library_path
        self.metrics_code_dir = metrics_code_dir
        
        # 确保代码目录存在
        os.makedirs(self.metrics_code_dir, exist_ok=True)
        
        # 加载指标库
        self.metrics_df = None
        self.metrics_dict = {}
        self.load_metrics()
    
    def load_metrics(self) -> pd.DataFrame:
        """加载指标库"""
        try:
            if os.path.exists(self.metrics_library_path):
                self.metrics_df = pd.read_excel(self.metrics_library_path)
                logger.info(f"成功加载 {len(self.metrics_df)} 个指标")
            else:
                # 如果文件不存在，尝试从JSON创建
                json_path = self.metrics_library_path.replace('.xlsx', '.json')
                if os.path.exists(json_path):
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    self.metrics_df = pd.DataFrame(data)
                    self.metrics_df.to_excel(self.metrics_library_path, index=False)
                    logger.info(f"从JSON创建指标库，包含 {len(self.metrics_df)} 个指标")
                else:
                    # 创建空的DataFrame
                    self.metrics_df = pd.DataFrame(columns=[
                        'metric name', 'Primary Category', 'Secondary Attribute',
                        'Standard Range', 'Unit', 'Parameter Definition',
                        'Data Input', 'Calculation Method', 'Professional Interpretation'
                    ])
                    logger.warning("未找到指标库文件，创建空指标库")
            
            # 构建指标字典
            self._build_metrics_dict()
            
            return self.metrics_df
            
        except Exception as e:
            logger.error(f"加载指标库失败: {e}")
            self.metrics_df = pd.DataFrame()
            return self.metrics_df
    
    def _build_metrics_dict(self):
        """构建指标字典以便快速查找"""
        self.metrics_dict = {}
        if self.metrics_df is not None and not self.metrics_df.empty:
            for _, row in self.metrics_df.iterrows():
                metric_name = row.get('metric name', '')
                if metric_name:
                    self.metrics_dict[metric_name] = row.to_dict()
    
    def reload_metrics(self):
        """重新加载指标库"""
        return self.load_metrics()
    
    def get_all_metrics(self) -> List[Dict]:
        """获取所有指标"""
        if self.metrics_df is not None and not self.metrics_df.empty:
            return self.metrics_df.to_dict('records')
        return []
    
    def get_metric_by_name(self, metric_name: str) -> Optional[Dict]:
        """根据名称获取指标"""
        return self.metrics_dict.get(metric_name)
    
    def get_metrics_by_category(self, category: str) -> List[Dict]:
        """根据类别获取指标"""
        if self.metrics_df is not None and not self.metrics_df.empty:
            filtered = self.metrics_df[
                self.metrics_df['Primary Category'].str.contains(category, case=False, na=False)
            ]
            return filtered.to_dict('records')
        return []
    
    def add_metric(self, metric_data: Dict) -> bool:
        """添加新指标"""
        try:
            # 验证必要字段
            required_fields = ['metric name', 'Primary Category', 'Calculation Method']
            for field in required_fields:
                if field not in metric_data:
                    logger.error(f"缺少必要字段: {field}")
                    return False
            
            # 添加到DataFrame
            new_df = pd.DataFrame([metric_data])
            self.metrics_df = pd.concat([self.metrics_df, new_df], ignore_index=True)
            
            # 保存到文件
            self.metrics_df.to_excel(self.metrics_library_path, index=False)
            
            # 更新字典
            self._build_metrics_dict()
            
            logger.info(f"成功添加指标: {metric_data['metric name']}")
            return True
            
        except Exception as e:
            logger.error(f"添加指标失败: {e}")
            return False
    
    def update_metric(self, metric_name: str, updates: Dict) -> bool:
        """更新指标"""
        try:
            if self.metrics_df is None or self.metrics_df.empty:
                return False
            
            # 找到指标索引
            mask = self.metrics_df['metric name'] == metric_name
            if not mask.any():
                logger.error(f"未找到指标: {metric_name}")
                return False
            
            # 更新数据
            for key, value in updates.items():
                self.metrics_df.loc[mask, key] = value
            
            # 保存到文件
            self.metrics_df.to_excel(self.metrics_library_path, index=False)
            
            # 更新字典
            self._build_metrics_dict()
            
            logger.info(f"成功更新指标: {metric_name}")
            return True
            
        except Exception as e:
            logger.error(f"更新指标失败: {e}")
            return False
    
    def delete_metric(self, metric_name: str) -> bool:
        """删除指标"""
        try:
            if self.metrics_df is None or self.metrics_df.empty:
                return False
            
            # 删除指标
            initial_len = len(self.metrics_df)
            self.metrics_df = self.metrics_df[self.metrics_df['metric name'] != metric_name]
            
            if len(self.metrics_df) == initial_len:
                logger.error(f"未找到指标: {metric_name}")
                return False
            
            # 保存到文件
            self.metrics_df.to_excel(self.metrics_library_path, index=False)
            
            # 更新字典
            self._build_metrics_dict()
            
            # 删除对应的代码文件
            code_path = self.get_metric_code_path(metric_name)
            if os.path.exists(code_path):
                os.remove(code_path)
                logger.info(f"删除代码文件: {code_path}")
            
            logger.info(f"成功删除指标: {metric_name}")
            return True
            
        except Exception as e:
            logger.error(f"删除指标失败: {e}")
            return False
    
    def get_metric_code_path(self, metric_name: str) -> str:
        """获取指标代码文件路径"""
        # 清理文件名中的特殊字符
        safe_name = metric_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        return os.path.join(self.metrics_code_dir, f"{safe_name}.py")
    
    def save_metric_code(self, metric_name: str, code: str) -> bool:
        """保存指标计算代码"""
        try:
            code_path = self.get_metric_code_path(metric_name)
            
            # 检查代码是否包含必要的函数
            has_calc_func = False
            for func_name in ['calculate', 'calculate_metric', 'calc', 'main']:
                if f'def {func_name}(' in code:
                    has_calc_func = True
                    break
            
            if not has_calc_func:
                logger.error(f"代码中未找到必要的计算函数（calculate, calculate_metric, calc, 或 main）")
                return False
            
            # 如果代码已经包含导入语句，不添加额外的头部
            if 'import' in code[:200]:  # 检查开头部分
                full_code = code
            else:
                # 添加标准头部
                full_code = f'''"""
指标计算代码: {metric_name}
自动生成的代码文件
"""

# 导入必要的库
import cv2
import numpy as np
import pandas as pd
from typing import Dict, Any, Union

{code}
'''
            
            # 保存代码
            with open(code_path, 'w', encoding='utf-8') as f:
                f.write(full_code)
            
            logger.info(f"成功保存指标代码: {code_path}")
            return True
            
        except Exception as e:
            logger.error(f"保存指标代码失败: {e}")
            return False
    
    def load_metric_code(self, metric_name: str) -> Optional[str]:
        """加载指标计算代码"""
        try:
            code_path = self.get_metric_code_path(metric_name)
            
            if os.path.exists(code_path):
                with open(code_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                logger.warning(f"未找到指标代码文件: {code_path}")
                return None
                
        except Exception as e:
            logger.error(f"加载指标代码失败: {e}")
            return None
    
    def has_metric_code(self, metric_name: str) -> bool:
        """检查指标是否有对应的计算代码"""
        code_path = self.get_metric_code_path(metric_name)
        return os.path.exists(code_path)
    
    def export_metrics(self, output_path: str, include_code: bool = True) -> bool:
        """导出指标库和代码"""
        try:
            # 导出Excel
            self.metrics_df.to_excel(output_path, index=False)
            
            if include_code:
                # 创建代码目录
                code_export_dir = output_path.replace('.xlsx', '_code')
                os.makedirs(code_export_dir, exist_ok=True)
                
                # 复制所有代码文件
                for metric in self.get_all_metrics():
                    metric_name = metric.get('metric name')
                    if metric_name and self.has_metric_code(metric_name):
                        src = self.get_metric_code_path(metric_name)
                        dst = os.path.join(code_export_dir, os.path.basename(src))
                        shutil.copy2(src, dst)
            
            logger.info(f"成功导出指标库到: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"导出指标库失败: {e}")
            return False
    
    def import_metrics(self, excel_path: str, code_dir: Optional[str] = None) -> bool:
        """导入指标库和代码"""
        try:
            # 导入Excel
            new_df = pd.read_excel(excel_path)
            
            # 合并或替换现有数据
            self.metrics_df = new_df
            self.metrics_df.to_excel(self.metrics_library_path, index=False)
            
            # 导入代码文件
            if code_dir and os.path.exists(code_dir):
                for filename in os.listdir(code_dir):
                    if filename.endswith('.py'):
                        src = os.path.join(code_dir, filename)
                        dst = os.path.join(self.metrics_code_dir, filename)
                        shutil.copy2(src, dst)
            
            # 重新加载
            self._build_metrics_dict()
            
            logger.info(f"成功导入指标库: {excel_path}")
            return True
            
        except Exception as e:
            logger.error(f"导入指标库失败: {e}")
            return False
    
    def validate_metrics(self) -> Dict[str, List[str]]:
        """验证指标库的完整性"""
        issues = {
            'missing_code': [],
            'missing_fields': [],
            'invalid_data': []
        }
        
        required_fields = ['metric name', 'Primary Category', 'Calculation Method']
        
        for metric in self.get_all_metrics():
            metric_name = metric.get('metric name', '')
            
            # 检查必要字段
            for field in required_fields:
                if not metric.get(field):
                    issues['missing_fields'].append(f"{metric_name}: 缺少 {field}")
            
            # 检查代码文件
            if metric_name and not self.has_metric_code(metric_name):
                issues['missing_code'].append(metric_name)
        
        return issues
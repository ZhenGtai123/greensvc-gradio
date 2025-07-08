"""
报告生成模块
生成包含图表、分析和热力图的综合报告
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple
import folium
from folium.plugins import HeatMap
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import tempfile
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

class ReportGenerator:
    """报告生成器"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.temp_dir = tempfile.mkdtemp(prefix='report_gen_')
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 设置中文字体（如果需要）
        self._setup_fonts()
        
        # 设置绘图样式
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def _setup_fonts(self):
        """设置中文字体支持"""
        try:
            # 尝试注册中文字体
            font_paths = [
                '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
                'C:/Windows/Fonts/simhei.ttf',
                '/System/Library/Fonts/PingFang.ttc'
            ]
            
            for font_path in font_paths:
                if os.path.exists(font_path):
                    pdfmetrics.registerFont(TTFont('ChineseFont', font_path))
                    break
        except Exception as e:
            logger.warning(f"字体设置失败: {e}")
    
    def generate_report(self, metrics_results: pd.DataFrame, selected_metrics: List[Dict],
                       vision_results: Dict, gps_data: Optional[Dict] = None,
                       openai_key: Optional[str] = None) -> str:
        """
        生成综合分析报告
        
        Args:
            metrics_results: 指标计算结果
            selected_metrics: 选中的指标信息
            vision_results: 视觉分析结果
            gps_data: GPS数据（可选）
            openai_key: OpenAI API密钥（可选）
            
        Returns:
            报告文件路径
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_name = f"spatial_analysis_report_{timestamp}"
            report_dir = os.path.join(self.output_dir, report_name)
            os.makedirs(report_dir, exist_ok=True)
            
            # 生成各个组件
            charts = self._generate_charts(metrics_results, selected_metrics, report_dir)
            
            heatmap_path = None
            if gps_data and gps_data.get('all_have_gps'):
                heatmap_path = self._generate_heatmap(gps_data, metrics_results, report_dir)
            
            ai_analysis = None
            if openai_key:
                ai_analysis = self._generate_ai_analysis(
                    metrics_results, selected_metrics, openai_key
                )
            
            # 生成PDF报告
            pdf_path = self._generate_pdf_report(
                report_dir, metrics_results, selected_metrics,
                charts, heatmap_path, ai_analysis
            )
            
            # 生成HTML报告
            html_path = self._generate_html_report(
                report_dir, metrics_results, selected_metrics,
                charts, heatmap_path, ai_analysis
            )
            
            # 保存原始数据
            self._save_raw_data(report_dir, metrics_results, selected_metrics)
            
            logger.info(f"报告生成成功: {pdf_path}")
            return pdf_path
            
        except Exception as e:
            logger.error(f"生成报告失败: {e}")
            raise
    
    def _generate_charts(self, metrics_results: pd.DataFrame, 
                        selected_metrics: List[Dict], report_dir: str) -> Dict[str, str]:
        """生成各种图表"""
        charts = {}
        
        try:
            # 1. 指标分布图
            if len(selected_metrics) > 0:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # 准备数据
                metric_names = [m['metric name'] for m in selected_metrics[:5]]  # 最多显示5个
                data_to_plot = []
                
                for metric_name in metric_names:
                    if metric_name in metrics_results.columns:
                        values = metrics_results[metric_name].dropna()
                        if len(values) > 0:
                            data_to_plot.append(values)
                
                if data_to_plot:
                    ax.boxplot(data_to_plot, labels=metric_names)
                    ax.set_xlabel('Metrics')
                    ax.set_ylabel('Values')
                    ax.set_title('Metric Distribution')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    
                    chart_path = os.path.join(report_dir, 'metric_distribution.png')
                    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                    charts['distribution'] = chart_path
                    plt.close()
            
            # 2. 相关性热图
            if len(metrics_results.columns) > 2:
                numeric_cols = metrics_results.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    correlation = metrics_results[numeric_cols].corr()
                    
                    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0,
                               square=True, linewidths=1, ax=ax)
                    ax.set_title('Metric Correlation Heatmap')
                    plt.tight_layout()
                    
                    chart_path = os.path.join(report_dir, 'correlation_heatmap.png')
                    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                    charts['correlation'] = chart_path
                    plt.close()
            
            # 3. 雷达图（如果有多个指标）
            if len(selected_metrics) >= 3:
                fig = self._create_radar_chart(metrics_results, selected_metrics[:8])
                if fig:
                    chart_path = os.path.join(report_dir, 'radar_chart.png')
                    fig.savefig(chart_path, dpi=300, bbox_inches='tight')
                    charts['radar'] = chart_path
                    plt.close()
            
            # 4. 时间序列图（如果有多个图片）
            if len(metrics_results) > 1:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                for i, metric in enumerate(selected_metrics[:5]):
                    metric_name = metric['metric name']
                    if metric_name in metrics_results.columns:
                        values = metrics_results[metric_name].values
                        ax.plot(range(len(values)), values, marker='o', label=metric_name)
                
                ax.set_xlabel('Image Index')
                ax.set_ylabel('Metric Value')
                ax.set_title('Metric Values Across Images')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                
                chart_path = os.path.join(report_dir, 'time_series.png')
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                charts['time_series'] = chart_path
                plt.close()
            
        except Exception as e:
            logger.error(f"生成图表失败: {e}")
        
        return charts
    
    def _create_radar_chart(self, metrics_results: pd.DataFrame, 
                           selected_metrics: List[Dict]) -> Optional[plt.Figure]:
        """创建雷达图"""
        try:
            # 准备数据
            categories = []
            values = []
            
            for metric in selected_metrics:
                metric_name = metric['metric name']
                if metric_name in metrics_results.columns:
                    mean_value = metrics_results[metric_name].mean()
                    if not np.isnan(mean_value):
                        categories.append(metric_name[:20] + '...' if len(metric_name) > 20 else metric_name)
                        # 标准化到0-1范围
                        min_val = metrics_results[metric_name].min()
                        max_val = metrics_results[metric_name].max()
                        if max_val > min_val:
                            normalized = (mean_value - min_val) / (max_val - min_val)
                        else:
                            normalized = 0.5
                        values.append(normalized)
            
            if len(categories) < 3:
                return None
            
            # 创建雷达图
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='polar')
            
            # 计算角度
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            values += values[:1]  # 闭合图形
            angles += angles[:1]
            
            # 绘制
            ax.plot(angles, values, 'o-', linewidth=2)
            ax.fill(angles, values, alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_ylim(0, 1)
            ax.set_title('Normalized Metric Values (Mean)', size=20, y=1.1)
            ax.grid(True)
            
            return fig
            
        except Exception as e:
            logger.error(f"创建雷达图失败: {e}")
            return None
    
    def _generate_heatmap(self, gps_data: Dict, metrics_results: pd.DataFrame, 
                         report_dir: str) -> Optional[str]:
        """生成空间热力图"""
        try:
            locations = gps_data.get('locations', [])
            if not locations:
                return None
            
            # 创建基础地图
            center_lat = np.mean([loc[0] for loc in locations])
            center_lon = np.mean([loc[1] for loc in locations])
            
            m = folium.Map(location=[center_lat, center_lon], zoom_start=15)
            
            # 添加热力图层
            # 可以根据某个指标的值来设置热力强度
            if not metrics_results.empty:
                # 使用第一个数值列作为强度
                numeric_cols = metrics_results.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    intensities = metrics_results[numeric_cols[0]].values[:len(locations)]
                    # 标准化强度值
                    intensities = (intensities - intensities.min()) / (intensities.max() - intensities.min() + 1e-8)
                    heat_data = [[loc[0], loc[1], intensity] 
                               for loc, intensity in zip(locations, intensities)]
                else:
                    heat_data = [[loc[0], loc[1], 1] for loc in locations]
            else:
                heat_data = [[loc[0], loc[1], 1] for loc in locations]
            
            HeatMap(heat_data).add_to(m)
            
            # 添加标记点
            for i, loc in enumerate(locations):
                folium.Marker(
                    location=[loc[0], loc[1]],
                    popup=f"Point {i+1}",
                    icon=folium.Icon(color='blue', icon='info-sign')
                ).add_to(m)
            
            # 保存地图
            map_path = os.path.join(report_dir, 'spatial_heatmap.html')
            m.save(map_path)
            
            # 同时生成静态图片（使用matplotlib）
            fig, ax = plt.subplots(figsize=(10, 8))
            
            lats = [loc[0] for loc in locations]
            lons = [loc[1] for loc in locations]
            
            # 创建散点图
            scatter = ax.scatter(lons, lats, c=intensities if 'intensities' in locals() else 'blue',
                               cmap='hot', s=200, alpha=0.6, edgecolors='black')
            
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_title('Spatial Distribution of Analysis Points')
            
            if 'intensities' in locals():
                plt.colorbar(scatter, ax=ax, label='Metric Intensity')
            
            plt.tight_layout()
            
            static_map_path = os.path.join(report_dir, 'spatial_distribution.png')
            plt.savefig(static_map_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return static_map_path
            
        except Exception as e:
            logger.error(f"生成热力图失败: {e}")
            return None
    
    def _generate_ai_analysis(self, metrics_results: pd.DataFrame, 
                             selected_metrics: List[Dict], openai_key: str) -> Optional[str]:
        """使用AI生成分析文本"""
        try:
            client = OpenAI(api_key=openai_key)
            
            # 准备数据摘要
            summary = {
                'total_images': len(metrics_results),
                'metrics': {}
            }
            
            for metric in selected_metrics:
                metric_name = metric['metric name']
                if metric_name in metrics_results.columns:
                    values = metrics_results[metric_name].dropna()
                    if len(values) > 0:
                        summary['metrics'][metric_name] = {
                            'mean': float(values.mean()),
                            'std': float(values.std()),
                            'min': float(values.min()),
                            'max': float(values.max()),
                            'interpretation': metric.get('Professional Interpretation', '')
                        }
            
            # 构建提示
            prompt = f"""
            请基于以下城市绿地空间视觉分析数据，生成专业的分析报告：
            
            分析了 {summary['total_images']} 张图片，计算了以下指标：
            
            {json.dumps(summary['metrics'], indent=2, ensure_ascii=False)}
            
            请从以下角度进行分析：
            1. 整体空间特征概述
            2. 各指标的含义和发现
            3. 指标间的关联性分析
            4. 空间设计建议
            5. 总结与展望
            
            请用专业但易懂的语言，生成约500字的分析报告。
            """
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "你是一位城市规划和景观设计专家，擅长空间视觉分析。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"AI分析生成失败: {e}")
            return None
    
    def _generate_pdf_report(self, report_dir: str, metrics_results: pd.DataFrame,
                            selected_metrics: List[Dict], charts: Dict[str, str],
                            heatmap_path: Optional[str], ai_analysis: Optional[str]) -> str:
        """生成PDF报告"""
        try:
            pdf_path = os.path.join(report_dir, 'analysis_report.pdf')
            doc = SimpleDocTemplate(pdf_path, pagesize=A4)
            
            # 获取样式
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Title'],
                fontSize=24,
                textColor=colors.HexColor('#1f77b4'),
                spaceAfter=30,
                alignment=TA_CENTER
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading1'],
                fontSize=16,
                textColor=colors.HexColor('#2ca02c'),
                spaceAfter=12
            )
            
            # 构建内容
            story = []
            
            # 标题页
            story.append(Paragraph("城市绿地空间视觉分析报告", title_style))
            story.append(Spacer(1, 0.5*inch))
            story.append(Paragraph(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
            story.append(PageBreak())
            
            # 摘要
            story.append(Paragraph("1. 分析摘要", heading_style))
            summary_text = f"""
            本次分析共处理了 {len(metrics_results)} 张图片，计算了 {len(selected_metrics)} 个空间视觉指标。
            通过深度学习模型进行语义分割和深度估计，结合专业的空间分析算法，
            全面评估了目标区域的空间特征。
            """
            story.append(Paragraph(summary_text, styles['Normal']))
            story.append(Spacer(1, 0.3*inch))
            
            # 指标结果表格
            story.append(Paragraph("2. 指标计算结果", heading_style))
            
            # 创建表格数据
            table_data = [['指标名称', '平均值', '标准差', '最小值', '最大值']]
            
            for metric in selected_metrics:
                metric_name = metric['metric name']
                if metric_name in metrics_results.columns:
                    values = metrics_results[metric_name].dropna()
                    if len(values) > 0:
                        table_data.append([
                            metric_name[:30] + '...' if len(metric_name) > 30 else metric_name,
                            f"{values.mean():.3f}",
                            f"{values.std():.3f}",
                            f"{values.min():.3f}",
                            f"{values.max():.3f}"
                        ])
            
            if len(table_data) > 1:
                t = Table(table_data)
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(t)
            
            story.append(PageBreak())
            
            # 图表展示
            story.append(Paragraph("3. 数据可视化", heading_style))
            
            for chart_name, chart_path in charts.items():
                if os.path.exists(chart_path):
                    img = RLImage(chart_path, width=5*inch, height=3.5*inch)
                    story.append(img)
                    story.append(Spacer(1, 0.2*inch))
            
            if heatmap_path and os.path.exists(heatmap_path):
                story.append(PageBreak())
                story.append(Paragraph("4. 空间分布", heading_style))
                img = RLImage(heatmap_path, width=5*inch, height=4*inch)
                story.append(img)
            
            # AI分析
            if ai_analysis:
                story.append(PageBreak())
                story.append(Paragraph("5. 智能分析", heading_style))
                
                # 将AI分析文本分段
                paragraphs = ai_analysis.split('\n\n')
                for para in paragraphs:
                    if para.strip():
                        story.append(Paragraph(para.strip(), styles['Normal']))
                        story.append(Spacer(1, 0.1*inch))
            
            # 构建PDF
            doc.build(story)
            
            return pdf_path
            
        except Exception as e:
            logger.error(f"生成PDF报告失败: {e}")
            # 返回一个简单的文本报告作为备选
            txt_path = os.path.join(report_dir, 'analysis_report.txt')
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("城市绿地空间视觉分析报告\n")
                f.write("="*50 + "\n\n")
                f.write(f"生成时间：{datetime.now()}\n\n")
                f.write("指标计算结果：\n")
                f.write(metrics_results.to_string())
            return txt_path
    
    def _generate_html_report(self, report_dir: str, metrics_results: pd.DataFrame,
                             selected_metrics: List[Dict], charts: Dict[str, str],
                             heatmap_path: Optional[str], ai_analysis: Optional[str]) -> str:
        """生成HTML报告"""
        try:
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>城市绿地空间视觉分析报告</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f4f4f4;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #1f77b4;
            text-align: center;
            margin-bottom: 30px;
        }}
        h2 {{
            color: #2ca02c;
            margin-top: 30px;
        }}
        .metric-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .metric-table th, .metric-table td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: center;
        }}
        .metric-table th {{
            background-color: #4CAF50;
            color: white;
        }}
        .metric-table tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .chart-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .chart-container img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .ai-analysis {{
            background-color: #e8f4f8;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .timestamp {{
            text-align: center;
            color: #666;
            font-style: italic;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>城市绿地空间视觉分析报告</h1>
        <p class="timestamp">生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>1. 分析概览</h2>
        <p>本次分析共处理了 <strong>{len(metrics_results)}</strong> 张图片，
        计算了 <strong>{len(selected_metrics)}</strong> 个空间视觉指标。</p>
        
        <h2>2. 指标统计结果</h2>
        <table class="metric-table">
            <thead>
                <tr>
                    <th>指标名称</th>
                    <th>平均值</th>
                    <th>标准差</th>
                    <th>最小值</th>
                    <th>最大值</th>
                </tr>
            </thead>
            <tbody>
"""
            
            # 添加指标数据
            for metric in selected_metrics:
                metric_name = metric['metric name']
                if metric_name in metrics_results.columns:
                    values = metrics_results[metric_name].dropna()
                    if len(values) > 0:
                        html_content += f"""
                <tr>
                    <td>{metric_name}</td>
                    <td>{values.mean():.3f}</td>
                    <td>{values.std():.3f}</td>
                    <td>{values.min():.3f}</td>
                    <td>{values.max():.3f}</td>
                </tr>
"""
            
            html_content += """
            </tbody>
        </table>
        
        <h2>3. 数据可视化</h2>
"""
            
            # 添加图表
            for chart_name, chart_path in charts.items():
                if os.path.exists(chart_path):
                    chart_filename = os.path.basename(chart_path)
                    html_content += f"""
        <div class="chart-container">
            <img src="{chart_filename}" alt="{chart_name}">
        </div>
"""
            
            # 添加AI分析
            if ai_analysis:
                html_content += f"""
        <h2>4. 智能分析</h2>
        <div class="ai-analysis">
            {ai_analysis.replace(chr(10), '<br>')}
        </div>
"""
            
            html_content += """
    </div>
</body>
</html>
"""
            
            # 保存HTML文件
            html_path = os.path.join(report_dir, 'analysis_report.html')
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return html_path
            
        except Exception as e:
            logger.error(f"生成HTML报告失败: {e}")
            return ""
    
    def _save_raw_data(self, report_dir: str, metrics_results: pd.DataFrame,
                      selected_metrics: List[Dict]):
        """保存原始数据"""
        try:
            # 保存指标结果
            metrics_results.to_csv(os.path.join(report_dir, 'metrics_results.csv'), index=False)
            metrics_results.to_excel(os.path.join(report_dir, 'metrics_results.xlsx'), index=False)
            
            # 保存选中的指标信息
            with open(os.path.join(report_dir, 'selected_metrics.json'), 'w', encoding='utf-8') as f:
                json.dump(selected_metrics, f, ensure_ascii=False, indent=2)
            
        except Exception as e:
            logger.error(f"保存原始数据失败: {e}")
    
    def cleanup(self):
        """清理临时文件"""
        try:
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info(f"清理临时目录: {self.temp_dir}")
        except Exception as e:
            logger.error(f"清理失败: {e}")

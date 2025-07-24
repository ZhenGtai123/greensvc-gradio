"""
Report Generation Module
Generates comprehensive reports with charts, analysis and heatmaps
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
    """Report Generator Class"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.temp_dir = tempfile.mkdtemp(prefix='report_gen_')
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def generate_report(self, metrics_results: pd.DataFrame, selected_metrics: List[Dict],
                       vision_results: Dict, gps_data: Optional[Dict] = None,
                       openai_key: Optional[str] = None) -> str:
        """
        Generate comprehensive analysis report
        
        Args:
            metrics_results: Metric calculation results
            selected_metrics: Selected metric information
            vision_results: Vision analysis results
            gps_data: GPS data (optional)
            openai_key: OpenAI API key (optional)
            
        Returns:
            Report file path
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_name = f"spatial_analysis_report_{timestamp}"
            report_dir = os.path.join(self.output_dir, report_name)
            os.makedirs(report_dir, exist_ok=True)
            
            # Generate components
            charts = self._generate_charts(metrics_results, selected_metrics, report_dir)
            
            heatmap_path = None
            if gps_data and gps_data.get('all_have_gps'):
                heatmap_path = self._generate_heatmap(gps_data, metrics_results, report_dir)
            
            ai_analysis = None
            if openai_key:
                ai_analysis = self._generate_ai_analysis(
                    metrics_results, selected_metrics, vision_results, openai_key
                )
            
            # Generate PDF report
            pdf_path = self._generate_pdf_report(
                report_dir, metrics_results, selected_metrics,
                charts, heatmap_path, ai_analysis
            )
            
            # Generate HTML report
            html_path = self._generate_html_report(
                report_dir, metrics_results, selected_metrics,
                charts, heatmap_path, ai_analysis
            )
            
            # Save raw data
            self._save_raw_data(report_dir, metrics_results, selected_metrics)
            
            logger.info(f"Report generated successfully: {pdf_path}")
            return pdf_path
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            raise
    
    def _generate_charts(self, metrics_results: pd.DataFrame, 
                        selected_metrics: List[Dict], report_dir: str) -> Dict[str, str]:
        """Generate various charts"""
        charts = {}
        
        try:
            # 1. Metric Distribution Chart
            if len(selected_metrics) > 0:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Prepare data
                metric_names = [m['metric name'] for m in selected_metrics[:5]]  # Show max 5
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
                    ax.set_title('Metric Distribution Analysis')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    
                    chart_path = os.path.join(report_dir, 'metric_distribution.png')
                    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                    charts['distribution'] = chart_path
                    plt.close()
            
            # 2. Correlation Heatmap
            if len(metrics_results.columns) > 2:
                numeric_cols = metrics_results.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    correlation = metrics_results[numeric_cols].corr()
                    
                    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0,
                               square=True, linewidths=1, ax=ax)
                    ax.set_title('Metric Correlation Analysis')
                    plt.tight_layout()
                    
                    chart_path = os.path.join(report_dir, 'correlation_heatmap.png')
                    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                    charts['correlation'] = chart_path
                    plt.close()
            
            # 3. Radar Chart
            if len(selected_metrics) >= 3:
                fig = self._create_radar_chart(metrics_results, selected_metrics[:8])
                if fig:
                    chart_path = os.path.join(report_dir, 'radar_chart.png')
                    fig.savefig(chart_path, dpi=300, bbox_inches='tight')
                    charts['radar'] = chart_path
                    plt.close()
            
            # 4. Trend Analysis
            if len(metrics_results) > 1:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                for i, metric in enumerate(selected_metrics[:5]):
                    metric_name = metric['metric name']
                    if metric_name in metrics_results.columns:
                        values = metrics_results[metric_name].values
                        ax.plot(range(len(values)), values, marker='o', label=metric_name)
                
                ax.set_xlabel('Image Index')
                ax.set_ylabel('Metric Value')
                ax.set_title('Metric Trend Analysis')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                
                chart_path = os.path.join(report_dir, 'trend_analysis.png')
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                charts['trend'] = chart_path
                plt.close()
            
        except Exception as e:
            logger.error(f"Failed to generate charts: {e}")
        
        return charts
    
    def _create_radar_chart(self, metrics_results: pd.DataFrame, 
                           selected_metrics: List[Dict]) -> Optional[plt.Figure]:
        """Create radar chart"""
        try:
            # Prepare data
            categories = []
            values = []
            
            for metric in selected_metrics:
                metric_name = metric['metric name']
                if metric_name in metrics_results.columns:
                    mean_value = metrics_results[metric_name].mean()
                    if not np.isnan(mean_value):
                        categories.append(metric_name[:20] + '...' if len(metric_name) > 20 else metric_name)
                        # Normalize to 0-1 range
                        min_val = metrics_results[metric_name].min()
                        max_val = metrics_results[metric_name].max()
                        if max_val > min_val:
                            normalized = (mean_value - min_val) / (max_val - min_val)
                        else:
                            normalized = 0.5
                        values.append(normalized)
            
            if len(categories) < 3:
                return None
            
            # Create radar chart
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='polar')
            
            # Calculate angles
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            values += values[:1]  # Close the plot
            angles += angles[:1]
            
            # Plot
            ax.plot(angles, values, 'o-', linewidth=2)
            ax.fill(angles, values, alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_ylim(0, 1)
            ax.set_title('Normalized Metric Values (Mean)', size=20, y=1.1)
            ax.grid(True)
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create radar chart: {e}")
            return None
    
    def _generate_heatmap(self, gps_data: Dict, metrics_results: pd.DataFrame, 
                         report_dir: str) -> Optional[str]:
        """Generate spatial heatmap"""
        try:
            locations = gps_data.get('locations', [])
            if not locations:
                return None
            
            # Create base map
            center_lat = np.mean([loc[0] for loc in locations])
            center_lon = np.mean([loc[1] for loc in locations])
            
            m = folium.Map(location=[center_lat, center_lon], zoom_start=15)
            
            # Add heatmap layer
            if not metrics_results.empty:
                numeric_cols = metrics_results.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    intensities = metrics_results[numeric_cols[0]].values[:len(locations)]
                    # Normalize intensities
                    intensities = (intensities - intensities.min()) / (intensities.max() - intensities.min() + 1e-8)
                    heat_data = [[loc[0], loc[1], intensity] 
                               for loc, intensity in zip(locations, intensities)]
                else:
                    heat_data = [[loc[0], loc[1], 1] for loc in locations]
            else:
                heat_data = [[loc[0], loc[1], 1] for loc in locations]
            
            HeatMap(heat_data).add_to(m)
            
            # Add markers
            for i, loc in enumerate(locations):
                folium.Marker(
                    location=[loc[0], loc[1]],
                    popup=f"Point {i+1}",
                    icon=folium.Icon(color='blue', icon='info-sign')
                ).add_to(m)
            
            # Save map
            map_path = os.path.join(report_dir, 'spatial_heatmap.html')
            m.save(map_path)
            
            # Generate static image
            fig, ax = plt.subplots(figsize=(10, 8))
            
            lats = [loc[0] for loc in locations]
            lons = [loc[1] for loc in locations]
            
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
            logger.error(f"Failed to generate heatmap: {e}")
            return None
    
    def _generate_ai_analysis(self, metrics_results: pd.DataFrame, 
                             selected_metrics: List[Dict], 
                             vision_results: Dict,
                             openai_key: str) -> Optional[Dict]:
        """Generate AI-powered analysis using OpenAI API"""
        try:
            client = OpenAI(api_key=openai_key)
            
            # Prepare comprehensive data summary
            summary = {
                'total_images': len(metrics_results),
                'metrics': {},
                'spatial_characteristics': self._analyze_spatial_characteristics(vision_results)
            }
            
            # Compile metric statistics
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
                            'cv': float(values.std() / values.mean()) if values.mean() != 0 else 0,
                            'category': metric.get('Primary Category', ''),
                            'unit': metric.get('Unit', ''),
                            'interpretation': metric.get('Professional Interpretation', '')
                        }
            
            # Generate different analysis sections
            analyses = {}
            
            # 1. Executive Summary
            exec_summary_prompt = f"""
            As an urban planning and landscape design expert, provide a concise executive summary 
            of the spatial visual analysis results. Focus on key findings and their implications.
            
            Data analyzed: {summary['total_images']} images
            Metrics calculated: {list(summary['metrics'].keys())}
            
            Key statistics:
            {json.dumps(summary['metrics'], indent=2)}
            
            Provide a 150-word executive summary highlighting:
            1. Most significant findings
            2. Overall spatial quality assessment
            3. Key recommendations
            """
            
            exec_response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert in urban spatial analysis and landscape architecture."},
                    {"role": "user", "content": exec_summary_prompt}
                ],
                max_tokens=300,
                temperature=0.7
            )
            analyses['executive_summary'] = exec_response.choices[0].message.content
            
            # 2. Detailed Metric Analysis
            metric_analysis_prompt = f"""
            Provide a detailed analysis of each spatial metric, explaining what the values indicate
            about the urban green space quality.
            
            Metrics data:
            {json.dumps(summary['metrics'], indent=2)}
            
            For each metric, explain:
            1. What the values indicate about spatial quality
            2. Whether the values are within optimal ranges
            3. Specific design implications
            
            Format as clear paragraphs, one for each metric.
            """
            
            metric_response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert in spatial metrics and urban design evaluation."},
                    {"role": "user", "content": metric_analysis_prompt}
                ],
                max_tokens=800,
                temperature=0.6
            )
            analyses['metric_analysis'] = metric_response.choices[0].message.content
            
            # 3. Spatial Recommendations
            recommendations_prompt = f"""
            Based on the spatial analysis results, provide specific design recommendations
            to improve the urban green space quality.
            
            Current metrics:
            {json.dumps(summary['metrics'], indent=2)}
            
            Spatial characteristics:
            {json.dumps(summary['spatial_characteristics'], indent=2)}
            
            Provide 5-7 actionable recommendations with rationale, focusing on:
            - Spatial configuration improvements
            - Visual quality enhancement
            - User experience optimization
            - Ecological considerations
            """
            
            rec_response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a landscape architect specializing in urban green space design."},
                    {"role": "user", "content": recommendations_prompt}
                ],
                max_tokens=600,
                temperature=0.7
            )
            analyses['recommendations'] = rec_response.choices[0].message.content
            
            # 4. Comparative Analysis (if multiple images)
            if len(metrics_results) > 1:
                comparison_prompt = f"""
                Analyze the variation in metrics across the {len(metrics_results)} analyzed images.
                
                Identify:
                1. Patterns in spatial quality variation
                2. Best and worst performing areas
                3. Factors contributing to quality differences
                
                Metric statistics:
                {json.dumps(summary['metrics'], indent=2)}
                """
                
                comp_response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are an expert in spatial data analysis and pattern recognition."},
                        {"role": "user", "content": comparison_prompt}
                    ],
                    max_tokens=400,
                    temperature=0.6
                )
                analyses['comparative_analysis'] = comp_response.choices[0].message.content
            
            return analyses
            
        except Exception as e:
            logger.error(f"AI analysis generation failed: {e}")
            return None
    
    def _analyze_spatial_characteristics(self, vision_results: Dict) -> Dict:
        """Analyze spatial characteristics from vision results"""
        characteristics = {
            'dominant_elements': [],
            'spatial_composition': {},
            'detected_features': []
        }
        
        try:
            # Aggregate statistics across all images
            if vision_results:
                # Get sample result for structure
                sample_result = next(iter(vision_results.values()))
                if 'statistics' in sample_result:
                    stats = sample_result['statistics']
                    if 'class_statistics' in stats:
                        # Find dominant classes
                        class_stats = stats['class_statistics']
                        sorted_classes = sorted(class_stats.items(), 
                                              key=lambda x: x[1].get('percentage', 0), 
                                              reverse=True)
                        characteristics['dominant_elements'] = [
                            {'class': cls, 'percentage': data.get('percentage', 0)}
                            for cls, data in sorted_classes[:5]
                        ]
                    
                    if 'fmb_statistics' in stats:
                        characteristics['spatial_composition'] = stats['fmb_statistics']
        
        except Exception as e:
            logger.error(f"Failed to analyze spatial characteristics: {e}")
        
        return characteristics
    
    def _generate_pdf_report(self, report_dir: str, metrics_results: pd.DataFrame,
                            selected_metrics: List[Dict], charts: Dict[str, str],
                            heatmap_path: Optional[str], ai_analysis: Optional[Dict]) -> str:
        """Generate PDF report"""
        try:
            pdf_path = os.path.join(report_dir, 'analysis_report.pdf')
            doc = SimpleDocTemplate(pdf_path, pagesize=A4)
            
            # Get styles
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
            
            subheading_style = ParagraphStyle(
                'CustomSubheading',
                parent=styles['Heading2'],
                fontSize=14,
                textColor=colors.HexColor('#ff7f0e'),
                spaceAfter=10
            )
            
            # Build content
            story = []
            
            # Title page
            story.append(Paragraph("Urban Green Space Visual Analysis Report", title_style))
            story.append(Spacer(1, 0.5*inch))
            story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
            story.append(PageBreak())
            
            # Executive Summary (if AI analysis available)
            if ai_analysis and 'executive_summary' in ai_analysis:
                story.append(Paragraph("Executive Summary", heading_style))
                story.append(Paragraph(ai_analysis['executive_summary'], styles['Normal']))
                story.append(Spacer(1, 0.3*inch))
            
            # Analysis Overview
            story.append(Paragraph("1. Analysis Overview", heading_style))
            overview_text = f"""
            This report presents a comprehensive spatial visual analysis of {len(metrics_results)} images 
            using {len(selected_metrics)} spatial metrics. The analysis employs advanced computer vision 
            techniques including semantic segmentation and depth estimation to evaluate spatial characteristics 
            and visual quality of urban green spaces.
            """
            story.append(Paragraph(overview_text.strip(), styles['Normal']))
            story.append(Spacer(1, 0.3*inch))
            
            # Metric Results Table
            story.append(Paragraph("2. Metric Calculation Results", heading_style))
            
            # Create summary statistics table
            table_data = [['Metric Name', 'Mean', 'Std Dev', 'Min', 'Max', 'CV']]
            
            for metric in selected_metrics:
                metric_name = metric['metric name']
                if metric_name in metrics_results.columns:
                    values = metrics_results[metric_name].dropna()
                    if len(values) > 0:
                        mean_val = values.mean()
                        std_val = values.std()
                        cv = std_val / mean_val if mean_val != 0 else 0
                        table_data.append([
                            metric_name[:30] + '...' if len(metric_name) > 30 else metric_name,
                            f"{mean_val:.3f}",
                            f"{std_val:.3f}",
                            f"{values.min():.3f}",
                            f"{values.max():.3f}",
                            f"{cv:.3f}"
                        ])
            
            if len(table_data) > 1:
                t = Table(table_data)
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F2F2F2')),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(t)
            
            story.append(PageBreak())
            
            # Detailed Metric Analysis (if AI analysis available)
            if ai_analysis and 'metric_analysis' in ai_analysis:
                story.append(Paragraph("3. Detailed Metric Analysis", heading_style))
                
                # Split AI analysis into paragraphs
                analysis_paragraphs = ai_analysis['metric_analysis'].split('\n\n')
                for para in analysis_paragraphs:
                    if para.strip():
                        story.append(Paragraph(para.strip(), styles['Normal']))
                        story.append(Spacer(1, 0.1*inch))
                
                story.append(PageBreak())
            
            # Data Visualization
            story.append(Paragraph("4. Data Visualization", heading_style))
            
            # Add charts with descriptions
            chart_descriptions = {
                'distribution': "Distribution of metric values across all analyzed images",
                'correlation': "Correlation analysis between different spatial metrics",
                'radar': "Normalized metric values showing relative performance",
                'trend': "Trend analysis showing metric variations across images"
            }
            
            for chart_name, chart_path in charts.items():
                if os.path.exists(chart_path):
                    story.append(Paragraph(chart_descriptions.get(chart_name, ""), styles['Normal']))
                    img = RLImage(chart_path, width=5*inch, height=3.5*inch)
                    story.append(img)
                    story.append(Spacer(1, 0.3*inch))
            
            # Spatial Distribution (if available)
            if heatmap_path and os.path.exists(heatmap_path):
                story.append(PageBreak())
                story.append(Paragraph("5. Spatial Distribution Analysis", heading_style))
                story.append(Paragraph(
                    "The following map shows the spatial distribution of analysis points with intensity based on metric values.",
                    styles['Normal']
                ))
                img = RLImage(heatmap_path, width=5*inch, height=4*inch)
                story.append(img)
            
            # Recommendations (if AI analysis available)
            if ai_analysis and 'recommendations' in ai_analysis:
                story.append(PageBreak())
                story.append(Paragraph("6. Design Recommendations", heading_style))
                
                rec_paragraphs = ai_analysis['recommendations'].split('\n\n')
                for para in rec_paragraphs:
                    if para.strip():
                        story.append(Paragraph(para.strip(), styles['Normal']))
                        story.append(Spacer(1, 0.1*inch))
            
            # Comparative Analysis (if available)
            if ai_analysis and 'comparative_analysis' in ai_analysis:
                story.append(PageBreak())
                story.append(Paragraph("7. Comparative Analysis", heading_style))
                story.append(Paragraph(ai_analysis['comparative_analysis'], styles['Normal']))
            
            # Methodology
            story.append(PageBreak())
            story.append(Paragraph("Appendix: Methodology", heading_style))
            methodology_text = """
            This analysis employs state-of-the-art computer vision techniques:
            
            • Semantic Segmentation: Deep learning models identify and classify different elements in the urban landscape
            • Depth Estimation: Monocular depth estimation provides spatial depth information
            • Spatial Metrics: Various quantitative metrics evaluate spatial characteristics including openness, complexity, and visual quality
            • Statistical Analysis: Comprehensive statistical analysis identifies patterns and correlations
            
            All metrics are calculated based on established urban design and landscape architecture principles.
            """
            story.append(Paragraph(methodology_text, styles['Normal']))
            
            # Build PDF
            doc.build(story)
            
            return pdf_path
            
        except Exception as e:
            logger.error(f"Failed to generate PDF report: {e}")
            # Return a simple text report as fallback
            txt_path = os.path.join(report_dir, 'analysis_report.txt')
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("Urban Green Space Visual Analysis Report\n")
                f.write("="*50 + "\n\n")
                f.write(f"Generated: {datetime.now()}\n\n")
                f.write("Metric Calculation Results:\n")
                f.write(metrics_results.to_string())
            return txt_path
    
    def _generate_html_report(self, report_dir: str, metrics_results: pd.DataFrame,
                             selected_metrics: List[Dict], charts: Dict[str, str],
                             heatmap_path: Optional[str], ai_analysis: Optional[Dict]) -> str:
        """Generate HTML report"""
        try:
            # Prepare AI analysis sections
            ai_sections = ""
            if ai_analysis:
                if 'executive_summary' in ai_analysis:
                    ai_sections += f"""
                <div class="section">
                    <h2>Executive Summary</h2>
                    <div class="ai-analysis">
                        {ai_analysis['executive_summary'].replace(chr(10), '<br>')}
                    </div>
                </div>
                """
                
                if 'metric_analysis' in ai_analysis:
                    ai_sections += f"""
                <div class="section">
                    <h2>Detailed Metric Analysis</h2>
                    <div class="ai-analysis">
                        {ai_analysis['metric_analysis'].replace(chr(10), '<br><br>')}
                    </div>
                </div>
                """
                
                if 'recommendations' in ai_analysis:
                    ai_sections += f"""
                <div class="section">
                    <h2>Design Recommendations</h2>
                    <div class="ai-analysis recommendations">
                        {ai_analysis['recommendations'].replace(chr(10), '<br><br>')}
                    </div>
                </div>
                """
                
                if 'comparative_analysis' in ai_analysis:
                    ai_sections += f"""
                <div class="section">
                    <h2>Comparative Analysis</h2>
                    <div class="ai-analysis">
                        {ai_analysis['comparative_analysis'].replace(chr(10), '<br><br>')}
                    </div>
                </div>
                """
            
            html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Urban Green Space Visual Analysis Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #1f77b4;
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.5em;
        }}
        h2 {{
            color: #2ca02c;
            margin-top: 40px;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 10px;
        }}
        h3 {{
            color: #ff7f0e;
            margin-top: 25px;
        }}
        .timestamp {{
            text-align: center;
            color: #666;
            font-style: italic;
            margin-bottom: 40px;
        }}
        .metric-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 25px 0;
            background-color: #fff;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .metric-table th {{
            background-color: #4472C4;
            color: white;
            padding: 12px;
            text-align: center;
            font-weight: 600;
        }}
        .metric-table td {{
            border: 1px solid #ddd;
            padding: 10px;
            text-align: center;
        }}
        .metric-table tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        .metric-table tr:hover {{
            background-color: #e8f4f8;
        }}
        .chart-container {{
            text-align: center;
            margin: 30px 0;
        }}
        .chart-container img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .ai-analysis {{
            background-color: #f8f9fa;
            padding: 25px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 4px solid #4472C4;
            line-height: 1.8;
        }}
        .recommendations {{
            border-left-color: #ff7f0e;
            background-color: #fff9e6;
        }}
        .section {{
            margin-bottom: 50px;
        }}
        .overview {{
            font-size: 1.1em;
            color: #555;
            text-align: justify;
            margin: 30px 0;
        }}
        .chart-description {{
            color: #666;
            font-style: italic;
            margin: 10px 0;
        }}
        @media print {{
            body {{
                background-color: white;
            }}
            .container {{
                box-shadow: none;
                padding: 20px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Urban Green Space Visual Analysis Report</h1>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        {ai_sections.split('<div class="section">')[1] if ai_analysis and 'executive_summary' in ai_analysis else ''}
        
        <div class="section">
            <h2>1. Analysis Overview</h2>
            <p class="overview">
                This report presents a comprehensive spatial visual analysis of <strong>{len(metrics_results)}</strong> images 
                using <strong>{len(selected_metrics)}</strong> spatial metrics. The analysis employs advanced computer vision 
                techniques including semantic segmentation and depth estimation to evaluate spatial characteristics 
                and visual quality of urban green spaces.
            </p>
        </div>
        
        <div class="section">
            <h2>2. Metric Statistics</h2>
            <table class="metric-table">
                <thead>
                    <tr>
                        <th>Metric Name</th>
                        <th>Mean</th>
                        <th>Std Dev</th>
                        <th>Min</th>
                        <th>Max</th>
                        <th>CV</th>
                    </tr>
                </thead>
                <tbody>
"""
            
            # Add metric data
            for metric in selected_metrics:
                metric_name = metric['metric name']
                if metric_name in metrics_results.columns:
                    values = metrics_results[metric_name].dropna()
                    if len(values) > 0:
                        mean_val = values.mean()
                        std_val = values.std()
                        cv = std_val / mean_val if mean_val != 0 else 0
                        html_content += f"""
                <tr>
                    <td>{metric_name}</td>
                    <td>{mean_val:.3f}</td>
                    <td>{std_val:.3f}</td>
                    <td>{values.min():.3f}</td>
                    <td>{values.max():.3f}</td>
                    <td>{cv:.3f}</td>
                </tr>
"""
            
            html_content += """
                </tbody>
            </table>
        </div>
"""
            
            # Add metric analysis section if available
            if ai_analysis and 'metric_analysis' in ai_analysis:
                html_content += ai_sections.split('<div class="section">')[2] if '<div class="section">' in ai_sections else ''
            
            # Add visualizations
            html_content += """
        <div class="section">
            <h2>3. Data Visualization</h2>
"""
            
            chart_descriptions = {
                'distribution': "Distribution of metric values across all analyzed images",
                'correlation': "Correlation analysis between different spatial metrics",
                'radar': "Normalized metric values showing relative performance",
                'trend': "Trend analysis showing metric variations across images"
            }
            
            for chart_name, chart_path in charts.items():
                if os.path.exists(chart_path):
                    chart_filename = os.path.basename(chart_path)
                    html_content += f"""
            <div class="chart-container">
                <p class="chart-description">{chart_descriptions.get(chart_name, '')}</p>
                <img src="{chart_filename}" alt="{chart_name}">
            </div>
"""
            
            html_content += """
        </div>
"""
            
            # Add recommendations if available
            if ai_analysis and 'recommendations' in ai_analysis:
                for section in ai_sections.split('<div class="section">'):
                    if 'Design Recommendations' in section:
                        html_content += '<div class="section">' + section
                        break
            
            # Add comparative analysis if available
            if ai_analysis and 'comparative_analysis' in ai_analysis:
                for section in ai_sections.split('<div class="section">'):
                    if 'Comparative Analysis' in section:
                        html_content += '<div class="section">' + section
                        break
            
            html_content += """
        <div class="section">
            <h2>Methodology</h2>
            <div class="overview">
                <p>This analysis employs state-of-the-art computer vision techniques:</p>
                <ul>
                    <li><strong>Semantic Segmentation:</strong> Deep learning models identify and classify different elements in the urban landscape</li>
                    <li><strong>Depth Estimation:</strong> Monocular depth estimation provides spatial depth information</li>
                    <li><strong>Spatial Metrics:</strong> Various quantitative metrics evaluate spatial characteristics including openness, complexity, and visual quality</li>
                    <li><strong>Statistical Analysis:</strong> Comprehensive statistical analysis identifies patterns and correlations</li>
                </ul>
                <p>All metrics are calculated based on established urban design and landscape architecture principles.</p>
            </div>
        </div>
    </div>
</body>
</html>
"""
            
            # Save HTML file
            html_path = os.path.join(report_dir, 'analysis_report.html')
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return html_path
            
        except Exception as e:
            logger.error(f"Failed to generate HTML report: {e}")
            return ""
    
    def _save_raw_data(self, report_dir: str, metrics_results: pd.DataFrame,
                      selected_metrics: List[Dict]):
        """Save raw data for future reference"""
        try:
            # Save metric results
            metrics_results.to_csv(os.path.join(report_dir, 'metrics_results.csv'), index=False)
            metrics_results.to_excel(os.path.join(report_dir, 'metrics_results.xlsx'), index=False)
            
            # Save selected metrics information
            with open(os.path.join(report_dir, 'selected_metrics.json'), 'w', encoding='utf-8') as f:
                json.dump(selected_metrics, f, ensure_ascii=False, indent=2)
            
            # Create a README file
            readme_content = f"""
# Analysis Report Data Files

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Files Included:

1. **metrics_results.csv** - Raw metric calculation results in CSV format
2. **metrics_results.xlsx** - Raw metric calculation results in Excel format
3. **selected_metrics.json** - Detailed information about the metrics used
4. **analysis_report.pdf** - Complete analysis report in PDF format
5. **analysis_report.html** - Complete analysis report in HTML format
6. **[chart_files].png** - Various visualization charts

## Metrics Analyzed:

Total images: {len(metrics_results)}
Total metrics: {len(selected_metrics)}

### Metric List:
{chr(10).join(['- ' + m['metric name'] for m in selected_metrics])}

## Usage:

- The CSV/Excel files can be imported into statistical software for further analysis
- The JSON file contains metadata about each metric including interpretation guidelines
- Charts can be used in presentations or publications
"""
            
            with open(os.path.join(report_dir, 'README.md'), 'w', encoding='utf-8') as f:
                f.write(readme_content)
            
        except Exception as e:
            logger.error(f"Failed to save raw data: {e}")
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temp directory: {self.temp_dir}")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
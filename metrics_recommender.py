"""
指标推荐API服务
基于用户输入推荐合适的空间视觉分析指标
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from openai import OpenAI
from pathlib import Path

import pandas as pd
import logging
import os
import json
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Metrics Recommendation API",
    description="API for recommending spatial visual characteristics metrics based on user needs",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class SimpleMetricsRequest(BaseModel):
    user_input: str = Field(..., description="User input text describing their needs")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API Key")
    temperature: Optional[float] = Field(default=0.3, description="Temperature for AI response (0.0-1.0)")
    max_tokens: Optional[int] = Field(default=1000, description="Maximum tokens for response")

class MetricsResponse(BaseModel):
    recommendation: str
    usage: Optional[Dict] = None
    error: Optional[str] = None

# Global variables
metrics_data = None
metrics_df = None

def load_metrics_data():
    """Load metrics data from Excel file"""
    global metrics_data, metrics_df
    try:
        # 尝试多个可能的路径
        possible_paths = [
            Path("./data/library_metrics.xlsx"),
            Path("./library_metrics.xlsx"),
            Path("../data/library_metrics.xlsx"),
            Path(os.path.join(os.path.dirname(__file__), "data", "library_metrics.xlsx")),
            Path(os.path.join(os.path.dirname(__file__), "library_metrics.xlsx"))
        ]
        
        metric_path = None
        for path in possible_paths:
            logger.info(f"Checking path: {path.resolve()}")
            if path.exists():
                metric_path = path
                break
        
        if metric_path is None:
            # 如果找不到Excel文件，尝试JSON文件
            json_paths = [
                Path("./data/library_metrics.json"),
                Path("./library_metrics.json"),
                Path(os.path.join(os.path.dirname(__file__), "library_metrics.json"))
            ]
            
            for path in json_paths:
                if path.exists():
                    logger.info(f"Found JSON file at: {path}")
                    with open(path, 'r', encoding='utf-8') as f:
                        metrics_list = json.load(f)
                    metrics_df = pd.DataFrame(metrics_list)
                    # 保存为Excel
                    excel_path = path.parent / "library_metrics.xlsx"
                    metrics_df.to_excel(excel_path, index=False)
                    logger.info(f"Created Excel file from JSON: {excel_path}")
                    metric_path = excel_path
                    break
        
        if metric_path is None:
            logger.error("No metrics file found, creating default metrics")
            # 创建默认指标
            default_metrics = create_default_metrics()
            metrics_df = pd.DataFrame(default_metrics)
            # 保存到data目录
            os.makedirs("data", exist_ok=True)
            metrics_df.to_excel("data/library_metrics.xlsx", index=False)
            metric_path = Path("data/library_metrics.xlsx")
        
        # 读取指标数据
        metrics_df = pd.read_excel(metric_path)
        metrics_data = metrics_df.to_dict(orient='records')
        metrics_data = json.dumps(metrics_data, ensure_ascii=False, indent=2)
        logger.info(f"Successfully loaded {len(metrics_df)} metrics from {metric_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error loading metrics data: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def create_default_metrics():
    """创建默认指标数据"""
    return [
        {
            "metric name": "Shape Edge Regularity Index (S_ERI)",
            "Primary Category": "Composition/Configuration",
            "Secondary Attribute": "Shape",
            "Classification Rationale": "Evaluates the regularity of the foreground element's boundary",
            "Standard Range": "[π/2, +∞)",
            "Unit": "Dimensionless",
            "Parameter Definition": "Based on isoperimetric inequality, ratio of perimeter to area",
            "Data Input": "Segmentation Image",
            "Calculation Method": "S_ERI = 0.25 * (P / A)",
            "Professional Interpretation": "High regularity aids recognition and sense of order; too low causes monotony"
        },
        {
            "metric name": "Shape Edge Contrast Index (S_ECI)",
            "Primary Category": "Composition/Configuration", 
            "Secondary Attribute": "Shape",
            "Classification Rationale": "Reflects the difference between foreground boundary and surrounding",
            "Standard Range": "[0,1]",
            "Unit": "Dimensionless",
            "Parameter Definition": "Measures label consistency of pixels in foreground contour neighborhood",
            "Data Input": "Segmentation Image",
            "Calculation Method": "S_ECI = Py / (Py + Pb)",
            "Professional Interpretation": "High contrast enhances spatial independence, low contrast increases continuity"
        },
        {
            "metric name": "Size View Field Ratio (S_VFR)",
            "Primary Category": "Composition/Configuration",
            "Secondary Attribute": "Size",
            "Classification Rationale": "Ratio of foreground area in the field of view",
            "Standard Range": "[0,1]",
            "Unit": "Dimensionless",
            "Parameter Definition": "Ratio of foreground pixels to total field of view pixels",
            "Data Input": "Segmentation Image",
            "Calculation Method": "S_VFR = Ai/Aj",
            "Professional Interpretation": "High foreground ratio enhances enclosure; low ratio extends the view"
        },
        {
            "metric name": "Shape Patch Similarity Index (S_PSI)",
            "Primary Category": "Composition/Configuration",
            "Secondary Attribute": "Shape",
            "Classification Rationale": "Measures distribution uniformity of openings in foreground",
            "Standard Range": "[0,+∞)",
            "Unit": "Dimensionless",
            "Parameter Definition": "Shape variability of all foreground holes' contours",
            "Data Input": "Segmentation Image",
            "Calculation Method": "S_PSI = CV(0.25Pi/Ai)",
            "Professional Interpretation": "High diversity stimulates exploration, low diversity promotes tranquility"
        },
        {
            "metric name": "Position Location Value (P_LV)",
            "Primary Category": "Position",
            "Secondary Attribute": "Location",
            "Classification Rationale": "Starting depth value of foreground element in visual field",
            "Standard Range": "[0,1]",
            "Unit": "Dimensionless",
            "Parameter Definition": "Normalized coordinate of foreground starting point along depth axis",
            "Data Input": "Segmentation Image/Depth Map",
            "Calculation Method": "P_LV = (start position) / (total depth)",
            "Professional Interpretation": "Positioning affects spatial sequence and experience of depth"
        }
    ]

def create_openai_client(api_key: str):
    """Create OpenAI client with provided API key"""
    try:
        return OpenAI(api_key=api_key)
    except Exception as e:
        logger.error(f"Error creating OpenAI client: {str(e)}")
        return None

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    logger.info("Starting up Metrics Recommendation API...")
    
    # Load metrics data
    if not load_metrics_data():
        logger.warning("Failed to load metrics data - API will have limited functionality")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Metrics Recommendation API",
        "version": "1.0.0",
        "status": "running",
        "metrics_loaded": metrics_df is not None and len(metrics_df) > 0
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "metrics_count": len(metrics_df) if metrics_df is not None else 0
    }

@app.post("/recommend/simple", response_model=MetricsResponse)
async def recommend_metrics_simple(request: SimpleMetricsRequest):
    """
    Recommend metrics based on user input describing their spatial visual characteristics needs
    """
    if metrics_data is None:
        # 尝试重新加载
        if not load_metrics_data():
            raise HTTPException(status_code=503, detail="Metrics data not available")

    # 获取API密钥
    api_key = os.getenv("OPENAI_API_KEY") or request.openai_api_key
    
    if not api_key:
        # 如果没有OpenAI API密钥，使用基于关键词的简单推荐
        logger.warning("No OpenAI API key provided, using keyword-based recommendation")
        return keyword_based_recommendation(request.user_input)
    
    openai_client = create_openai_client(api_key)
    if openai_client is None:
        raise HTTPException(status_code=400, detail="Invalid OpenAI API key or client creation failed")

    try:
        # Create system prompt
        system_prompt = f"""You are a helpful assistant to help the user decide which metrics to use based on their needs. The available metrics and their descriptions are provided below:

        {metrics_data}

        Please output a python list with the following structure:
        [
            {{
                'name': 'metric name',
                'reason': 'why this metric was chosen'
            }},
            ...
        ]

        – Only output the list with the above format. 
        - Do not include any additional text or formatting.
        - Do not wrap the output in Markdown code fences (```)
        - You could choose multiple metrics if they are relevant to the user's needs.
        - Choose metrics that are most relevant to the user's description.
        - Consider the spatial analysis context when making recommendations."""

        response = openai_client.chat.completions.create(
            model="gpt-4" if "gpt-4" in api_key else "gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"User's needs: {request.user_input}"}
            ],
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        
        return MetricsResponse(
            recommendation=response.choices[0].message.content,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        )
        
    except Exception as e:
        logger.error(f"Error in recommend_metrics_simple: {str(e)}")
        logger.error(traceback.format_exc())
        
        # 失败时返回基于关键词的推荐
        return keyword_based_recommendation(request.user_input)

def keyword_based_recommendation(user_input: str) -> MetricsResponse:
    """基于关键词的简单推荐（备用方案）"""
    try:
        if metrics_df is None:
            return MetricsResponse(
                recommendation="[]",
                error="Metrics data not loaded"
            )
        
        # 关键词映射
        keyword_mapping = {
            "shape": ["S_ERI", "S_ECI", "S_PSI"],
            "边界": ["S_ERI", "S_ECI"],
            "规则": ["S_ERI"],
            "对比": ["S_ECI"],
            "size": ["S_VFR", "S_ADI"],
            "大小": ["S_VFR", "S_ADI"],
            "面积": ["S_VFR"],
            "position": ["P_LV", "P_LN", "P_ECI"],
            "位置": ["P_LV", "P_LN"],
            "深度": ["P_LV"],
            "texture": ["T_ERI", "T_ESI", "T_ISI"],
            "纹理": ["T_ERI", "T_ESI"],
            "open": ["openness"],
            "开放": ["openness"],
            "封闭": ["enclosure"],
            "layer": ["P_LV", "fmb"],
            "层次": ["P_LV", "fmb"],
            "前景": ["foreground", "S_VFR"],
            "中景": ["middleground"],
            "背景": ["background"]
        }
        
        recommendations = []
        user_input_lower = user_input.lower()
        
        # 基于关键词匹配
        for keyword, metric_keywords in keyword_mapping.items():
            if keyword in user_input_lower:
                for metric_keyword in metric_keywords:
                    # 在指标库中查找匹配的指标
                    for _, metric in metrics_df.iterrows():
                        metric_name = metric.get('metric name', '')
                        if metric_keyword.upper() in metric_name.upper():
                            if not any(r['name'] == metric_name for r in recommendations):
                                recommendations.append({
                                    'name': metric_name,
                                    'reason': f'与"{keyword}"相关的指标'
                                })
        
        # 如果没有找到匹配，返回最常用的指标
        if not recommendations:
            common_metrics = ['Shape Edge Regularity Index (S_ERI)', 
                            'Size View Field Ratio (S_VFR)',
                            'Position Location Value (P_LV)']
            for metric_name in common_metrics:
                if metric_name in metrics_df['metric name'].values:
                    recommendations.append({
                        'name': metric_name,
                        'reason': '常用的基础分析指标'
                    })
        
        return MetricsResponse(
            recommendation=json.dumps(recommendations, ensure_ascii=False)
        )
        
    except Exception as e:
        logger.error(f"Error in keyword_based_recommendation: {str(e)}")
        return MetricsResponse(
            recommendation="[]",
            error=str(e)
        )

@app.get("/metrics")
async def get_all_metrics():
    """获取所有可用的指标"""
    if metrics_df is None:
        raise HTTPException(status_code=503, detail="Metrics data not available")
    
    return {
        "metrics": metrics_df.to_dict(orient='records'),
        "count": len(metrics_df)
    }

@app.get("/metrics/{metric_name}")
async def get_metric_detail(metric_name: str):
    """获取特定指标的详细信息"""
    if metrics_df is None:
        raise HTTPException(status_code=503, detail="Metrics data not available")
    
    metric = metrics_df[metrics_df['metric name'] == metric_name]
    if metric.empty:
        raise HTTPException(status_code=404, detail=f"Metric '{metric_name}' not found")
    
    return metric.iloc[0].to_dict()

if __name__ == "__main__":
    import uvicorn
    
    # 尝试从环境变量获取端口
    port = int(os.getenv("METRICS_API_PORT", "8001"))
    
    logger.info(f"Starting Metrics Recommendation API on port {port}")
    logger.info("API documentation available at http://localhost:{port}/docs")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info"
    )
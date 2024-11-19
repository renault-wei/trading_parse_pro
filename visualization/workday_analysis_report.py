import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
from datetime import datetime
from visualization.chart_generator import generate_price_comparison, generate_period_heatmap, generate_statistical_tables, generate_prediction_analysis

class WorkdayAnalysisReport:
    """工作日分析报告生成器"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.output_dir = Path(config.get('OUTPUT', 'charts_dir'))
        
    def generate_report(self, data: pd.DataFrame, output_name: str = 'workday_analysis'):
        """生成完整的分析报告"""
        try:
            # 创建HTML模板
            html_content = self._create_html_template()
            
            # 生成各个分析图表
            figures = {
                'price_comparison': generate_price_comparison(data),
                'period_heatmap': generate_period_heatmap(data),
                'statistical_tables': generate_statistical_tables(data),
                'prediction_analysis': generate_prediction_analysis(prediction_data)
            }
            
            # 将图表插入HTML
            html_content = self._insert_figures(html_content, figures)
            
            # 保存报告
            output_path = self.output_dir / f'{output_name}.html'
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            self.logger.info(f"工作日分析报告已生成: {output_path}")
            
        except Exception as e:
            self.logger.error(f"生成工作日分析报告时出错: {str(e)}")
            raise
            
    def _create_html_template(self) -> str:
        """创建HTML模板"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>工作日vs非工作日分析报告</title>
            <meta charset="utf-8">
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }
                .container {
                    max-width: 1800px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }
                .section {
                    margin-bottom: 30px;
                }
                .chart-container {
                    margin-bottom: 20px;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }
                th, td {
                    padding: 8px;
                    border: 1px solid #ddd;
                    text-align: center;
                }
                th {
                    background-color: #f8f9fa;
                }
                .header {
                    text-align: center;
                    margin-bottom: 30px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>工作日vs非工作日分析报告</h1>
                    <p>生成时间: {timestamp}</p>
                </div>
                
                <div class="section">
                    <h2>1. 价格走势分析</h2>
                    <div id="price_comparison"></div>
                </div>
                
                <div class="section">
                    <h2>2. 时段特征分析</h2>
                    <div id="period_heatmap"></div>
                </div>
                
                <div class="section">
                    <h2>3. 统计指标分析</h2>
                    <div id="statistical_tables"></div>
                </div>
            </div>
        </body>
        </html>
        """.format(timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
    def _insert_figures(self, html_content: str, figures: dict) -> str:
        """将图表插入HTML"""
        for key, figure in figures.items():
            html_content = html_content.replace(f'<div id="{key}"></div>', figure.to_html(full_html=False, include_plotlyjs='cdn'))
        return html_content
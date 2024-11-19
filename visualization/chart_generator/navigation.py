from pathlib import Path
from datetime import datetime
from .base import BaseChartGenerator

class NavigationGenerator(BaseChartGenerator):
    """导航页面生成器"""
    
    def generate_navigation_page(self, output_dirs: dict):
        """生成导航页面"""
        try:
            # 收集所有生成的图表文件
            chart_files = self._collect_chart_files(output_dirs)
          #  self.logger.info(f"查看chart_files: {chart_files}")

            
            # 生成HTML内容
            html_content = self._create_navigation_html(chart_files)
            
            # 保存导航页面
            nav_path = Path('trading_system/output/index.html')
            nav_path.parent.mkdir(parents=True, exist_ok=True)
            nav_path.write_text(html_content, encoding='utf-8')
            
            self.logger.info(f"导航页面已生成: {nav_path}")
            return str(nav_path)
            
        except Exception as e:
            self.logger.error(f"生成导航页面时出错: {str(e)}")
            return None
            
    def _collect_chart_files(self, output_dirs: dict) -> dict:
        """收集所有生成的图表文件"""
        chart_files = {}
        base_dir = Path('trading_system/output')
        
        for category, dir_path in output_dirs.items():
            path = Path(dir_path)
            if path.exists():
                chart_files[category] = [
                    file.relative_to(base_dir) 
                    for file in path.glob('*.html')
                ]
                
        return chart_files
        
    def _create_navigation_html(self, chart_files: dict) -> str:
        """创建导航页面HTML内容"""
        # 定义模块信息
        modules = {
            'price_analysis': {
                'title': '价格分析',
                'description': '价格趋势、波动性和分布特征分析',
                'icon': '💰',
                'charts': [
                    'price_trend.html',
                    'price_distribution.html',
                    'price_volatility.html'
                ]
            },
            'time_patterns': {
                'title': '时间模式',
                'description': '日内、周内、月度和季节性价格模式',
                'icon': '⏰',
                'charts': [
                    'daily_pattern.html',
                    'weekly_pattern.html',
                    'seasonal_pattern.html'
                ]
            },
            'features': {
                'title': '特征分析',
                'description': '价格影响因素特征分析',
                'icon': '📊',
                'charts': [
                    'feature_importance.html',
                    'feature_correlation.html',
                    'feature_distribution.html'
                ]
            },
            'regression': {
                'title': '回归分析',
                'description': '价格影响因素的回归分析结果',
                'icon': '📈',
                'charts': [
                    'price_factors_regression.html',
                    'regression_diagnostics.html',
                    'factor_importance.html'
                ]
            },
            'supply_demand': {
                'title': '供需分析',
                'description': '供需关系对价格的影响分析',
                'icon': '⚖️',
                'charts': [
                    'supply_demand_balance.html',
                    'supply_demand_impact.html',
                    'price_elasticity.html',
                    'supply_demand_dashboard.html'
                ]
            },
            'workday_analysis': {
                'title': '工作日分析',
                'description': '工作日与工作日价格特征对比',
                'icon': '📅',
                'charts': [
                    'workday_comparison.html',
                    'holiday_effect.html',
                    'workday_pattern.html'
                ]
            },
            'peak_valley': {
                'title': '峰谷分析',
                'description': '电价峰谷特征和转换规律分析',
                'icon': '🌊',
                'charts': [
                    'peak_valley_pattern.html',
                    'price_transition.html',
                    'peak_valley_distribution.html'
                ]
            },
            'predictions': {
                'title': '预测分析',
                'description': '价格预测模型和结果分析',
                'icon': '🔮',
                'charts': [
                    'prediction_analysis.html',
                    'prediction_notice.html'
                ]
            }
        }
        
        # 生成HTML内容
        html_content = self._get_html_template()
        
        # 添加模块卡片
        module_cards = []
        for category, info in modules.items():
            if category in chart_files and chart_files[category]:
          #      self.logger.info(f"查看chart_files[category]: {chart_files[category]}")
                chart_links = '\n'.join([
                    f'<li><a href="./{file}" class="chart-link" target="_blank">{file.stem}</a></li>'
                    for file in chart_files[category]
                ])
                
                module_cards.append(f"""
                    <div class="module-card">
                        <div class="module-icon">{info['icon']}</div>
                        <div class="module-title">{info['title']}</div>
                        <div class="module-description">{info['description']}</div>
                        <ul class="chart-list">
                            {chart_links}
                        </ul>
                    </div>
                """)
                
        # 打印调试信息
       # print("生成的模块卡片:", module_cards)
        
        # 组装最终的HTML内容
        html_content = html_content.format(
            module_cards='\n'.join(module_cards),
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
        
        return html_content
        
    def _get_html_template(self) -> str:
        """获取HTML模板"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>电力价格分析导航</title>
            <meta charset="utf-8">
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 30px;
                    padding: 20px;
                    background-color: #fff;
                    border-radius: 10px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .module-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                .module-card {{
                    background-color: #fff;
                    border-radius: 10px;
                    padding: 20px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    transition: transform 0.3s ease;
                }}
                .module-card:hover {{
                    transform: translateY(-5px);
                }}
                .module-icon {{
                    font-size: 2em;
                    margin-bottom: 10px;
                }}
                .module-title {{
                    font-size: 1.2em;
                    font-weight: bold;
                    margin-bottom: 10px;
                    color: #333;
                }}
                .module-description {{
                    color: #666;
                    margin-bottom: 15px;
                }}
                .chart-list {{
                    list-style: none;
                    padding: 0;
                }}
                .chart-link {{
                    display: block;
                    padding: 8px 0;
                    color: #007bff;
                    text-decoration: none;
                }}
                .chart-link:hover {{
                    color: #0056b3;
                }}
                .timestamp {{
                    text-align: center;
                    color: #666;
                    margin-top: 20px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>电力价格分析导航</h1>
                    <p>选择以下模块查看详细分析结果</p>
                </div>
                <div class="module-grid">
                    {module_cards}
                </div>
                <div class="timestamp">
                    生成时间: {timestamp}
                </div>
            </div>
        </body>
        </html>
        """ 
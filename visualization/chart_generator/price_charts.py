import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from pathlib import Path
from .base import BaseChartGenerator

class PriceChartGenerator(BaseChartGenerator):
    """价格相关图表生成器"""
    
    def generate_price_trend_analysis(self, data: pd.DataFrame):
        """生成价格趋势分析图表"""
        try:
            # 创建子图
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=[
                    '日内价格趋势叠加',
                    '月度小时均价趋势'
                ],
                vertical_spacing=0.15
            )
            
            # 1. 每日价格趋势叠加
            daily_groups = data.groupby(data.index.date)
            
            # 为每一天画一条线
            for date, group in daily_groups:
                group = group.sort_index()
                fig.add_trace(
                    go.Scatter(
                        x=group['hour'],
                        y=group['price'],
                        name=str(date),
                        line=dict(width=1),
                        opacity=0.3,
                        showlegend=False
                    ),
                    row=1, col=1
                )
            
            # 添加平均趋势线
            daily_avg = data.groupby('hour')['price'].mean()
            fig.add_trace(
                go.Scatter(
                    x=daily_avg.index,
                    y=daily_avg.values,
                    name='日均价格',
                    line=dict(color='red', width=3),
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # 2. 月度小时均价趋势
            monthly_hourly_avg = data.groupby([data.index.month, 'hour'])['price'].mean().unstack()
            
            month_names = ['一月', '二月', '三月', '四月', '五月', '六月', 
                          '七月', '八月', '九月', '十月', '十一月', '十二月']
            
            for month in monthly_hourly_avg.index:
                month_data = monthly_hourly_avg.loc[month]
                fig.add_trace(
                    go.Scatter(
                        x=month_data.index,
                        y=month_data.values,
                        name=month_names[month-1],
                        line=dict(width=2)
                    ),
                    row=2, col=1
                )
            
            # 更新布局
            fig.update_layout(
                title='价格趋势分析',
                template=self.theme,
                height=1000,
                width=1200,
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.05
                )
            )
            
            # 更新x轴设置
            for row in [1, 2]:
                fig.update_xaxes(
                    title_text='小时',
                    range=[0, 24],
                    tickmode='linear',
                    tick0=0,
                    dtick=1,
                    constrain='domain',
                    fixedrange=True,
                    row=row, col=1
                )
            
            # 更新y轴标签
            fig.update_yaxes(title_text='价格', row=1, col=1)
            fig.update_yaxes(title_text='价格', row=2, col=1)
            
            # 保存图表
            output_path = Path(self.output_dirs['price_analysis']) / 'price_trend_analysis.html'
            fig.write_html(
                str(output_path),
                config={
                    'scrollZoom': True,
                    'displayModeBar': True,
                    'modeBarButtonsToRemove': ['hoverClosestCartesian', 'hoverCompareCartesian']
                }
            )

            # 生成分布表
            self.generate_price_distribution(data)
            
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"生成价格趋势分析图表时出错: {str(e)}")
            return None
            
    def generate_price_distribution(self, data: pd.DataFrame):
        """生成价格分布分析图表"""
        try:
            fig = go.Figure()
            
            # 添加价格分布直方图
            fig.add_trace(go.Histogram(
                x=data['price'],
                nbinsx=50,
                name='价格分布'
            ))
            
            # 添加核密度估计
            from scipy import stats
            kde_x = np.linspace(data['price'].min(), data['price'].max(), 100)
            kde = stats.gaussian_kde(data['price'].dropna())
            kde_y = kde(kde_x)
            
            fig.add_trace(go.Scatter(
                x=kde_x,
                y=kde_y * len(data['price']) * (data['price'].max() - data['price'].min()) / 50,
                name='密度估计',
                line=dict(color='red')
            ))
            
            # 更新布局
            fig.update_layout(
                title='价格分布分析',
                xaxis_title='价格',
                yaxis_title='频数',
                template=self.theme,
                showlegend=True
            )
            
            # 保存图表
            output_path = Path(self.output_dirs['price_analysis']) / 'price_distribution.html'
            fig.write_html(str(output_path))
            
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"生成价格分布分析图表时出错: {str(e)}")
            return None 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from utils.logger import Logger

class SupplyDemandCharts:
    """供需分析图表生成器"""
    
    def __init__(self):
        self.logger = Logger().get_logger()
        
    def create_supply_demand_dashboard(self, data: pd.DataFrame, impact_analysis: dict):
        """创建供需分析仪表板"""
        try:
            # 创建子图
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    '供需压力时序图',
                    '供需比率分布',
                    '价格与供需压力相关性',
                    '供需影响分析'
                ),
                vertical_spacing=0.15,
                horizontal_spacing=0.1
            )
            
            # 1. 供需压力时序图
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['supply_pressure_ma'],
                    name='供给压力',
                    line=dict(color='red')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['demand_pressure_ma'],
                    name='需求压力',
                    line=dict(color='green')
                ),
                row=1, col=1
            )
            
            # 2. 供需比率分布
            fig.add_trace(
                go.Histogram(
                    x=data['supply_demand_ratio'].clip(-5, 5),  # 限制范围以便更好地显示
                    name='供需比率分布',
                    nbinsx=50
                ),
                row=1, col=2
            )
            
            # 3. 价格与供需压力的散点图
            fig.add_trace(
                go.Scatter(
                    x=data['supply_pressure_ma'],
                    y=data['price'],
                    mode='markers',
                    name='供给压力vs价格',
                    marker=dict(
                        color='red',
                        size=4,
                        opacity=0.5
                    )
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data['demand_pressure_ma'],
                    y=data['price'],
                    mode='markers',
                    name='需求压力vs价格',
                    marker=dict(
                        color='green',
                        size=4,
                        opacity=0.5
                    )
                ),
                row=2, col=1
            )
            
            # 4. 供需影响分析
            correlations = impact_analysis['correlation']
            pred_corr = impact_analysis['predictive_correlation']
            
            fig.add_trace(
                go.Bar(
                    x=['供给相关性', '需求相关性', '供给预测性', '需求预测性'],
                    y=[
                        correlations.get('supply_pressure_ma', 0),
                        correlations.get('demand_pressure_ma', 0),
                        pred_corr['supply'],
                        pred_corr['demand']
                    ],
                    name='相关性分析'
                ),
                row=2, col=2
            )
            
            # 更新布局
            fig.update_layout(
                height=1000,
                width=1400,
                title_text='供需分析仪表板',
                showlegend=True,
                template='plotly_white'
            )
            
            # 更新轴标签
            fig.update_xaxes(title_text='时间', row=1, col=1)
            fig.update_xaxes(title_text='供需比率', row=1, col=2)
            fig.update_xaxes(title_text='压力值', row=2, col=1)
            
            fig.update_yaxes(title_text='压力值', row=1, col=1)
            fig.update_yaxes(title_text='频数', row=1, col=2)
            fig.update_yaxes(title_text='价格', row=2, col=1)
            fig.update_yaxes(title_text='相关系数', row=2, col=2)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"创建供需分析图表时发生错误: {str(e)}")
            return None
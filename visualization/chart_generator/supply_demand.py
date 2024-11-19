import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from utils.logger import Logger
import numpy as np

class SupplyDemandChartGenerator:
    """供需分析图表生成器"""
    
    def __init__(self, theme: str = 'plotly_white'):
        self.logger = Logger().get_logger()
        self.theme = theme
        
    def generate_supply_demand_analysis(self, data: pd.DataFrame) -> None:
        """生成供需分析图表"""
        try:
            # 1. 生成供需平衡图表
            balance_fig = self.generate_supply_demand_balance(data)
            balance_fig.write_html('trading_system/output/supply_demand/supply_demand_balance.html')
            
            # 2. 生成供需影响分析图表
            impact_fig = self.generate_supply_demand_impact(data)
            impact_fig.write_html('trading_system/output/supply_demand/supply_demand_impact.html')
            
            # 3. 生成价格弹性分析图表
            elasticity_fig = self.generate_price_elasticity_plot(data)
            elasticity_fig.write_html('trading_system/output/supply_demand/price_elasticity.html')
            
            # 4. 生成供需分析仪表板
            impact_analysis = {
                'correlation': {
                    'supply_pressure_ma': data['supply_pressure_ma'].corr(data['price']),
                    'demand_pressure_ma': data['demand_pressure_ma'].corr(data['price'])
                },
                'predictive_correlation': {
                    'supply': data['supply_pressure_ma'].shift(1).corr(data['price']),
                    'demand': data['demand_pressure_ma'].shift(1).corr(data['price'])
                }
            }
            dashboard_fig = self.generate_supply_demand_dashboard(data, impact_analysis)
            dashboard_fig.write_html('trading_system/output/supply_demand/supply_demand_dashboard.html')
            
        except Exception as e:
            self.logger.error(f"生成供需分析图表时出错: {str(e)}")
            raise
            
    def generate_supply_demand_balance(self, data: pd.DataFrame) -> go.Figure:
        """生成供需平衡图表"""
        fig = go.Figure()
        
        # 添加供给压力和需求压力的时序图
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['supply_pressure_ma'],
            name='供给压力',
            line=dict(color='red')
        ))
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['demand_pressure_ma'],
            name='需求压力',
            line=dict(color='green')
        ))
        
        fig.update_layout(
            title='供需平衡分析',
            xaxis_title='时间',
            yaxis_title='压力值',
            template=self.theme
        )
        
        return fig
        
    def generate_supply_demand_impact(self, data: pd.DataFrame) -> go.Figure:
        """生成供需影响分析图表"""
        fig = go.Figure()
        
        # 添加供需压力与价格的散点图
        fig.add_trace(go.Scatter(
            x=data['supply_pressure_ma'],
            y=data['price'],
            mode='markers',
            name='供给压力vs价格',
            marker=dict(color='red', size=4, opacity=0.5)
        ))
        
        fig.add_trace(go.Scatter(
            x=data['demand_pressure_ma'],
            y=data['price'],
            mode='markers',
            name='需求压力vs价格',
            marker=dict(color='green', size=4, opacity=0.5)
        ))
        
        fig.update_layout(
            title='供需影响分析',
            xaxis_title='压力值',
            yaxis_title='价格',
            template=self.theme
        )
        
        return fig
        
    def generate_price_elasticity_plot(self, data: pd.DataFrame) -> go.Figure:
        """生成价格弹性分析图表"""
        fig = go.Figure()
        
        # 计算价格变化和供给、需求变化
        price_changes = data['price'].pct_change()
        supply_changes = data['supply_pressure_ma'].pct_change()
        demand_changes = data['demand_pressure_ma'].pct_change()
        
        # 计算弹性
        rolling_supply_elasticity = price_changes.rolling(window=24).corr(supply_changes)
        rolling_demand_elasticity = price_changes.rolling(window=24).corr(demand_changes)
        
        # 添加供给弹性和需求弹性到图表
        fig.add_trace(go.Scatter(
            x=data.index,
            y=rolling_supply_elasticity,
            name='供给弹性',
            line=dict(color='red')
        ))
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=rolling_demand_elasticity,
            name='需求弹性',
            line=dict(color='green')
        ))
        
        fig.update_layout(
            title='价格弹性分析',
            xaxis_title='时间',
            yaxis_title='弹性系数',
            template=self.theme
        )
        
        return fig
        
    def generate_supply_demand_dashboard(self, data: pd.DataFrame, impact_analysis: dict) -> go.Figure:
        """生成供需分析仪表板"""
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
            # 计算供需比率
            supply_demand_ratio = data['supply_pressure_ma'] / data['demand_pressure_ma']
            data['supply_demand_ratio'] = supply_demand_ratio.replace([np.inf, -np.inf], np.nan)
            
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
                template=self.theme
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
            self.logger.error(f"创建供需分析仪表板时出错: {str(e)}")
            raise  # 抛出异常而不是返回 None

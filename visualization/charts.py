import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List

class ChartGenerator:
    """图表生成器"""
    
    def __init__(self, theme: str = 'plotly'):
        self.theme = theme
        
    def plot_price_chart(self, data, title: str = "价格走势"):
        """绘制价格走势图"""
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close']
        ))
        fig.update_layout(title=title)
        return fig
        
    def plot_performance(self, performance_data):
        """绘制回测业绩图表"""
        pass
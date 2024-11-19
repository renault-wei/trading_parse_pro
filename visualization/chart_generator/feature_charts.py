import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
from .base import BaseChartGenerator

class FeatureChartGenerator(BaseChartGenerator):
    """特征分析图表生成器"""
    
    def generate_feature_analysis(self, data: pd.DataFrame):
        """生成特征分析图表"""
        try:
            # 创建子图
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    '价格和移动平均', '价格波动率',
                    '日内模式（正弦）', '日内模式（余弦）',
                    '高峰低谷时段分布', '工作日vs非工作日价格分布'
                )
            )
            
            # 1. 价格和移动平均
            self._add_price_ma_plot(fig, data)
            
            # 2. 价格波动率
            self._add_volatility_plot(fig, data)
            
            # 3. 日内模式（正弦和余弦）
            self._add_intraday_patterns(fig, data)
            
            # 4. 高峰低谷和工作日分布
            self._add_distribution_plots(fig, data)
            
            # 更新布局
            fig.update_layout(
                height=1200,
                width=1600,
                title_text="特征分析仪表板",
                showlegend=True,
                template=self.theme
            )
            
            # 保存主图表
            main_path = Path(self.output_dirs['features']) / 'feature_analysis.html'
            fig.write_html(str(main_path))
            
            # 生成补充图表
            self._generate_supplementary_charts(data)
            
            return str(main_path)
            
        except Exception as e:
            self.logger.error(f"生成特征分析图表时出错: {str(e)}")
            return None
            
    def _add_price_ma_plot(self, fig: go.Figure, data: pd.DataFrame):
        """添加价格和移动平均线图"""
        fig.add_trace(
            go.Scatter(
                x=data.index, 
                y=data['price'], 
                name='价格'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index, 
                y=data['price_ma24'], 
                name='24小时移动平均'
            ),
            row=1, col=1
        )
        
    def _add_volatility_plot(self, fig: go.Figure, data: pd.DataFrame):
        """添加波动率图"""
        fig.add_trace(
            go.Scatter(
                x=data.index, 
                y=data['price_volatility'], 
                name='波动率'
            ),
            row=1, col=2
        )
        
    def _add_intraday_patterns(self, fig: go.Figure, data: pd.DataFrame):
        """添加日内模式图"""
        # 正弦模式
        fig.add_trace(
            go.Scatter(
                x=data['trade_hour'], 
                y=data['hour_sin'],
                mode='markers', 
                name='小时正弦'
            ),
            row=2, col=1
        )
        
        # 余弦模式
        fig.add_trace(
            go.Scatter(
                x=data['trade_hour'], 
                y=data['hour_cos'],
                mode='markers', 
                name='小时余弦'
            ),
            row=2, col=2
        )
        
    def _add_distribution_plots(self, fig: go.Figure, data: pd.DataFrame):
        """添加分布图"""
        # 高峰低谷时段分布
        peak_valley_data = pd.DataFrame({
            '早高峰': data['is_morning_peak'].mean(),
            '晚高峰': data['is_evening_peak'].mean(),
            '低谷': data['is_valley'].mean()
        }, index=[0]).melt()
        
        fig.add_trace(
            go.Bar(
                x=peak_valley_data['variable'],
                y=peak_valley_data['value'],
                name='时段分布'
            ),
            row=3, col=1
        )
        
        # 工作日vs非工作日价格分布
        fig.add_trace(
            go.Box(
                x=data['is_workday'].map({1: '工作日', 0: '非工作日'}),
                y=data['price'],
                name='价格分布'
            ),
            row=3, col=2
        )
        
    def _generate_supplementary_charts(self, data: pd.DataFrame):
        """生成补充图表"""
        try:
            # 1. 每小时价格箱线图
            hour_fig = go.Figure()
            hour_fig.add_trace(
                go.Box(
                    x=data['trade_hour'],
                    y=data['price'],
                    name='每小时价格分布'
                )
            )
            
            hour_fig.update_layout(
                title_text='每小时价格分布',
                xaxis_title='小时',
                yaxis_title='价格',
                template=self.theme
            )
            
            hour_path = Path(self.output_dirs['features']) / 'hourly_price_distribution.html'
            hour_fig.write_html(str(hour_path))
            
            # 2. 特征相关性热力图
            feature_cols = [
                'price', 'price_ma24', 'price_volatility',
                'hour_sin', 'hour_cos', 'is_workday',
                'is_morning_peak', 'is_evening_peak', 'is_valley'
            ]
            
            corr_matrix = data[feature_cols].corr()
            
            corr_fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0
            ))
            
            corr_fig.update_layout(
                title='特征相关性热力图',
                template=self.theme
            )
            
            corr_path = Path(self.output_dirs['features']) / 'feature_correlation.html'
            corr_fig.write_html(str(corr_path))
            
        except Exception as e:
            self.logger.error(f"生成补充图表时出错: {str(e)}") 
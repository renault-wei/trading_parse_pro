from typing import Dict, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime
import numpy as np
from pathlib import Path

class ReportGenerator:
    """报告生成器"""
    
    def __init__(self, theme: str = 'plotly_dark'):
        self.theme = theme
        
    def generate_seasonal_report(self, analysis_results: Dict):
        """生成季节性分析报告"""
        # 创建子图
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                '日内收益模式', '日内波动率',
                '月度收益模式', '月度波动率',
                '年度收益模式', '年度波动率'
            )
        )
        
        # 添加日内模式图表
        daily_returns = analysis_results['daily_patterns']['returns']
        daily_volatility = analysis_results['daily_patterns']['volatility']
        
        fig.add_trace(
            go.Bar(x=daily_returns.index, y=daily_returns.values, name='小时收益率'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=daily_volatility.index, y=daily_volatility.values, name='小时波动率'),
            row=1, col=2
        )
        
        # 添加月度模式图表
        monthly_returns = analysis_results['monthly_patterns']['returns']
        monthly_volatility = analysis_results['monthly_patterns']['volatility']
        
        fig.add_trace(
            go.Bar(x=monthly_returns.index, y=monthly_returns.values, name='月度收益率'),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(x=monthly_volatility.index, y=monthly_volatility.values, name='月度波动率'),
            row=2, col=2
        )
        
        # 添加年度模式图表
        yearly_returns = analysis_results['yearly_patterns']['returns']
        yearly_volatility = analysis_results['yearly_patterns']['volatility']
        
        fig.add_trace(
            go.Bar(x=yearly_returns.index, y=yearly_returns.values, name='年度收益率'),
            row=3, col=1
        )
        fig.add_trace(
            go.Bar(x=yearly_volatility.index, y=yearly_volatility.values, name='年度波动率'),
            row=3, col=2
        )
        
        # 更新布局
        fig.update_layout(
            height=1200,
            width=1000,
            title_text="季节性分析报告",
            template=self.theme,
            showlegend=True
        )
        
        return fig 

class BacktestReportGenerator:
    """回测报告生成器"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.output_dir = Path(config.get('OUTPUT', 'charts_dir'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_monthly_returns_report(self, results: dict, strategy_name: str):
        """生成月度收益曲线报告"""
        try:
            # 获取每日统计数据
            daily_stats = results['daily_stats']
            
            # 确保时间索引格式正确
            if not isinstance(daily_stats.index, pd.DatetimeIndex):
                daily_stats.index = pd.to_datetime(daily_stats.index)
                
            # 按月份分组
            monthly_groups = daily_stats.groupby(daily_stats.index.to_period('M'))
            
            # 创建子图
            n_months = len(monthly_groups)
            n_rows = (n_months + 2) // 3  # 每行3个图
            
            fig = make_subplots(
                rows=n_rows,
                cols=3,
                subplot_titles=[str(month) for month in monthly_groups.groups.keys()],
                vertical_spacing=0.1,
                horizontal_spacing=0.05
            )
            
            # 为每个月创建子图
            for i, (month, data) in enumerate(monthly_groups):
                row = i // 3 + 1
                col = i % 3 + 1
                
                # 计算累积收益率
                cumulative_returns = (1 + data['daily_returns']).cumprod()
                
                # 添加收益曲线
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=cumulative_returns,
                        name=f'{month}',
                        line=dict(width=2),
                        showlegend=True
                    ),
                    row=row,
                    col=col
                )
                
                # 添加最大回撤阴影
                cummax = cumulative_returns.cummax()
                drawdowns = cumulative_returns / cummax - 1
                max_drawdown = drawdowns.min()
                
                # 添加最大回撤标注
                fig.add_annotation(
                    x=data.index[0],
                    y=cumulative_returns.iloc[0],
                    text=f'最大回撤: {max_drawdown:.2%}',
                    showarrow=False,
                    row=row,
                    col=col
                )
                
            # 更新布局
            fig.update_layout(
                title=f'{strategy_name} 月度收益曲线',
                height=300 * n_rows,
                width=1200,
                showlegend=True,
                template='plotly_dark'
            )
            
            # 更新所有子图的y轴格式
            fig.update_yaxes(tickformat='.2%')
            
            # 保存HTML文件
            output_file = self.output_dir / f'monthly_returns_{strategy_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
            fig.write_html(str(output_file))
            
            # 生成汇总统计
            self._generate_summary_stats(results, strategy_name)
            
            self.logger.info(f"月度收益报告已生成: {output_file}")
            
        except Exception as e:
            self.logger.error(f"生成月度收益报告时出错: {str(e)}")
            raise
            
    def _generate_summary_stats(self, results: dict, strategy_name: str):
        """生成汇总统计报告"""
        try:
            daily_stats = results['daily_stats']
            
            # 计算月度统计
            monthly_returns = daily_stats['daily_returns'].resample('M').apply(
                lambda x: (1 + x).prod() - 1
            )
            
            # 创建统计图
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=[
                    '月度收益分布',
                    '收益率散点图',
                    '回撤分析',
                    '滚动夏普比率'
                ]
            )
            
            # 月度收益分布
            fig.add_trace(
                go.Histogram(
                    x=monthly_returns,
                    name='月度收益分布',
                    nbinsx=20
                ),
                row=1,
                col=1
            )
            
            # 收益率散点图
            fig.add_trace(
                go.Scatter(
                    x=daily_stats.index,
                    y=daily_stats['daily_returns'],
                    mode='markers',
                    name='日收益率',
                    marker=dict(size=4)
                ),
                row=1,
                col=2
            )
            
            # 回撤分析
            cumulative_returns = (1 + daily_stats['daily_returns']).cumprod()
            drawdowns = cumulative_returns / cumulative_returns.cummax() - 1
            
            fig.add_trace(
                go.Scatter(
                    x=daily_stats.index,
                    y=drawdowns,
                    name='回撤',
                    fill='tozeroy'
                ),
                row=2,
                col=1
            )
            
            # 滚动夏普比率 (252交易日窗口)
            rolling_returns = daily_stats['daily_returns'].rolling(252)
            rolling_sharpe = (
                np.sqrt(252) * rolling_returns.mean() / rolling_returns.std()
            )
            
            fig.add_trace(
                go.Scatter(
                    x=daily_stats.index,
                    y=rolling_sharpe,
                    name='滚动夏普比率'
                ),
                row=2,
                col=2
            )
            
            # 更新布局
            fig.update_layout(
                title=f'{strategy_name} 策略统计分析',
                height=800,
                width=1200,
                showlegend=True,
                template='plotly_dark'
            )
            
            # 保存HTML文件
            output_file = self.output_dir / f'strategy_stats_{strategy_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
            fig.write_html(str(output_file))
            
            self.logger.info(f"策略统计报告已生成: {output_file}")
            
        except Exception as e:
            self.logger.error(f"生成统计报告时出错: {str(e)}")
            raise
            
    def generate_performance_table(self, results: dict, start_date: str, end_date: str, data_stats: dict):
        """生成收益率表现报告"""
        try:
            # 创建图表
            fig = go.Figure()
            
            # 添加回测信息表格
            info_header = ['回测信息', '值']
            info_data = [
                ['回测起始日期', start_date],
                ['回测结束日期', end_date],
                ['总记录数', data_stats.get('total_records', '')],
                ['日期范围', f"{data_stats.get('date_range_start', '')} 到 {data_stats.get('date_range_end', '')}"],
                ['小时范围', f"{data_stats.get('hour_range_start', '')} 到 {data_stats.get('hour_range_end', '')}"]
            ]
            
            # 添加回测信息表格
            fig.add_trace(
                go.Table(
                    header=dict(
                        values=info_header,
                        fill_color='darkgrey',
                        align='left',
                        font=dict(color='white', size=12)
                    ),
                    cells=dict(
                        values=list(zip(*info_data)),
                        fill_color='black',
                        align='left',
                        font=dict(color='white', size=12)
                    ),
                    columnwidth=[2, 3]
                )
            )
            
            # 计算性能指标
            performance_data = self._calculate_performance_metrics(results)
            
            # 创建性能指标表格
            header = ['品种', 'AR', 'SR', 'MD', 'MAR', 'max_position(W)', 'margin(W)', 'profit']
            
            # 格式化性能数据
            cells_data = [
                performance_data['symbol'],
                performance_data['ar'].map('{:.3f}'.format),
                performance_data['sr'].map('{:.3f}'.format),
                performance_data['md'].map('{:.3f}'.format),
                performance_data['mar'].map('{:.3f}'.format),
                performance_data['max_position'].map('{:.1f}'.format),
                performance_data['margin'].map('{:.1f}'.format),
                performance_data['profit'].map('{:.1f}'.format)
            ]
            
            # 添加性能指标表格
            fig.add_trace(
                go.Table(
                    header=dict(
                        values=header,
                        fill_color='darkgrey',
                        align='center',
                        font=dict(color='white', size=12)
                    ),
                    cells=dict(
                        values=cells_data,
                        fill_color='black',
                        align='center',
                        font=dict(color='white', size=12)
                    )
                )
            )
            
            # 更新布局
            fig.update_layout(
                title='收益率表现',
                template='plotly_dark',
                height=800,
                width=1200,
                showlegend=False
            )
            
            # 保存HTML文件
            output_path = self.output_dir / 'performance_table.html'
            fig.write_html(str(output_path))
            
            self.logger.info(f"收益率表现报告已生成: {output_path}")
            
        except Exception as e:
            self.logger.error(f"生成收益率表现报告时出错: {str(e)}")
            raise
            
    def _calculate_performance_metrics(self, results: dict) -> pd.DataFrame:
        """计算性能指标"""
        try:
            daily_stats = results['daily_stats']
            
            # 计算各项指标
            ar = (1 + results['total_returns']) ** (365 / results.get('trading_days', 252)) - 1
            sr = results['sharpe_ratio']
            md = abs(results['max_drawdown'])
            mar = ar / md if md != 0 else 0
            
            # 创建性能数据DataFrame
            performance_data = pd.DataFrame({
                'symbol': ['default'],  # 如果有多个品种，这里需要修改
                'ar': [ar * 100],  # 转换为百分比
                'sr': [sr],
                'md': [md * 100],  # 转换为百分比
                'mar': [mar],
                'max_position': [results.get('max_position', 0) / 10000],  # 转换为万
                'margin': [results.get('margin_used', 0) / 10000],  # 转换为万
                'profit': [results.get('total_profit', 0)]
            })
            
            return performance_data
            
        except Exception as e:
            self.logger.error(f"计算性能指标时出错: {str(e)}")
            raise
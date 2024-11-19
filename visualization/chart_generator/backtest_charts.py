import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from pathlib import Path
from datetime import datetime
from .base import BaseChartGenerator

class BacktestChartGenerator(BaseChartGenerator):
    """回测分析图表生成器"""
    
    def generate_backtest_results(self, backtest_results: dict):
        """生成回测结果图表"""
        try:
            # 首先确保所有数据使用相同的索引
            common_index = backtest_results['price'].index
            
            # 重新索引所有数据
            aligned_data = self._align_data(backtest_results, common_index)
            
            # 创建主图表
            fig = self._create_main_plot(aligned_data, common_index)
            
            # 保存图表
            output_path = Path(self.output_dirs['backtest']) / 'backtest_results.html'
            fig.write_html(str(output_path))
            
            # 生成回测报告
            self.generate_backtest_report(backtest_results)
            
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"生成回测图表时出错: {str(e)}")
            return None
            
    def _align_data(self, backtest_results: dict, common_index: pd.DatetimeIndex) -> dict:
        """对齐所有数据到相同的索引"""
        return {
            'price': backtest_results['price'].reindex(common_index),
            'signals': backtest_results['signals'].reindex(common_index),
            'returns': backtest_results['returns'].reindex(common_index),
            'drawdown': backtest_results['drawdown'].reindex(common_index),
            'positions': backtest_results['positions'].reindex(common_index) 
                        if 'positions' in backtest_results else None
        }
        
    def _create_main_plot(self, aligned_data: dict, common_index: pd.DatetimeIndex) -> go.Figure:
        """创建主回测图表"""
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=(
                '价格和交易信号',
                '累计收益率(%)',
                '回撤(%)',
                '持仓'
            ),
            vertical_spacing=0.05,
            row_heights=[0.4, 0.2, 0.2, 0.2]
        )
        
        # 1. 价格和交易信号
        self._add_price_signals_plot(fig, aligned_data, common_index)
        
        # 2. 累计收益率
        self._add_returns_plot(fig, aligned_data, common_index)
        
        # 3. 回撤
        self._add_drawdown_plot(fig, aligned_data, common_index)
        
        # 4. 持仓
        if aligned_data['positions'] is not None:
            self._add_positions_plot(fig, aligned_data, common_index)
            
        # 更新布局
        fig.update_layout(
            height=1200,
            title_text='回测分析结果',
            showlegend=True,
            template=self.theme,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # 更新轴标签
        self._update_axes_labels(fig)
        
        return fig
        
    def _add_price_signals_plot(self, fig: go.Figure, data: dict, index: pd.DatetimeIndex):
        """添加价格和信号图"""
        # 价格线
        fig.add_trace(
            go.Scatter(
                x=index,
                y=data['price'],
                name='价格',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # 买入点
        buy_signals = data['signals'] == 1
        if buy_signals.any():
            buy_prices = data['price'][buy_signals]
            fig.add_trace(
                go.Scatter(
                    x=buy_prices.index,
                    y=buy_prices,
                    mode='markers',
                    name='买入信号',
                    marker=dict(color='green', size=8, symbol='triangle-up')
                ),
                row=1, col=1
            )
            
        # 卖出点
        sell_signals = data['signals'] == -1
        if sell_signals.any():
            sell_prices = data['price'][sell_signals]
            fig.add_trace(
                go.Scatter(
                    x=sell_prices.index,
                    y=sell_prices,
                    mode='markers',
                    name='卖出信号',
                    marker=dict(color='red', size=8, symbol='triangle-down')
                ),
                row=1, col=1
            )
            
    def _add_returns_plot(self, fig: go.Figure, data: dict, index: pd.DatetimeIndex):
        """添加收益率图"""
        cumulative_returns = (1 + data['returns']).cumprod() - 1
        fig.add_trace(
            go.Scatter(
                x=index,
                y=cumulative_returns * 100,
                name='累计收益率',
                line=dict(color='purple')
            ),
            row=2, col=1
        )
        
    def _add_drawdown_plot(self, fig: go.Figure, data: dict, index: pd.DatetimeIndex):
        """添加回撤图"""
        fig.add_trace(
            go.Scatter(
                x=index,
                y=data['drawdown'] * 100,
                name='回撤',
                line=dict(color='red'),
                fill='tozeroy'
            ),
            row=3, col=1
        )
        
    def _add_positions_plot(self, fig: go.Figure, data: dict, index: pd.DatetimeIndex):
        """添加持仓图"""
        fig.add_trace(
            go.Scatter(
                x=index,
                y=data['positions'],
                name='持仓',
                line=dict(color='orange')
            ),
            row=4, col=1
        )
        
    def _update_axes_labels(self, fig: go.Figure):
        """更新坐标轴标签"""
        fig.update_xaxes(title_text='时间', row=4, col=1)
        fig.update_yaxes(title_text='价格', row=1, col=1)
        fig.update_yaxes(title_text='收益率(%)', row=2, col=1)
        fig.update_yaxes(title_text='回撤(%)', row=3, col=1)
        fig.update_yaxes(title_text='持仓', row=4, col=1)
        
    def generate_backtest_report(self, backtest_results: dict):
        """生成回测分析报告"""
        try:
            # 计算交易次数
            total_trades = (backtest_results['signals'].diff() != 0).sum() // 2
            
            # 生成HTML报告
            html_content = self._create_report_html(backtest_results, total_trades)
            
            # 保存报告
            output_path = Path(self.output_dirs['backtest']) / 'backtest_report.html'
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            self.logger.info(f"回测报告已生成: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"生成回测分析报告时出错: {str(e)}")
            return None
            
    def _create_report_html(self, results: dict, total_trades: int) -> str:
        """创建HTML报告内容"""
        return f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
                .metric {{ margin: 10px 0; }}
                .summary {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h2>回测分析报告</h2>
            
            <div class="summary">
                <h3>总体表现</h3>
                <div class="metric">
                    总收益率: <span class="{self._get_color_class(results['total_returns'])}">{results['total_returns']:.2%}</span>
                </div>
                <div class="metric">
                    夏普比率: <span class="{self._get_color_class(results['sharpe_ratio'])}">{results['sharpe_ratio']:.2f}</span>
                </div>
                <div class="metric">
                    最大回撤: <span class="negative">{results['max_drawdown']:.2%}</span>
                </div>
                <div class="metric">
                    总交易次数: {total_trades}
                </div>
            </div>
            
            <div class="details">
                <h3>月度表现</h3>
                {self._generate_monthly_stats(results['returns'])}
            </div>
        </body>
        </html>
        """
        
    def _get_color_class(self, value: float) -> str:
        """获取颜色类名"""
        if isinstance(value, pd.Series):
            value = value.iloc[-1]
        return 'positive' if value > 0 else 'negative'
        
    def _generate_monthly_stats(self, returns: pd.Series) -> str:
        """生成月度统计信息"""
        try:
            monthly_returns = returns.resample('M').sum()
            
            if monthly_returns.empty:
                return "<p>无可用的月度统计数据</p>"
                
            table_rows = []
            for date, value in monthly_returns.items():
                if isinstance(value, pd.Series):
                    value = value.iloc[0]
                elif pd.isna(value):
                    value = 0.0
                    
                color_class = 'positive' if value > 0 else 'negative'
                row = f"""
                <tr>
                    <td>{date.strftime('%Y-%m')}</td>
                    <td class="{color_class}">{value:.2%}</td>
                </tr>
                """
                table_rows.append(row)
                
            if not table_rows:
                return "<p>无有效的月度统计数据</p>"
                
            return f"""
            <table border="1" cellpadding="5" cellspacing="0">
                <tr>
                    <th>月份</th>
                    <th>收益率</th>
                </tr>
                {''.join(table_rows)}
            </table>
            """
            
        except Exception as e:
            self.logger.error(f"生成月度统计时出错: {str(e)}")
            return "<p>生成月度统计失败</p>" 
import pandas as pd
from datetime import datetime
from visualization.report_generator import BacktestReportGenerator
from core.backtest.engine import BacktestEngine
from core.analysis.backtest_analysis import BacktestAnalyzer
import os
from pathlib import Path

class BacktestManager:
    """回测模块管理器"""
    
    def __init__(self, config, logger, chart_generator):
        self.config = config
        self.logger = logger
        self.chart_generator = chart_generator
        self.report_generator = BacktestReportGenerator(config, logger)
        self.backtest_analyzer = BacktestAnalyzer(logger)
        self.data = None
        
    def run_backtest(self, data: pd.DataFrame, strategy) -> dict:
        """运行回测"""
        try:
            self.logger.info("开始运行回测...")
            self.data = data
            
            # 获取回测配置
            backtest_config = {
                'initial_capital': float(self.config.get('TRADING', 'initial_capital')),
                'commission_rate': float(self.config.get('TRADING', 'commission_rate')),
                'slippage': float(self.config.get('TRADING', 'slippage'))
            }
            self.logger.info(f"回测配置: {backtest_config}")
            
            # 初始化回测引擎
            engine = BacktestEngine(data, backtest_config)
            self.logger.info("回测引擎初始化完成")
            
            # 运行回测
            self.logger.info("开始执行回测策略...")
            engine.run_backtest(strategy)
            
            # 获取回测结果
            results = engine.get_performance_metrics()
            trades = engine.get_trades_history()
            self.logger.info(f"回测完成，共执行 {len(trades)} 笔交易")
            
            # 检查和补充必要的字段
            required_fields = [
                'initial_capital', 'final_capital', 'total_returns', 
                'annual_returns', 'sharpe_ratio', 'max_drawdown',
                'total_trades', 'trading_days'
            ]
            
            # 打印结果摘要用于调试
            self.logger.debug("回测结果摘要:")
            for key, value in results.items():
                if not isinstance(value, (pd.DataFrame, pd.Series)):
                    self.logger.debug(f"{key}: {value}")
            
            # 转换交易记录为DataFrame格式
            trades_df = pd.DataFrame([
                {
                    'timestamp': t.timestamp,
                    'direction': t.direction,
                    'quantity': t.quantity,
                    'price': t.price,
                    'pnl': getattr(t, 'pnl', 0),
                    'commission': t.commission,
                    'slippage': t.slippage
                }
                for t in trades
            ])
            
            # 确保所有必要的字段都存在
            processed_results = {
                'initial_capital': results.get('initial_capital', backtest_config['initial_capital']),
                'final_capital': results.get('final_capital', backtest_config['initial_capital']),
                'total_returns': results.get('total_returns', 0.0),
                'annual_returns': results.get('annual_returns', 0.0),
                'sharpe_ratio': results.get('sharpe_ratio', 0.0),
                'max_drawdown': results.get('max_drawdown', 0.0),
                'total_trades': results.get('total_trades', 0),
                'trading_days': results.get('trading_days', 0),
                'win_rate': results.get('win_rate', 0.0),
                'max_position': results.get('max_position', 0.0),
                'margin': results.get('margin', 0.0),
                'trades': trades_df,
                'daily_stats': results.get('daily_stats', pd.DataFrame())
            }
            
            # 生成回测报告和图表
            self.logger.info("开始生成回测报告和图表...")
            self.chart_generator.generate_backtest_reports(processed_results)
            
            # 生成信号日志报告
            self.logger.info("开始生成信号日志报告...")
            self._generate_signal_log_report(trades_df)
            
            # 生成详细报告
            self.logger.info("开始生成详细分析报告...")
            self.generate_report(processed_results, strategy.__class__.__name__)
            
            self.logger.info("回测分析完成")
            return processed_results
            
        except Exception as e:
            self.logger.error(f"运行回测时出错: {str(e)}", exc_info=True)
            raise
            
    def generate_report(self, results: dict, strategy_name: str = None):
        """生成回测报告"""
        try:
            if self.data is None:
                raise ValueError("没有可用的回测数据")
                
            # 获取数据统计信息
            data_stats = {
                'total_records': len(self.data),
                'date_range_start': self.data.index.min().strftime('%Y-%m-%d %H:%M:%S'),
                'date_range_end': self.data.index.max().strftime('%Y-%m-%d %H:%M:%S'),
                'hour_range': f"{self.data.index.hour.min()}-{self.data.index.hour.max()}"
            }
            
            # 添加策略信息
            if strategy_name:
                data_stats['strategy_name'] = strategy_name
                
            # 生成月度收益曲线报告
            self.report_generator.generate_monthly_returns_report(
                results,
                strategy_name
            )
            
            # 生成收益率表现报告
            self.report_generator.generate_performance_table(
                results,
                self.data.index.min().strftime('%Y-%m-%d'),
                self.data.index.max().strftime('%Y-%m-%d'),
                data_stats
            )
            
            # 分析交易时段表现
            time_patterns = self._analyze_time_patterns(results['trades'])
            self._generate_time_pattern_report(time_patterns)
            
            self.logger.info("回测报告生成完成")
            
        except Exception as e:
            self.logger.error(f"生成回测报告时出错: {str(e)}")
            raise
            
    def _analyze_time_patterns(self, trades_df: pd.DataFrame) -> dict:
        """分析交易时段表现"""
        try:
            self.logger.info("开始分析交易时段表现...")
            self.logger.debug(f"交易数据形状: {trades_df.shape}")
            self.logger.debug(f"交易数据列: {trades_df.columns.tolist()}")
            
            # 如果trades_df为空，返回空的统计结果
            if trades_df.empty:
                self.logger.warning("没有交易记录，返回空的统计结果")
                return {
                    'hourly_stats': pd.DataFrame(),
                    'weekday_stats': pd.DataFrame()
                }
                
            # 确保trades_df有正确的列
            if 'timestamp' not in trades_df.columns:
                self.logger.warning("交易数据缺少timestamp列，尝试使用索引")
                if isinstance(trades_df.index, pd.DatetimeIndex):
                    trades_df['timestamp'] = trades_df.index
                else:
                    self.logger.error("无法获取时间信息")
                    return {
                        'hourly_stats': pd.DataFrame(),
                        'weekday_stats': pd.DataFrame()
                    }
            
            # 确保timestamp列是datetime类型
            if not pd.api.types.is_datetime64_any_dtype(trades_df['timestamp']):
                self.logger.info("转换timestamp列为datetime类型")
                trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            
            try:
                # 按小时统计
                hourly_stats = trades_df.groupby(trades_df['timestamp'].dt.hour).agg({
                    'pnl': ['mean', 'sum', 'count'],
                    'quantity': 'mean'
                }).round(2)
                self.logger.debug(f"小时统计完成: {hourly_stats.shape}")
            except Exception as e:
                self.logger.error(f"计算小时统计时出错: {str(e)}")
                hourly_stats = pd.DataFrame()
            
            try:
                # 按工作日统计
                weekday_stats = trades_df.groupby(trades_df['timestamp'].dt.weekday).agg({
                    'pnl': ['mean', 'sum', 'count'],
                    'quantity': 'mean'
                }).round(2)
                self.logger.debug(f"工作日统计完成: {weekday_stats.shape}")
            except Exception as e:
                self.logger.error(f"计算工作日统计时出错: {str(e)}")
                weekday_stats = pd.DataFrame()
            
            return {
                'hourly_stats': hourly_stats,
                'weekday_stats': weekday_stats
            }
            
        except Exception as e:
            self.logger.error(f"分析交易时段表现时出错: {str(e)}")
            # 返回空的统计结果而不是抛出异常
            return {
                'hourly_stats': pd.DataFrame(),
                'weekday_stats': pd.DataFrame()
            }
            
    def _generate_time_pattern_report(self, time_patterns: dict):
        """生成时段分析报告"""
        try:
            output_dir = Path(self.config.get('OUTPUT', 'charts_dir'))
            report_path = output_dir / 'time_pattern_analysis.html'
            
            # 使用更简单的HTML模板
            html_content = f'''<!DOCTYPE html>
<html>
<head>
<title>交易时段分析报告</title>
<style>
body {{ margin: 20px; }}
table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
th {{ background-color: #f2f2f2; }}
.section {{ margin: 20px 0; }}
</style>
</head>
<body>
<h1 style="text-align: center;">交易时段分析报告</h1>

<div class="section">
<h2>小时统计</h2>
{time_patterns['hourly_stats'].to_html()}
</div>

<div class="section">
<h2>工作日统计</h2>
{time_patterns['weekday_stats'].to_html()}
</div>
</body>
</html>'''
            
            # 写入文件
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            self.logger.info(f"时段分析报告已生成: {report_path}")
            
        except Exception as e:
            self.logger.error(f"生成时段分析报告时出错: {str(e)}")
            raise
            
    def _cleanup_old_reports(self):
        """清理旧的回测报告"""
        try:
            output_dir = Path(self.config.get('OUTPUT', 'charts_dir'))
            if output_dir.exists():
                # 删除旧的回测报告文件
                for pattern in ['monthly_returns_*.html', 'strategy_stats_*.html', 
                              'time_pattern_*.html', 'backtest_*.html']:
                    for file in output_dir.glob(pattern):
                        file.unlink()
                        
            self.logger.info("已清理旧的回测报告文件")
            
        except Exception as e:
            self.logger.warning(f"清理旧报告文件时出错: {str(e)}")
            
    def _generate_signal_log_report(self, trades_df: pd.DataFrame):
        """生成信号日志报告"""
        try:
            # 检查交易记录是否为空
            if trades_df.empty:
                self.logger.warning("没有交易记录，跳过生成信号日志报告")
                return
                
            output_dir = Path(self.config.get('OUTPUT', 'charts_dir'))
            report_path = output_dir / 'signal_log_analysis.html'
            
            # 检查必要的列是否存在
            required_columns = ['timestamp', 'price', 'direction', 'quantity', 'pnl', 'commission', 'slippage']
            missing_columns = [col for col in required_columns if col not in trades_df.columns]
            
            if missing_columns:
                self.logger.warning(f"交易记录缺少以下列: {missing_columns}")
                # 为缺失的列添加默认值
                for col in missing_columns:
                    if col == 'timestamp':
                        trades_df['timestamp'] = pd.to_datetime('now')
                    elif col in ['price', 'quantity', 'pnl', 'commission', 'slippage']:
                        trades_df[col] = 0.0
                    elif col == 'direction':
                        trades_df['direction'] = 'unknown'
            
            # 创建信号日志表格
            signal_log = pd.DataFrame({
                '时间': trades_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S'),
                '价格': trades_df['price'].round(2),
                '方向': trades_df['direction'].map({'buy': '买入', 'sell': '卖出', 'unknown': '未知'}),
                '数量': trades_df['quantity'],
                '盈亏': trades_df['pnl'].round(2),
                '手续费': trades_df['commission'].round(2),
                '滑点': trades_df['slippage'].round(2)
            })
            
            # 添加累计盈亏列
            signal_log['累计盈亏'] = signal_log['盈亏'].cumsum().round(2)
            
            # 创建HTML报告
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("""
                <html>
                <head>
                    <title>交易信号日志分析</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 20px; }
                        .header { margin: 20px 0; }
                        table { 
                            border-collapse: collapse; 
                            width: 100%;
                            margin-top: 20px;
                        }
                        th, td { 
                            border: 1px solid #ddd; 
                            padding: 8px; 
                            text-align: left; 
                        }
                        th { 
                            background-color: #f2f2f2;
                            position: sticky;
                            top: 0;
                        }
                        tr:nth-child(even) { background-color: #f9f9f9; }
                        tr:hover { background-color: #f5f5f5; }
                        .positive { color: #4CAF50; }
                        .negative { color: #f44336; }
                        .summary { 
                            margin: 20px 0;
                            padding: 10px;
                            background-color: #f8f9fa;
                            border-radius: 4px;
                        }
                    </style>
                </head>
                <body>
                    <h1>交易信号日志分析</h1>
                    
                    <div class="summary">
                        <h2>交易统计摘要</h2>
                        <p>总交易次数: {}</p>
                        <p>总盈亏: {:.2f}</p>
                        <p>平均盈亏: {:.2f}</p>
                        <p>最大单笔盈利: {:.2f}</p>
                        <p>最大单笔亏损: {:.2f}</p>
                    </div>
                    
                    <div class="header">
                        <h2>详细交易记录</h2>
                    </div>
                    
                    {}
                </body>
                </html>
                """.format(
                    len(signal_log),
                    signal_log['盈亏'].sum(),
                    signal_log['盈亏'].mean(),
                    signal_log['盈亏'].max(),
                    signal_log['盈亏'].min(),
                    signal_log.to_html(
                        classes='display',
                        index=False,
                        float_format=lambda x: '{:.2f}'.format(x) if isinstance(x, float) else x
                    )
                ))
                
            self.logger.info(f"信号日志分析报告已生成: {report_path}")
            
        except Exception as e:
            self.logger.error(f"生成信号日志分析报告时出错: {str(e)}")
            self.logger.debug(f"交易数据形状: {trades_df.shape if not trades_df.empty else '空DataFrame'}")
            self.logger.debug(f"交易数据列: {trades_df.columns.tolist() if not trades_df.empty else '无'}")
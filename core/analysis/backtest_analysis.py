"""回测分析模块"""
import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime

class BacktestAnalyzer:
    """回测分析器"""
    
    def __init__(self, logger):
        self.logger = logger
        
    def analyze_performance(self, trades: List, daily_stats: pd.DataFrame) -> Dict:
        """
        分析回测性能
        
        Args:
            trades: 交易记录列表
            daily_stats: 每日统计数据
            
        Returns:
            Dict: 性能分析结果
        """
        try:
            # 计算基础指标
            total_trades = len(trades)
            winning_trades = sum(1 for t in trades if t.pnl > 0)
            losing_trades = sum(1 for t in trades if t.pnl < 0)
            
            # 计算胜率
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # 计算收益相关指标
            total_pnl = sum(t.pnl for t in trades)
            avg_win = np.mean([t.pnl for t in trades if t.pnl > 0]) if winning_trades > 0 else 0
            avg_loss = np.mean([t.pnl for t in trades if t.pnl < 0]) if losing_trades > 0 else 0
            
            # 计算收益风险比
            profit_factor = (
                abs(sum(t.pnl for t in trades if t.pnl > 0)) /
                abs(sum(t.pnl for t in trades if t.pnl < 0))
                if losing_trades > 0 else float('inf')
            )
            
            # 计算最大回撤
            cumulative_returns = (1 + daily_stats['daily_returns']).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = cumulative_returns / rolling_max - 1
            max_drawdown = drawdowns.min()
            
            # 计算夏普比率
            risk_free_rate = 0.02  # 假设无风险利率为2%
            excess_returns = daily_stats['daily_returns'] - risk_free_rate/252
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'daily_stats': daily_stats
            }
            
        except Exception as e:
            self.logger.error(f"分析回测性能时出错: {str(e)}")
            raise
            
    def analyze_trade_patterns(self, trades: List) -> Dict:
        """
        分析交易模式
        
        Args:
            trades: 交易记录列表
            
        Returns:
            Dict: 交易模式分析结果
        """
        try:
            self.logger.info("开始分析交易模式...")
            
            # 按时间段分析
            trades_df = pd.DataFrame([
                {
                    'timestamp': t.timestamp,
                    'direction': t.direction,
                    'quantity': t.quantity,
                    'price': t.price,
                    'pnl': getattr(t, 'pnl', 0),  # 使用getattr处理可能不存在的属性
                    'commission': getattr(t, 'commission', 0),
                    'slippage': getattr(t, 'slippage', 0)
                }
                for t in trades
            ])
            
            if trades_df.empty:
                self.logger.warning("没有交易记录")
                return {
                    'hourly_patterns': pd.DataFrame(),
                    'weekday_patterns': pd.DataFrame(),
                    'monthly_patterns': pd.DataFrame(),
                    'direction_patterns': pd.DataFrame(),
                    'trades_df': trades_df
                }
            
            self.logger.debug(f"交易数据形状: {trades_df.shape}")
            self.logger.debug(f"交易数据列: {trades_df.columns.tolist()}")
            
            # 添加时间特征
            trades_df['hour'] = trades_df['timestamp'].dt.hour
            trades_df['weekday'] = trades_df['timestamp'].dt.weekday
            trades_df['month'] = trades_df['timestamp'].dt.month
            
            self.logger.info("开始计算时段统计...")
            
            # 分时段分析
            try:
                hourly_stats = trades_df.groupby('hour').agg({
                    'pnl': ['mean', 'sum', 'count'],
                    'quantity': 'mean'
                }).round(2)
                self.logger.debug(f"小时统计完成: {hourly_stats.shape}")
            except Exception as e:
                self.logger.error(f"计算小时统计时出错: {str(e)}")
                hourly_stats = pd.DataFrame()
            
            # 分星期分析
            try:
                weekday_stats = trades_df.groupby('weekday').agg({
                    'pnl': ['mean', 'sum', 'count'],
                    'quantity': 'mean'
                }).round(2)
                self.logger.debug(f"星期统计完成: {weekday_stats.shape}")
            except Exception as e:
                self.logger.error(f"计算星期统计时出错: {str(e)}")
                weekday_stats = pd.DataFrame()
            
            # 分月份分析
            try:
                monthly_stats = trades_df.groupby('month').agg({
                    'pnl': ['mean', 'sum', 'count'],
                    'quantity': 'mean'
                }).round(2)
                self.logger.debug(f"月度统计完成: {monthly_stats.shape}")
            except Exception as e:
                self.logger.error(f"计算月度统计时出错: {str(e)}")
                monthly_stats = pd.DataFrame()
            
            # 交易方向分析
            try:
                direction_stats = trades_df.groupby('direction').agg({
                    'pnl': ['mean', 'sum', 'count'],
                    'quantity': 'mean'
                }).round(2)
                self.logger.debug(f"方向统计完成: {direction_stats.shape}")
            except Exception as e:
                self.logger.error(f"计算方向统计时出错: {str(e)}")
                direction_stats = pd.DataFrame()
            
            self.logger.info("交易模式分析完成")
            
            return {
                'hourly_patterns': hourly_stats,
                'weekday_patterns': weekday_stats,
                'monthly_patterns': monthly_stats,
                'direction_patterns': direction_stats,
                'trades_df': trades_df
            }
            
        except Exception as e:
            self.logger.error(f"分析交易模式时出错: {str(e)}")
            raise
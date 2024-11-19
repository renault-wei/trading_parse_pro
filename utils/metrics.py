import numpy as np
import pandas as pd
from typing import Dict, Optional

class PerformanceMetrics:
    """性能指标计算"""
    
    @staticmethod
    def calculate_returns_metrics(returns: pd.Series) -> Dict[str, float]:
        """
        计算收益率相关指标
        
        Args:
            returns: 收益率序列
            
        Returns:
            包含各项指标的字典
        """
        metrics = {}
        
        # 累计收益
        metrics['total_return'] = (1 + returns).prod() - 1
        
        # 年化收益
        n_years = len(returns) / 252
        metrics['annual_return'] = (1 + metrics['total_return']) ** (1/n_years) - 1
        
        # 波动率
        metrics['volatility'] = returns.std() * np.sqrt(252)
        
        # 夏普比率
        risk_free_rate = 0.02  # 假设无风险利率为2%
        excess_returns = returns - risk_free_rate/252
        metrics['sharpe_ratio'] = np.sqrt(252) * excess_returns.mean() / returns.std()
        
        # 最大回撤
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns / rolling_max - 1
        metrics['max_drawdown'] = drawdowns.min()
        
        return metrics
        
    @staticmethod
    def calculate_trading_metrics(
        trades: pd.DataFrame,
        capital: float
    ) -> Dict[str, float]:
        """
        计算交易相关指标
        
        Args:
            trades: 交易记录DataFrame
            capital: 初始资金
            
        Returns:
            包含各项指标的字典
        """
        metrics = {}
        
        # 胜率
        winning_trades = trades[trades['pnl'] > 0]
        metrics['win_rate'] = len(winning_trades) / len(trades)
        
        # 盈亏比
        avg_win = winning_trades['pnl'].mean()
        losing_trades = trades[trades['pnl'] < 0]
        avg_loss = abs(losing_trades['pnl'].mean())
        metrics['profit_factor'] = avg_win / avg_loss if avg_loss != 0 else np.inf
        
        # 资金利用率
        metrics['capital_utilization'] = trades['volume'].mean() / capital
        
        return metrics 
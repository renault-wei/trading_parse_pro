import pandas as pd
import numpy as np
from utils.logger import Logger

class IntradayStrategy:
    """日内交易策略"""
    
    def __init__(self, config=None):
        self.logger = Logger().get_logger()
        self.config = config
        self.factor_weights = {}
        
    def update_factor_weights(self, regression_results):
        """更新因子权重"""
        if regression_results and 'feature_importance' in regression_results:
            self.factor_weights = dict(zip(
                regression_results['feature_importance']['feature'],
                regression_results['feature_importance']['importance']
            ))
            self.logger.info("因子权重已更新")
            
    def generate_signals(self, data):
        """生成交易信号"""
        signals = pd.Series(index=data.index, data=0)  # 初始化信号序列
        returns = pd.Series(index=data.index, data=0.0)  # 初始化收益序列
        
        # 计算技术指标
        data['sma_short'] = data['price'].rolling(window=5).mean()
        data['sma_long'] = data['price'].rolling(window=20).mean()
        
        # 生成交易信号
        signals[data['sma_short'] > data['sma_long']] = 1  # 买入信号
        signals[data['sma_short'] < data['sma_long']] = -1  # 卖出信号
        
        # 计算收益
        price_returns = data['price'].pct_change()
        returns = signals.shift(1) * price_returns  # 根据前一个信号计算收益
        
        return {
            'signals': signals,
            'returns': returns,
            'positions': signals.copy(),
            'total_returns': returns.cumsum(),
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252),
            'max_drawdown': (returns.cumsum() - returns.cumsum().expanding().max()).min(),
            'total_trades': (signals.diff() != 0).sum() // 2
        }
            
    def run_backtest(self, data):
        """运行回测"""
        try:
            # 生成交易信号
            signals = self.generate_signals(data)
            
            # 计算收益
            returns = data['price'].pct_change() * signals.shift(1)
            
            # 计算累计收益
            equity_curve = (1 + returns).cumprod()
            
            # 统计交易次数
            trades = signals[signals != 0]
            
            # 返回回测结果
            results = {
                'returns': returns,
                'equity_curve': equity_curve,
                'trades': trades,
                'signals': signals
            }
            
            self.logger.info(f"回测完成，总交易次数: {len(trades)}")
            return results
            
        except Exception as e:
            self.logger.error(f"回测执行失败: {str(e)}")
            return None
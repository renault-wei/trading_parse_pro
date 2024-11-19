from typing import Dict, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

class VolatilityAnalyzer:
    """波动率分析器"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.results = {}
        
    def analyze(self):
        """执行波动率分析"""
        try:
            self.results = {
                'historical_volatility': self._calculate_historical_volatility(),
                'intraday_pattern': self._analyze_intraday_pattern(),
                'volatility_distribution': self._analyze_volatility_distribution(),
                'volatility_clustering': self._analyze_volatility_clustering(),
                'regime_analysis': self._analyze_volatility_regime()
            }
        except Exception as e:
            print(f"波动率分析错误: {str(e)}")
            
    def _calculate_historical_volatility(self) -> Dict:
        """计算历史波动率"""
        try:
            # 日度波动率
            daily_vol = self.data['returns'].std() * np.sqrt(252)
            
            # 月度波动率
            monthly_vol = self.data.resample('M')['returns'].std() * np.sqrt(12)
            
            # 滚动波动率
            rolling_vol = self.data['returns'].rolling(
                window=24
            ).std() * np.sqrt(252)
            
            return {
                'daily': daily_vol,
                'monthly': monthly_vol,
                'rolling': rolling_vol
            }
        except Exception as e:
            print(f"计算历史波动率错误: {str(e)}")
            return {}
            
    def _analyze_intraday_pattern(self) -> Dict:
        """分析日内波动率模式"""
        try:
            # 每小时波动率
            hourly_vol = self.data.groupby('hour')['returns'].std()
            
            # 高低峰时段波动率
            peak_hours_vol = self.data[
                self.data['is_morning_peak'] | self.data['is_evening_peak']
            ]['returns'].std()
            
            valley_hours_vol = self.data[
                self.data['is_valley']
            ]['returns'].std()
            
            return {
                'hourly_volatility': hourly_vol,
                'peak_hours_volatility': peak_hours_vol,
                'valley_hours_volatility': valley_hours_vol
            }
        except Exception as e:
            print(f"分析日内波动率模式错误: {str(e)}")
            return {}
            
    def _analyze_volatility_distribution(self) -> Dict:
        """分析波动率分布"""
        try:
            returns = self.data['returns'].dropna()
            
            # 计算描述性统计
            stats = {
                'mean': returns.mean(),
                'std': returns.std(),
                'skew': returns.skew(),
                'kurtosis': returns.kurtosis(),
                'percentiles': {
                    '1%': returns.quantile(0.01),
                    '5%': returns.quantile(0.05),
                    '95%': returns.quantile(0.95),
                    '99%': returns.quantile(0.99)
                }
            }
            
            return stats
        except Exception as e:
            print(f"分析波动率分布错误: {str(e)}")
            return {}
            
    def _analyze_volatility_clustering(self) -> Dict:
        """分析波动率聚集效应"""
        try:
            returns = self.data['returns'].dropna()
            abs_returns = abs(returns)
            
            # 计算自相关系数
            lags = range(1, 11)
            autocorr = [abs_returns.autocorr(lag=lag) for lag in lags]
            
            return {
                'lags': list(lags),
                'autocorrelation': autocorr
            }
        except Exception as e:
            print(f"分析波动率聚集效应错误: {str(e)}")
            return {}
            
    def _analyze_volatility_regime(self) -> Dict:
        """分析波动率区间"""
        try:
            rolling_vol = self.data['returns'].rolling(window=24).std() * np.sqrt(252)
            
            # 定义波动率区间
            low_vol_threshold = rolling_vol.quantile(0.33)
            high_vol_threshold = rolling_vol.quantile(0.67)
            
            # 标记波动率区间
            regimes = pd.Series(index=rolling_vol.index, dtype=str)
            regimes[rolling_vol <= low_vol_threshold] = 'low'
            regimes[(rolling_vol > low_vol_threshold) & (rolling_vol <= high_vol_threshold)] = 'medium'
            regimes[rolling_vol > high_vol_threshold] = 'high'
            
            return {
                'thresholds': {
                    'low': low_vol_threshold,
                    'high': high_vol_threshold
                },
                'regimes': regimes
            }
        except Exception as e:
            print(f"分析波动率区间错误: {str(e)}")
            return {}
            
    def get_results(self) -> dict:
        """获取分析结果"""
        return self.results
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from .base_analyzer import BaseAnalyzer

class CorrelationAnalyzer(BaseAnalyzer):
    """相关性分析器"""
    
    def __init__(self, data: pd.DataFrame, other_assets: Dict[str, pd.DataFrame] = None):
        """
        初始化相关性分析器
        
        Args:
            data: 主要资产的数据
            other_assets: 其他资产的数据字典，格式为 {asset_name: asset_data}
        """
        super().__init__(data)
        self.other_assets = other_assets or {}
        self.results = {
            'price_correlation': {},
            'return_correlation': {},
            'rolling_correlation': {},
            'tail_correlation': {}
        }
        
    def analyze(self, windows: List[int] = [30, 60, 90], quantile: float = 0.1):
        """
        执行相关性分析
        
        Args:
            windows: 滚动窗口大小列表
            quantile: 尾部相关性的分位数
        """
        self.validate_data()
        self._calculate_price_correlation()
        self._calculate_return_correlation()
        self._calculate_rolling_correlation(windows)
        self._calculate_tail_correlation(quantile)
        
    def _calculate_price_correlation(self):
        """计算价格相关性"""
        price_data = pd.DataFrame({'main': self.data['close']})
        
        for name, asset_data in self.other_assets.items():
            price_data[name] = asset_data['close']
            
        self.results['price_correlation'] = price_data.corr()
        
    def _calculate_return_correlation(self):
        """计算收益率相关性"""
        returns = pd.DataFrame({
            'main': self.data['close'].pct_change()
        })
        
        for name, asset_data in self.other_assets.items():
            returns[name] = asset_data['close'].pct_change()
            
        self.results['return_correlation'] = returns.corr()
        
    def _calculate_rolling_correlation(self, windows: List[int]):
        """计算滚动相关性"""
        main_returns = self.data['close'].pct_change()
        
        for name, asset_data in self.other_assets.items():
            other_returns = asset_data['close'].pct_change()
            
            for window in windows:
                rolling_corr = main_returns.rolling(window).corr(other_returns)
                self.results['rolling_correlation'][f'{name}_{window}d'] = rolling_corr
                
    def _calculate_tail_correlation(self, quantile: float):
        """计算尾部相关性"""
        main_returns = self.data['close'].pct_change()
        
        for name, asset_data in self.other_assets.items():
            other_returns = asset_data['close'].pct_change()
            
            # 计算下尾相关性
            lower_mask = (main_returns <= main_returns.quantile(quantile)) & \
                        (other_returns <= other_returns.quantile(quantile))
            lower_tail_corr = np.corrcoef(
                main_returns[lower_mask],
                other_returns[lower_mask]
            )[0,1]
            
            # 计算上尾相关性
            upper_mask = (main_returns >= main_returns.quantile(1-quantile)) & \
                        (other_returns >= other_returns.quantile(1-quantile))
            upper_tail_corr = np.corrcoef(
                main_returns[upper_mask],
                other_returns[upper_mask]
            )[0,1]
            
            self.results['tail_correlation'][name] = {
                'lower_tail': lower_tail_corr,
                'upper_tail': upper_tail_corr
            }
            
    def get_results(self) -> Dict:
        """获取分析结果"""
        return self.results
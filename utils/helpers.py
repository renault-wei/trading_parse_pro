import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta

class DataHelper:
    """数据处理辅助类"""
    
    @staticmethod
    def clean_price_data(df: pd.DataFrame, price_col: str = 'price') -> pd.DataFrame:
        """
        清理价格数据
        - 处理异常值
        - 填充缺失值
        - 标准化处理
        """
        cleaned = df.copy()
        
        # 计算3倍标准差范围
        mean = cleaned[price_col].mean()
        std = cleaned[price_col].std()
        upper_bound = mean + 3 * std
        lower_bound = mean - 3 * std
        
        # 处理异常值
        cleaned.loc[cleaned[price_col] > upper_bound, price_col] = upper_bound
        cleaned.loc[cleaned[price_col] < lower_bound, price_col] = lower_bound
        
        # 填充缺失值
        cleaned[price_col] = cleaned[price_col].fillna(method='ffill')
        
        # 添加对数价格
        cleaned['log_price'] = np.log(cleaned[price_col])
        
        return cleaned
        
    @staticmethod
    def calculate_returns(prices: pd.Series, method: str = 'log') -> pd.Series:
        """计算收益率，默认使用对数收益率"""
        if method == 'simple':
            returns = prices.pct_change()
        elif method == 'log':
            returns = np.log(prices / prices.shift(1))
        else:
            raise ValueError("method must be 'simple' or 'log'")
            
        return returns
        
    @staticmethod
    def calculate_volatility(returns: pd.Series, window: int = 24) -> pd.Series:
        """计算波动率，使用对数收益率"""
        return returns.rolling(window=window).std() * np.sqrt(252)  # 年化
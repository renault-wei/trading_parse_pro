from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import logging

class BaseAnalyzer(ABC):
    """分析器基类"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.results = {}
        self.logger = logging.getLogger('trading_system')
        
    @abstractmethod
    def analyze(self):
        """执行分析"""
        pass
    
    @abstractmethod
    def get_results(self):
        """获取分析结果"""
        pass
    
    def validate_data(self):
        """数据验证"""
        if self.data.empty:
            raise ValueError("数据为空")
            
        required_columns = ['price', 'hour', 'day', 'month']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"数据缺少必需的列: {missing_columns}")
 
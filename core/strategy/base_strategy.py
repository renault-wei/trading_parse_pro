from abc import ABC, abstractmethod
from datetime import datetime
import pandas as pd
from typing import Dict, List

class BaseStrategy(ABC):
    """策略基类"""
    
    def __init__(self, params: dict = None):
        """
        初始化策略
        
        Args:
            params: 策略参数字典
        """
        self.params = params or {}
        
    @abstractmethod
    def generate_signals(
        self,
        timestamp: datetime,
        data: pd.DataFrame,
        positions: Dict
    ) -> List[dict]:
        """
        生成交易信号
        
        Args:
            timestamp: 当前时间戳
            data: 历史数据，包含到当前时间戳的所有数据
            positions: 当前持仓信息
            
        Returns:
            List[dict]: 交易信号列表，每个信号是一个字典，包含 direction, quantity 等信息
        """
        pass 
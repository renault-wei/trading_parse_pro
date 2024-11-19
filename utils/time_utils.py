from datetime import datetime, timedelta
from typing import List, Tuple, Optional
import pandas as pd

class TimeHelper:
    """时间处理辅助类"""
    
    @staticmethod
    def get_trading_sessions(
        date: datetime,
        peak_hours: List[int] = None
    ) -> List[Tuple[datetime, datetime]]:
        """
        获取交易时段
        
        Args:
            date: 日期
            peak_hours: 高峰时段小时列表
            
        Returns:
            时段列表，每个元素为(开始时间, 结束时间)
        """
        if peak_hours is None:
            peak_hours = [8,9,10,11,12,17,18,19,20,21]
            
        sessions = []
        for hour in peak_hours:
            start = datetime.combine(date.date(), datetime.min.time().replace(hour=hour))
            end = start + timedelta(hours=1)
            sessions.append((start, end))
            
        return sessions
        
    @staticmethod
    def is_trading_hour(dt: datetime, peak_hours: List[int] = None) -> bool:
        """
        判断是否为交易时段
        
        Args:
            dt: 待判断的时间
            peak_hours: 高峰时段小时列表
            
        Returns:
            是否为交易时段
        """
        if peak_hours is None:
            peak_hours = [8,9,10,11,12,17,18,19,20,21]
            
        return dt.hour in peak_hours
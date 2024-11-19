"""时间模式分析模块"""
import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime, time

class TimePatternAnalyzer:
    """时间模式分析器"""
    
    def __init__(self, logger):
        self.logger = logger
        # 定义时段阈值
        self.time_periods = {
            'morning_peak': {'start': time(6), 'end': time(8)},    # 早高峰
            'morning_valley': {'start': time(10), 'end': time(12)}, # 早低谷
            'afternoon_valley': {'start': time(12), 'end': time(14)}, # 午低谷
            'evening_peak': {'start': time(16), 'end': time(18)},   # 晚高峰
            'night_valley': {'start': time(23), 'end': time(5)}     # 夜低谷
        }
        
    def analyze_time_patterns(self, data: pd.DataFrame) -> Dict:
        """
        分析时间模式
        
        Args:
            data: 包含时间索引的DataFrame
            
        Returns:
            Dict: 时间模式分析结果
        """
        try:
            # 添加时间特征
            features = self._extract_time_features(data)
            
            # 分析工作日模式
            workday_patterns = self._analyze_workday_patterns(features)
            
            # 分析时段模式
            period_patterns = self._analyze_period_patterns(features)
            
            # 分析季节性模式
            seasonal_patterns = self._analyze_seasonal_patterns(features)
            
            return {
                'workday_patterns': workday_patterns,
                'period_patterns': period_patterns,
                'seasonal_patterns': seasonal_patterns,
                'features': features
            }
            
        except Exception as e:
            self.logger.error(f"分析时间模式时出错: {str(e)}")
            raise
            
    def _extract_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """提取时间特征"""
        features = data.copy()
        
        # 基础时间特征
        features['hour'] = features.index.hour
        features['weekday'] = features.index.weekday
        features['month'] = features.index.month
        features['season'] = (features.index.month - 1) // 3 + 1
        
        # 工作日标记
        features['is_workday'] = (features['weekday'] < 5).astype(int)
        
        # 时段标记
        for period, times in self.time_periods.items():
            features[f'is_{period}'] = self._is_in_period(
                features.index.time, times['start'], times['end']
            )
            
        return features
        
    def _is_in_period(self, times, start: time, end: time) -> pd.Series:
        """判断时间是否在指定时段内"""
        if start.hour < end.hour:
            return ((times >= start) & (times <= end)).astype(int)
        else:  # 处理跨日时段
            return ((times >= start) | (times <= end)).astype(int)
            
    def _analyze_workday_patterns(self, data: pd.DataFrame) -> Dict:
        """分析工作日模式"""
        # 工作日vs非工作日
        workday_stats = data.groupby('is_workday').agg({
            'price': ['mean', 'std', 'min', 'max'],
            'volume': ['mean', 'sum']
        })
        
        # 分星期统计
        weekday_stats = data.groupby('weekday').agg({
            'price': ['mean', 'std', 'min', 'max'],
            'volume': ['mean', 'sum']
        })
        
        return {
            'workday_stats': workday_stats,
            'weekday_stats': weekday_stats
        }
        
    def _analyze_period_patterns(self, data: pd.DataFrame) -> Dict:
        """分析时段模式"""
        period_stats = {}
        
        # 分析每个时段的特征
        for period in self.time_periods.keys():
            period_data = data[data[f'is_{period}'] == 1]
            
            period_stats[period] = {
                'price_stats': period_data['price'].agg(['mean', 'std', 'min', 'max']),
                'volume_stats': period_data['volume'].agg(['mean', 'sum']),
                'count': len(period_data)
            }
            
        return period_stats
        
    def _analyze_seasonal_patterns(self, data: pd.DataFrame) -> Dict:
        """分析季节性模式"""
        # 月度统计
        monthly_stats = data.groupby('month').agg({
            'price': ['mean', 'std', 'min', 'max'],
            'volume': ['mean', 'sum']
        })
        
        # 季节统计
        seasonal_stats = data.groupby('season').agg({
            'price': ['mean', 'std', 'min', 'max'],
            'volume': ['mean', 'sum']
        })
        
        # 分析工作日和时段在不同季节的表现
        seasonal_workday_stats = data.groupby(['season', 'is_workday']).agg({
            'price': 'mean',
            'volume': 'mean'
        })
        
        return {
            'monthly_stats': monthly_stats,
            'seasonal_stats': seasonal_stats,
            'seasonal_workday_stats': seasonal_workday_stats
        } 
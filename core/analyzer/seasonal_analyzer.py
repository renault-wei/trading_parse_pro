import pandas as pd
import numpy as np
from datetime import datetime

class SeasonalAnalyzer:
    """季节性分析器"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.results = {}
        
    def analyze(self):
        """执行季节性分析"""
        self.results = {
            'daily_patterns': self._analyze_daily_patterns(),
            'weekly_patterns': self._analyze_weekly_patterns(),
            'monthly_patterns': self._analyze_monthly_patterns(),
            'seasonal_patterns': self._analyze_seasonal_patterns()
        }
        
    def _analyze_daily_patterns(self) -> dict:
        """分析日内模式"""
        try:
            # 计算每小时的平均收益率
            hourly_returns = self.data.groupby('hour')['returns'].mean()
            
            # 计算每小时的波动率
            hourly_volatility = self.data.groupby('hour')['returns'].std()
            
            # 计算交易量分布（如果有交易量数据）
            hourly_volume = pd.Series(dtype=float)
            if 'volume' in self.data.columns:
                hourly_volume = self.data.groupby('hour')['volume'].mean()
                
            # 计算高低峰时段的特征
            peak_hours_returns = self.data[self.data['is_morning_peak'] | self.data['is_evening_peak']]['returns'].mean()
            valley_hours_returns = self.data[self.data['is_valley']]['returns'].mean()
            
            return {
                'hourly_returns': hourly_returns,
                'hourly_volatility': hourly_volatility,
                'hourly_volume': hourly_volume,
                'peak_hours_return': peak_hours_returns,
                'valley_hours_return': valley_hours_returns
            }
        except Exception as e:
            print(f"日内模式分析错误: {str(e)}")
            return {}
            
    def _analyze_weekly_patterns(self) -> dict:
        """分析周度模式"""
        try:
            # 计算每周各天的平均收益率
            daily_returns = self.data.groupby('weekday')['returns'].mean()
            
            # 计算每周各天的波动率
            daily_volatility = self.data.groupby('weekday')['returns'].std()
            
            # 计算工作日和周末的差异
            weekday_returns = self.data[self.data['weekday'].isin([0,1,2,3,4])]['returns'].mean()
            weekend_returns = self.data[self.data['weekday'].isin([5,6])]['returns'].mean()
            
            return {
                'daily_returns': daily_returns,
                'daily_volatility': daily_volatility,
                'weekday_return': weekday_returns,
                'weekend_return': weekend_returns
            }
        except Exception as e:
            print(f"周度模式分析错误: {str(e)}")
            return {}
            
    def _analyze_monthly_patterns(self) -> dict:
        """分析月度模式"""
        try:
            # 计算每月的平均收益率
            monthly_returns = self.data.groupby('month')['returns'].mean()
            
            # 计算每月的波动率
            monthly_volatility = self.data.groupby('month')['returns'].std()
            
            # 计算月初、月中、月末的模式
            day_of_month = self.data.index.day
            month_start_returns = self.data[day_of_month <= 5]['returns'].mean()
            month_end_returns = self.data[day_of_month >= 25]['returns'].mean()
            
            return {
                'monthly_returns': monthly_returns,
                'monthly_volatility': monthly_volatility,
                'month_start_return': month_start_returns,
                'month_end_return': month_end_returns
            }
        except Exception as e:
            print(f"月度模式分析错误: {str(e)}")
            return {}
            
    def _analyze_seasonal_patterns(self) -> dict:
        """分析季节性模式"""
        try:
            # 定义季节
            spring_months = [3, 4, 5]
            summer_months = [6, 7, 8]
            autumn_months = [9, 10, 11]
            winter_months = [12, 1, 2]
            
            # 计算各季节的平均收益率
            spring_returns = self.data[self.data['month'].isin(spring_months)]['returns'].mean()
            summer_returns = self.data[self.data['month'].isin(summer_months)]['returns'].mean()
            autumn_returns = self.data[self.data['month'].isin(autumn_months)]['returns'].mean()
            winter_returns = self.data[self.data['month'].isin(winter_months)]['returns'].mean()
            
            # 计算各季节的波动率
            spring_volatility = self.data[self.data['month'].isin(spring_months)]['returns'].std()
            summer_volatility = self.data[self.data['month'].isin(summer_months)]['returns'].std()
            autumn_volatility = self.data[self.data['month'].isin(autumn_months)]['returns'].std()
            winter_volatility = self.data[self.data['month'].isin(winter_months)]['returns'].std()
            
            return {
                'seasonal_returns': {
                    'spring': spring_returns,
                    'summer': summer_returns,
                    'autumn': autumn_returns,
                    'winter': winter_returns
                },
                'seasonal_volatility': {
                    'spring': spring_volatility,
                    'summer': summer_volatility,
                    'autumn': autumn_volatility,
                    'winter': winter_volatility
                }
            }
        except Exception as e:
            print(f"季节性分析错误: {str(e)}")
            return {}
            
    def get_results(self) -> dict:
        """获取分析结果"""
        return self.results
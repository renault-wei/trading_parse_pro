from typing import Dict
import pandas as pd
from .base_analyzer import BaseAnalyzer

class PeriodPatternAnalyzer(BaseAnalyzer):
    """时段模式分析器"""
    
    def __init__(self, data: pd.DataFrame):
        super().__init__(data)
        self.period_patterns = {
            'spring': {'months': [3,4,5]},    # 春季
            'summer': {'months': [6,7,8]},    # 夏季
            'autumn': {'months': [9,10,11]},  # 秋季
            'winter': {'months': [12,1,2]}    # 冬季
        }
        self.hour_patterns = {
            'deep_valley': {'hours': range(0,6)},      # 深谷时段
            'valley': {'hours': [10,14]},              # 谷时段
            'flat': {'hours': range(6,10)},            # 平段时段
            'peak': {'hours': range(19,21)},           # 峰时段
            'deep_peak': {'hours': range(17,19)}       # 尖峰时段
        }
        self.results = {}
        
    def analyze(self):
        """执行分析"""
        try:
            self.logger.info("开始时段模式分析...")
            
            # 检查原始数据
            self.logger.info(f"""
            原始数据信息:
            - 总记录数: {len(self.data)}
            - 时间范围: {self.data.index.min()} 到 {self.data.index.max()}
            - 数据列: {self.data.columns.tolist()}
            - 价格范围: [{self.data['price'].min():.2f}, {self.data['price'].max():.2f}]
            """)
            
            # 添加季节标记
            self.data['season'] = self.data.index.month.map(self._get_season)
            season_counts = self.data['season'].value_counts()
            self.logger.info(f"季节分布:\n{season_counts}")
            
            # 分析每个季节的时段特征
            seasonal_patterns = {}
            for season in ['spring', 'summer', 'autumn', 'winter']:
                self.logger.info(f"\n开始分析 {season} 季节...")
                season_data = self.data[self.data['season'] == season]
                
                if not season_data.empty:
                    self.logger.info(f"- {season}季数据点数: {len(season_data)}")
                    
                    # 计算每个时段的特征
                    hourly_stats = season_data.groupby('hour')['price'].agg([
                        'mean', 'std', 'min', 'max',
                        lambda x: x.quantile(0.25),
                        lambda x: x.quantile(0.75)
                    ]).round(2)
                    
                    self.logger.info(f"- 小时统计:\n{hourly_stats}")
                    
                    # 识别时段类型
                    hourly_stats['period_type'] = hourly_stats.index.map(self._get_period_type)
                    
                    # 计算时段汇总统计
                    period_summary = self._calculate_period_summary(season_data)
                    self.logger.info(f"- 时段汇总:\n{period_summary}")
                    
                    seasonal_patterns[season] = {
                        'hourly_stats': hourly_stats,
                        'period_summary': period_summary
                    }
                    
                    self.logger.info(f"""
                    {season}季分析结果:
                    - 数据点数: {len(season_data)}
                    - 平均价格: {season_data['price'].mean():.2f}
                    - 价格范围: [{season_data['price'].min():.2f}, {season_data['price'].max():.2f}]
                    - 时段统计: {period_summary}
                    """)
                else:
                    self.logger.warning(f"{season}季无数据")
            
            # 分析时段转换特征
            self.logger.info("\n开始分析时段转换特征...")
            transition_patterns = self._analyze_period_transitions()
            
            self.results = {
                'seasonal_patterns': seasonal_patterns,
                'transition_patterns': transition_patterns
            }
            
            self.logger.info("时段模式分析完成")
            
        except Exception as e:
            self.logger.error(f"时段模式分析出错: {str(e)}")
            self.logger.info(f"数据示例:\n{self.data.head()}")
            self.logger.info(f"数据类型:\n{self.data.dtypes}")
            self.logger.info(f"数据索引类型: {type(self.data.index)}")
            raise
            
    def _get_season(self, month: int) -> str:
        """获取季节"""
        for season, info in self.period_patterns.items():
            if month in info['months']:
                return season
        return 'unknown'
        
    def _get_period_type(self, hour: int) -> str:
        """获取时段类型"""
        for period, info in self.hour_patterns.items():
            if hour in info['hours']:
                return period
        return 'normal'
        
    def _calculate_period_summary(self, data: pd.DataFrame) -> Dict:
        """计算时段汇总统计"""
        summary = {}
        for period, info in self.hour_patterns.items():
            period_data = data[data['hour'].isin(info['hours'])]
            if not period_data.empty:
                summary[period] = {
                    'mean_price': period_data['price'].mean(),
                    'price_volatility': period_data['price'].std() / period_data['price'].mean(),
                    'data_points': len(period_data)
                }
        return summary
        
    def _analyze_period_transitions(self) -> Dict:
        """分析时段转换特征"""
        transitions = {}
        for season in ['spring', 'summer', 'autumn', 'winter']:
            season_data = self.data[self.data['season'] == season]
            if not season_data.empty:
                # 计算时段转换时的价格变化
                transitions[season] = {
                    'valley_to_peak': self._calculate_transition_change(season_data, 'valley', 'peak'),
                    'peak_to_valley': self._calculate_transition_change(season_data, 'peak', 'valley')
                }
        return transitions
        
    def _calculate_transition_change(self, data: pd.DataFrame, from_period: str, to_period: str) -> float:
        """计算时段转换时的价格变化"""
        from_hours = self.hour_patterns[from_period]['hours']
        to_hours = self.hour_patterns[to_period]['hours']
        
        from_prices = data[data['hour'].isin(from_hours)]['price']
        to_prices = data[data['hour'].isin(to_hours)]['price']
        
        if not from_prices.empty and not to_prices.empty:
            return (to_prices.mean() - from_prices.mean()) / from_prices.mean()
        return 0.0
        
    def get_results(self) -> Dict:
        """获取分析结果"""
        return self.results
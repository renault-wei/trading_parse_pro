from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from scipy import stats
from .base_analyzer import BaseAnalyzer

class FactorAnalyzer(BaseAnalyzer):
    """因子分析器"""
    
    def __init__(self, data: pd.DataFrame, windows: List[int] = [24, 48, 72, 168]):
        super().__init__(data)
        self.windows = windows
        self.results = {
            'price_factors': {},
            'time_factors': {},
            'technical_factors': {},
            'transition_factors': {},  # 新增时段转换因子
            'factor_correlation': None,
            'factor_ic': {},
            'factor_turnover': {}
        }
        
    def analyze(self):
        """执行因子分析"""
        self.validate_data()
        self._calculate_price_factors()
        self._calculate_time_factors()
        self._calculate_technical_factors()
        self._calculate_transition_factors()  # 新增时段转换因子计算
        self._analyze_factor_correlation()
        self._calculate_factor_ic()
        self._calculate_factor_turnover()
        
    def _calculate_price_factors(self):
        """计算价格相关因子"""
        price = self.data['price']
        self.logger.info("开始计算价格因子...")
        
        # 保留布林带上轨因子，增加其权重
        for window in self.windows:
            ma = price.rolling(window=window).mean()
            std = price.rolling(window=window).std()
            bb_upper = (ma + 2 * std - price) / price
            
            # 增加布林带上轨的变种因子
            bb_upper_std = (bb_upper - bb_upper.rolling(window=24).mean()) / bb_upper.rolling(window=24).std()
            bb_upper_momentum = bb_upper.pct_change(24)
            
            self.results['price_factors'][f'bb_upper_{window}h'] = bb_upper
            self.results['price_factors'][f'bb_upper_std_{window}h'] = bb_upper_std
            self.results['price_factors'][f'bb_upper_momentum_{window}h'] = bb_upper_momentum
            
            self.logger.info(f"{window}小时布林带上轨: 均值={bb_upper.mean():.4f}, 标准差={bb_upper.std():.4f}")
            
        # 减少动量因子，只保留关键窗口
        key_windows = [24, 168]  # 1天和1周
        for window in key_windows:
            momentum = price.pct_change(window)
            self.results['price_factors'][f'momentum_{window}h'] = momentum
            self.logger.info(f"{window}小时动量: 均值={momentum.mean():.4f}, 标准差={momentum.std():.4f}")
            
    def _calculate_time_factors(self):
        """计算时间相关因子"""
        self.logger.info("开始计算时间因子...")
        
        # 1. 日内时段因子
        self.results['time_factors'] = {
            'is_morning_peak': self.data['is_morning_peak'],
            'is_evening_peak': self.data['is_evening_peak'],
            'is_valley': self.data['is_valley'],
            'is_flat': self.data['is_flat']
        }
        
        # 2. 日内相对时间因子
        hour_dummies = pd.get_dummies(self.data['hour'], prefix='hour')
        for col in hour_dummies.columns:
            self.results['time_factors'][col] = hour_dummies[col]
        
        # 3. 工作日/非工作日因子
        if 'data_type' in self.data.columns:
            self.results['time_factors']['is_workday'] = (self.data['data_type'] == 0)
        
        # 4. 时段价格偏离因子
        for period in ['morning_peak', 'evening_peak', 'valley', 'flat']:
            mask = self.data[f'is_{period}']
            if mask.any():
                period_mean = self.data.loc[mask, 'price'].mean()
                self.results['time_factors'][f'{period}_deviation'] = (
                    self.data['price'] - period_mean
                ) / period_mean
        
        # 5. 时段间价格变化因子
        daily_data = self.data.groupby(self.data.index.date)
        
        # 计算早高峰到晚高峰的价格变化
        morning_peak_price = daily_data.apply(
            lambda x: x[x['is_morning_peak']]['price'].mean()
        )
        evening_peak_price = daily_data.apply(
            lambda x: x[x['is_evening_peak']]['price'].mean()
        )
        
        peak_change = pd.Series(index=self.data.index)
        for date in morning_peak_price.index:
            if date in evening_peak_price.index:
                change = (evening_peak_price[date] - morning_peak_price[date]) / morning_peak_price[date]
                peak_change[self.data.index.date == date] = change
                
        self.results['time_factors']['peak_price_change'] = peak_change
        
        # 6. 时段持续性因子
        for period in ['morning_peak', 'evening_peak', 'valley', 'flat']:
            mask = self.data[f'is_{period}']
            if mask.any():
                # 计算当前时段价格与前一个相同时段价格的变化
                period_price = self.data.loc[mask, 'price']
                self.results['time_factors'][f'{period}_persistence'] = period_price.pct_change(24)
        
        # 7. 时段波动率因子
        window = 24 * 7  # 一周
        for period in ['morning_peak', 'evening_peak', 'valley', 'flat']:
            mask = self.data[f'is_{period}']
            if mask.any():
                period_returns = self.data.loc[mask, 'price'].pct_change()
                self.results['time_factors'][f'{period}_volatility'] = period_returns.rolling(window).std()
        
        self.logger.info(f"""
        时间因子计算完成:
        - 基础时段因子: {len([k for k in self.results['time_factors'].keys() if k.startswith('is_')])}个
        - 小时虚拟变量: {len([k for k in self.results['time_factors'].keys() if k.startswith('hour_')])}个
        - 时段价格偏离因子: {len([k for k in self.results['time_factors'].keys() if k.endswith('_deviation')])}个
        - 时段持续性因子: {len([k for k in self.results['time_factors'].keys() if k.endswith('_persistence')])}个
        - 时段波动率因子: {len([k for k in self.results['time_factors'].keys() if k.endswith('_volatility')])}个
        """)
        
    def _calculate_technical_factors(self):
        """计算技术因子"""
        self.logger.info("开始计算技术因子...")
        price = self.data['price']
        
        # 1. RSI因子优化
        for window in [24, 72, 168]:  # 减少窗口数量，专注于关键周期
            delta = price.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=window).mean()
            avg_loss = loss.rolling(window=window).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            # 添加RSI动量
            rsi_momentum = rsi - rsi.rolling(window=24).mean()
            
            self.results['technical_factors'][f'rsi_{window}h'] = rsi
            self.results['technical_factors'][f'rsi_momentum_{window}h'] = rsi_momentum
            
        # 2. 价格波动特征
        for window in [24, 72, 168]:
            # 波动率归一化
            rolling_std = price.rolling(window).std()
            rolling_mean = price.rolling(window).mean()
            normalized_vol = rolling_std / rolling_mean
            
            # 波动率变化
            vol_change = normalized_vol.pct_change(24)
            
            self.results['technical_factors'][f'normalized_vol_{window}h'] = normalized_vol
            self.results['technical_factors'][f'vol_change_{window}h'] = vol_change
            
        # 3. 价格区间特征
        for window in [24, 72, 168]:
            roll_max = price.rolling(window).max()
            roll_min = price.rolling(window).min()
            
            # 价格位置
            price_position = (price - roll_min) / (roll_max - roll_min)
            # 区间宽度
            range_width = (roll_max - roll_min) / price
            
            self.results['technical_factors'][f'price_position_{window}h'] = price_position
            self.results['technical_factors'][f'range_width_{window}h'] = range_width
            
        self.logger.info("技术因子计算完成")
        
    def _calculate_transition_factors(self):
        """计算时段转换因子"""
        self.logger.info("开始计算时段转换因子...")
        
        # 定义时段
        morning_peak = range(8, 13)  # 8-12点
        evening_peak = range(17, 22)  # 17-21点
        valley = list(range(0, 6)) + list(range(22, 24))  # 0-5点和22-23点
        flat = range(13, 17)  # 13-16点
        
        # 计算时段内价格特征
        def calculate_period_features(data: pd.DataFrame, period_hours: range or list, name: str) -> pd.Series:
            period_data = data[data['hour'].isin(period_hours)]
            if period_data.empty:
                return pd.Series(index=data.index)
                
            # 计算时段均价
            period_mean = period_data.groupby(period_data.index.date)['price'].mean()
            # 扩展到所有时间点
            period_mean_expanded = pd.Series(index=data.index)
            for date in period_mean.index:
                period_mean_expanded[data.index.date == date] = period_mean[date]
                
            return period_mean_expanded
            
        # 计算各时段均价
        morning_peak_price = calculate_period_features(self.data, morning_peak, 'morning_peak')
        evening_peak_price = calculate_period_features(self.data, evening_peak, 'evening_peak')
        valley_price = calculate_period_features(self.data, valley, 'valley')
        flat_price = calculate_period_features(self.data, flat, 'flat')
        
        # 计算时段转换因子
        # 1. 峰谷价差
        self.results['transition_factors']['peak_valley_spread'] = (
            (morning_peak_price + evening_peak_price) / 2 - valley_price
        ) / valley_price
        
        # 2. 早晚峰价差
        self.results['transition_factors']['peak_spread'] = (
            evening_peak_price - morning_peak_price
        ) / morning_peak_price
        
        # 3. 平段相对价差
        self.results['transition_factors']['flat_relative'] = (
            flat_price - (morning_peak_price + evening_peak_price) / 2
        ) / ((morning_peak_price + evening_peak_price) / 2)
        
        # 4. 时段动量
        for period_price, period_name in [
            (morning_peak_price, 'morning_peak'),
            (evening_peak_price, 'evening_peak'),
            (valley_price, 'valley'),
            (flat_price, 'flat')
        ]:
            # 计算时段间的价格动量
            self.results['transition_factors'][f'{period_name}_momentum'] = period_price.pct_change()
            
        # 5. 时段波动率
        for period_price, period_name in [
            (morning_peak_price, 'morning_peak'),
            (evening_peak_price, 'evening_peak'),
            (valley_price, 'valley'),
            (flat_price, 'flat')
        ]:
            # 计算时段的波动率
            self.results['transition_factors'][f'{period_name}_volatility'] = (
                period_price.rolling(window=24).std() / period_price.rolling(window=24).mean()
            )
            
        self.logger.info("时段转换因子计算完成")
        
    def _analyze_factor_correlation(self):
        """分析因子间相关性"""
        # 收集所有因子数据
        factor_data = pd.DataFrame()
        
        # 优先添加布林带上轨和时段转换因子
        priority_factors = ['bb_upper', 'transition_factors']
        
        for category, factors in self.results.items():
            if isinstance(factors, dict) and category not in ['factor_correlation', 'factor_ic', 'factor_turnover']:
                is_priority = any(pf in category for pf in priority_factors)
                
                for name, values in factors.items():
                    if isinstance(values, pd.Series):
                        # 为优先因子添加更高权重
                        if is_priority:
                            factor_data[f'{category}_{name}_weighted'] = values * 1.5
                        factor_data[f'{category}_{name}'] = values
                        
        self.results['factor_correlation'] = factor_data.corr()
        
    def _calculate_factor_ic(self):
        """计算因子IC值"""
        self.logger.info("开始计算因子IC值")
        future_return = self.data['price'].pct_change().shift(-1)
        
        for category, factors in self.results.items():
            if isinstance(factors, dict) and category not in ['factor_correlation', 'factor_ic', 'factor_turnover']:
                category_ic = {}
                for name, values in factors.items():
                    if isinstance(values, pd.Series):
                        rolling_ic = values.rolling(window=24).corr(future_return)
                        ic_mean = rolling_ic.mean()
                        self.logger.info(f"因子 {name} 的IC值: {ic_mean:.4f}")
                        category_ic[name] = ic_mean
                self.results['factor_ic'][category] = category_ic
                
    def _calculate_factor_turnover(self):
        """计算因子换手率"""
        self.logger.info("开始计算因子换手率")
        
        for category, factors in self.results.items():
            if isinstance(factors, dict) and category not in ['factor_correlation', 'factor_ic', 'factor_turnover']:
                category_turnover = {}
                for name, values in factors.items():
                    if isinstance(values, pd.Series):
                        try:
                            # 处理零值和NaN
                            mean_abs = abs(values).replace(0, np.nan).mean()
                            if pd.isna(mean_abs) or mean_abs == 0:
                                turnover = 0
                            else:
                                turnover = abs(values.diff()).mean() / mean_abs
                                
                            self.logger.info(f"因子 {name} 的换手率: {turnover:.4f}")
                            category_turnover[name] = turnover
                        except Exception as e:
                            self.logger.warning(f"计算因子 {name} 换手率时出错: {str(e)}")
                            category_turnover[name] = 0
                            
                self.results['factor_turnover'][category] = category_turnover
                
    def _analyze_factor_effectiveness(self):
        """分析因子有效性"""
        self.logger.info("开始分析因子有效性...")
        
        # 计算未来收益
        future_return = self.data['price'].pct_change().shift(-1)
        
        factor_stats = {}
        for category, factors in self.results.items():
            if isinstance(factors, dict) and category not in ['factor_correlation', 'factor_ic', 'factor_turnover']:
                category_stats = {}
                for name, values in factors.items():
                    if isinstance(values, pd.Series):
                        # 计算IC值
                        ic = values.corr(future_return)
                        # 计算IC_IR
                        rolling_ic = values.rolling(24).corr(future_return)
                        ic_ir = rolling_ic.mean() / rolling_ic.std()
                        # 计算因子自相关性
                        auto_corr = values.autocorr()
                        
                        category_stats[name] = {
                            'ic': ic,
                            'ic_ir': ic_ir,
                            'auto_corr': auto_corr
                        }
                        
                factor_stats[category] = category_stats
                
        self.results['factor_effectiveness'] = factor_stats
        
        # 输出分析结果
        for category, stats in factor_stats.items():
            self.logger.info(f"\n{category} 因子有效性分析:")
            for factor, metrics in stats.items():
                self.logger.info(f"""
                因子: {factor}
                - IC值: {metrics['ic']:.4f}
                - IC_IR: {metrics['ic_ir']:.4f}
                - 自相关性: {metrics['auto_corr']:.4f}
                """)
        
    def get_results(self) -> Dict:
        """获取分析结果"""
        return self.results
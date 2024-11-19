import pandas as pd
import numpy as np
from datetime import datetime
from time import time
from utils.helpers import DataHelper
from utils.logger import Logger

class DataProcessor:
    """数据处理类"""
    
    def __init__(self, config=None, logger=None):
        """
        初始化数据处理器
        
        Args:
            config: 配置信息
            logger: 日志记录器
        """
        self.config = config
        self.logger = logger or Logger().get_logger()
        self.raw_data = None
        self.processed_data = None
        
    def process_data(self, data):
        """处理数据"""
        try:
            self.raw_data = data.copy()
            processed = data.copy()
            

            
            # 确保保留原始的trade_date和trade_hour
            if 'trade_date' not in processed.columns or 'trade_hour' not in processed.columns:
                self.logger.error("缺少必要的trade_date或trade_hour列")
                return None
                
            # 确保 trade_date 是 datetime 类型
            if 'trade_date' in processed.columns:
                processed['trade_date'] = pd.to_datetime(processed['trade_date'], errors='coerce')
                if processed['trade_date'].isnull().any():
                    self.logger.warning("trade_date 列中存在无法转换的值，已被设置为 NaT。")
            
            # 添加工作日标记
            processed['is_workday'] = processed['trade_date'].dt.dayofweek < 5  # 周一到周五为工作日
            
            # 基础特征计算
            processed['price'] = processed['price'].astype(float)
            processed['log_price'] = np.log(processed['price'])
            
            # 计算收益率
            processed['returns'] = processed['price'].pct_change()  # 计算收益率
            
            # 时间特征 (trade_hour已经是0-23范围)
            processed['hour'] = processed['trade_hour']
            processed['day'] = processed['trade_date'].dt.day
            processed['weekday'] = processed['trade_date'].dt.dayofweek
            processed['month'] = processed['trade_date'].dt.month
            
            # 添加 hour_sin 和 hour_cos 特征
            if 'trade_hour' in processed.columns:
                processed['hour_sin'] = np.sin(2 * np.pi * processed['trade_hour'] / 24)
                processed['hour_cos'] = np.cos(2 * np.pi * processed['trade_hour'] / 24)
            
            # 添加 price_ma24 特征
            processed['price_ma24'] = processed['price'].rolling(window=24).mean()  # 计算24小时移动平均
            
            # 添加 price_volatility 特征
            processed['price_volatility'] = processed['price'].rolling(window=24).std()
            
            # 填充缺失值
            processed.fillna(method='ffill', inplace=True)  # 前向填充
            processed.fillna(method='bfill', inplace=True)  # 后向填充
            
  
            
            self.processed_data = processed
            self.logger.info(f"数据处理完成，最终列: {processed.columns.tolist()}")
            
            return processed
            
        except Exception as e:
            self.logger.error(f"数据处理失败: {str(e)}")
            return None
            
    def get_processed_data(self):
        """获取处理后的数据"""
        return self.processed_data
    
    def add_periodic_price_features(self, data):
        """添加周期性价格特征"""
        self.logger.info("开始计算周期性价格特征...")
        
        try:
            # 1. 小时级别的周期性特征
            data['hour_avg_price'] = data.groupby('trade_hour')['price'].transform(
                lambda x: x.expanding().mean()
            )
            
            # 2. 工作日/非工作日 + 小时的周期性特征
            data['hour_workday_avg_price'] = data.groupby(['is_workday', 'trade_hour'])['price'].transform(
                lambda x: x.expanding().mean()
            )
            
            # 3. 最近N天同时段的价格特征
            for days in [7, 14, 30]:  # 一周、两周、一个月
                # 获取过去N天同时段的平均价格
                data[f'hour_avg_price_{days}d'] = data.groupby('trade_hour')['price'].transform(
                    lambda x: x.rolling(window=days, min_periods=1).mean()
                )
                
                # 获取过去N天同时段的价格波动
                data[f'hour_std_price_{days}d'] = data.groupby('trade_hour')['price'].transform(
                    lambda x: x.rolling(window=days, min_periods=1).std()
                )
                
                # 当前价格与历史同时段均价的偏离度
                data[f'price_deviation_{days}d'] = (
                    data['price'] - data[f'hour_avg_price_{days}d']
                ) / data[f'hour_std_price_{days}d'].replace(0, np.nan).fillna(
                    data[f'hour_std_price_{days}d'].mean()
                )
            
            # 4. 添加价格区间特征
            for days in [7, 30]:
                # 同时段最高价
                data[f'hour_max_price_{days}d'] = data.groupby('trade_hour')['price'].transform(
                    lambda x: x.rolling(window=days, min_periods=1).max()
                )
                # 同时段最低价
                data[f'hour_min_price_{days}d'] = data.groupby('trade_hour')['price'].transform(
                    lambda x: x.rolling(window=days, min_periods=1).min()
                )
                # 当前价格在历史区间的位置
                data[f'price_position_{days}d'] = (
                    data['price'] - data[f'hour_min_price_{days}d']
                ) / (
                    data[f'hour_max_price_{days}d'] - data[f'hour_min_price_{days}d']
                ).replace(0, np.nan).fillna(0.5)
            
            self.logger.info("周期性价格特征计算完成")
            return data
            
        except Exception as e:
            self.logger.error(f"计算周期性价格特征时出错: {str(e)}")
            return None

class SupplyDemandAnalyzer:
    def __init__(self):
        self.supply_threshold = {
            'high': 0.7,    # 供应过剩阈值
            'low': 0.3      # 供应不足阈值
        }
        self.demand_threshold = {
            'high': 0.7,    # 需求旺盛阈值
            'low': 0.3      # 需求低迷阈值
        }
    
    def calculate_supply_demand_ratio(self, supply_volume, demand_volume):
        """计算供需比率"""
        if demand_volume == 0:
            return float('inf')
        return supply_volume / demand_volume
    
    def get_market_status(self, supply_index, demand_index):
        """
        判断市场状态
        返回: (supply_status, demand_status, price_trend)
        """
        if supply_index > self.supply_threshold['high']:
            if demand_index > self.demand_threshold['high']:
                return '供需旺盛', '价格上涨趋势'
            elif demand_index < self.demand_threshold['low']:
                return '供应过剩', '价格下跌趋势'
            else:
                return '供应主导', '价格稳定偏弱'
        elif supply_index < self.supply_threshold['low']:
            if demand_index > self.demand_threshold['high']:
                return '需过剩', '价格上涨趋势'
            elif demand_index < self.demand_threshold['low']:
                return '供需低迷', '价格下跌趋势'
            else:
                return '供应不足', '价格稳定偏强'
        else:
            if demand_index > self.demand_threshold['high']:
                return '需求主导', '价格上涨趋势'
            elif demand_index < self.demand_threshold['low']:
                return '需求低迷', '价格下跌趋势'
            else:
                return '供需衡', '价格稳定'

class RegressionAnalyzer:
    """回归分析器"""
    def __init__(self, logger):
        self.logger = logger
        
    def analyze_price_factors(self, data: pd.DataFrame) -> dict:
        """
        分析价格影因素
        
        Args:
            data: 包含价格和各种因素的DataFrame
            
        Returns:
            dict: 回归分析结果
        """
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import StandardScaler
            
            # 准备特征变量
            features = [
                'hour', 'weekday', 'month', 'season',
                'is_morning_peak', 'is_evening_peak', 
                'is_valley', 'is_flat'
            ]
            
            X = data[features].copy()
            y = data['price']
            
            # 标准化特征
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=features)
            
            # 执行回归分析
            model = LinearRegression()
            model.fit(X_scaled, y)
            
            # 计算各个特征的影响程度
            feature_importance = pd.DataFrame({
                'feature': features,
                'coefficient': model.coef_,
                'abs_importance': abs(model.coef_)
            })
            
            # 按重要性排序
            feature_importance = feature_importance.sort_values(
                'abs_importance', ascending=False
            )
            
            # 计算模型评分
            r2_score = model.score(X_scaled, y)
            
            # 计算预测值和残差
            y_pred = model.predict(X_scaled)
            residuals = y - y_pred
            
            return {
                'feature_importance': feature_importance,
                'r2_score': r2_score,
                'residuals': residuals,
                'predictions': y_pred,
                'model': model,
                'scaler': scaler
            }
            
        except Exception as e:
            self.logger.error(f"回归分析出错: {str(e)}")
            raise
            
    def analyze_supply_demand_impact(self, data: pd.DataFrame, 
                                   supply_index: pd.Series, 
                                   demand_index: pd.Series) -> dict:
        """
        分析供需对价格的影响
        
        Args:
            data: 价数据
            supply_index: 供应指数
            demand_index: 需求指数
            
        Returns:
            dict: 分析结果
        """
        try:
            from sklearn.linear_model import LinearRegression
            import statsmodels.api as sm
            
            # 准备数据
            X = pd.DataFrame({
                'supply': supply_index,
                'demand': demand_index
            })
            y = data['price']
            
            # 添加常数项
            X_with_const = sm.add_constant(X)
            
            # 使用statsmodels进行回归分析
            model = sm.OLS(y, X_with_const)
            results = model.fit()
            
            # 计算供需弹性
            supply_elasticity = results.params['supply'] * np.mean(supply_index) / np.mean(y)
            demand_elasticity = results.params['demand'] * np.mean(demand_index) / np.mean(y)
            
            return {
                'summary': results.summary(),
                'params': results.params,
                'r2': results.rsquared,
                'adj_r2': results.rsquared_adj,
                'supply_elasticity': supply_elasticity,
                'demand_elasticity': demand_elasticity,
                'p_values': results.pvalues,
                'confidence_intervals': results.conf_int()
            }
            
        except Exception as e:
            self.logger.error(f"供需影响分析出错: {str(e)}")
            raise
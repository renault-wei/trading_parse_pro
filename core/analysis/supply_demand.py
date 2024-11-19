"""供需分析模块"""
import pandas as pd
import numpy as np
from utils.logger import Logger

class SupplyDemandAnalyzer:
    """供需分析器"""
    
    def __init__(self):
        """初始化供需分析器"""
        self.logger = Logger().get_logger()
        
    def calculate_supply_demand_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算供需指标"""
        try:
            if data is None or len(data) == 0:
                self.logger.error("输入数据为空")
                return None
                
            # 创建数据副本
            df = data.copy()
            
            # 计算价格变化
            df['price_change'] = df['price'].diff()
            
            # 计算供给压力（价格下跌时的压力）
            df['supply_pressure'] = df['price_change'].apply(lambda x: abs(min(x, 0)))
            
            # 计算需求压力（价格上涨时的压力）
            df['demand_pressure'] = df['price_change'].apply(lambda x: max(x, 0))
            
            # 计算移动平均压力
            window = 24  # 24小时窗口
            df['supply_pressure_ma'] = df['supply_pressure'].rolling(window=window, min_periods=1).mean()
            df['demand_pressure_ma'] = df['demand_pressure'].rolling(window=window, min_periods=1).mean()
            
            # 计算供需比率
            df['supply_demand_ratio'] = df['supply_pressure_ma'] / df['demand_pressure_ma'].replace(0, np.nan)
            
            # 填充缺失值
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            self.logger.info("供需指标计算完成")
            return df
            
        except Exception as e:
            self.logger.error(f"计算供需指标时出错: {str(e)}")
            return None
            
    def analyze_price_impact(self, data: pd.DataFrame) -> dict:
        """分析供需对价格的影响"""
        try:
            if data is None or len(data) == 0:
                self.logger.error("输入数据为空")
                return None
                
            # 计算相关性
            correlation = data[['price', 'supply_pressure_ma', 'demand_pressure_ma']].corr()
            
            # 计算预测相关性（使用滞后一期的供需压力）
            lagged_supply = data['supply_pressure_ma'].shift(1)
            lagged_demand = data['demand_pressure_ma'].shift(1)
            price_changes = data['price'].diff()
            
            supply_correlation = price_changes.corr(lagged_supply)
            demand_correlation = price_changes.corr(lagged_demand)
            
            # 计算p值
            from scipy import stats
            _, supply_p = stats.pearsonr(lagged_supply.dropna(), price_changes.dropna())
            _, demand_p = stats.pearsonr(lagged_demand.dropna(), price_changes.dropna())
            
            results = {
                'correlation': correlation.iloc[0, 1:].to_dict(),
                'predictive_correlation': {
                    'supply': supply_correlation,
                    'demand': demand_correlation
                },
                'p_values': {
                    'supply': supply_p,
                    'demand': demand_p
                }
            }
            
            self.logger.info("价格影响分析完成")
            return results
            
        except Exception as e:
            self.logger.error(f"分析价格影响时出错: {str(e)}")
            return None
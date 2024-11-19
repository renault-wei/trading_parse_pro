import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
import logging

logger = logging.getLogger(__name__)

class SupplyDemandAnalyzer:
    def __init__(self):
        self.logger = logger
        
    def calculate_supply_demand_metrics(self, data):
        """计算供需指标"""
        try:
            if 'log_price' not in data.columns:
                self.logger.error("数据中缺少log_price列")
                return None
                
            # 计算价格变化
            data['price_change'] = data['log_price'].diff()
            
            # 计算供需压力
            data['supply_pressure'] = data['price_change'].rolling(window=24).apply(
                lambda x: np.sum(x[x < 0])
            )
            
            data['demand_pressure'] = data['price_change'].rolling(window=24).apply(
                lambda x: np.sum(x[x > 0])
            )
            
            # 计算供需比率
            data['supply_demand_ratio'] = (
                data['supply_pressure'].abs() / 
                (data['demand_pressure'] + data['supply_pressure'].abs())
            )
            
            return data
            
        except Exception as e:
            self.logger.error(f"计算供需指标时发生错误: {str(e)}")
            return None
            
    def analyze_price_impact(self, data):
        """分析供需对价格的影响"""
        try:
            if 'supply_demand_ratio' not in data.columns:
                data = self.calculate_supply_demand_metrics(data)
                
            if data is None:
                return None
                
            # 计算相关性
            correlation = stats.pearsonr(
                data['supply_demand_ratio'].dropna(),
                data['price_change'].dropna()
            )
            
            # 计算预测能力
            future_returns = data['price_change'].shift(-1)
            predictive_corr = stats.pearsonr(
                data['supply_demand_ratio'].dropna(),
                future_returns.dropna()
            )
            
            results = {
                'correlation': correlation[0],
                'correlation_pvalue': correlation[1],
                'predictive_correlation': predictive_corr[0],
                'predictive_pvalue': predictive_corr[1],
                'data': data
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"分析价格影响时发生错误: {str(e)}")
            return None 
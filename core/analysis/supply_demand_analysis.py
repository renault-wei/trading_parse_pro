"""供需分析模块"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from scipy import stats
import plotly.graph_objects as go

class SupplyDemandAnalyzer:
    """供需分析器"""
    
    def __init__(self, logger):
        self.logger = logger
        self.thresholds = {
            'supply': {'high': 0.7, 'low': 0.3},
            'demand': {'high': 0.7, 'low': 0.3}
        }
        
    def analyze_market_equilibrium(self, 
                                 supply: pd.Series, 
                                 demand: pd.Series, 
                                 price: pd.Series) -> Dict:
        """
        分析市场均衡状态
        
        Args:
            supply: 供应量序列
            demand: 需求量序列
            price: 价格序列
            
        Returns:
            Dict: 市场均衡分析结果
        """
        try:
            # 计算供需差额
            surplus = supply - demand
            
            # 计算市场压力指数
            market_pressure = surplus / ((supply + demand) / 2)
            
            # 计算价格弹性
            price_elasticity = self._calculate_price_elasticity(surplus, price)
            
            # 识别市场状态
            market_states = self._identify_market_states(supply, demand)
            
            # 计算均衡价格
            equilibrium_price = self._estimate_equilibrium_price(supply, demand, price)
            
            return {
                'surplus': surplus,
                'market_pressure': market_pressure,
                'price_elasticity': price_elasticity,
                'market_states': market_states,
                'equilibrium_price': equilibrium_price
            }
            
        except Exception as e:
            self.logger.error(f"分析市场均衡时出错: {str(e)}")
            raise
            
    def analyze_seasonal_patterns(self, 
                                supply: pd.Series, 
                                demand: pd.Series, 
                                timestamps: pd.DatetimeIndex) -> Dict:
        """
        分析供需的季节性模式
        
        Args:
            supply: 供应量序列
            demand: 需求量序列
            timestamps: 时间戳索引
            
        Returns:
            Dict: 季节性分析结果
        """
        try:
            # 创建数据框
            df = pd.DataFrame({
                'supply': supply,
                'demand': demand,
                'hour': timestamps.hour,
                'weekday': timestamps.weekday,
                'month': timestamps.month,
                'season': (timestamps.month - 1) // 3 + 1
            })
            
            # 分时段分析
            hourly_patterns = df.groupby('hour').agg({
                'supply': ['mean', 'std'],
                'demand': ['mean', 'std']
            })
            
            # 分星期分析
            weekday_patterns = df.groupby('weekday').agg({
                'supply': ['mean', 'std'],
                'demand': ['mean', 'std']
            })
            
            # 分月份分析
            monthly_patterns = df.groupby('month').agg({
                'supply': ['mean', 'std'],
                'demand': ['mean', 'std']
            })
            
            # 分季节分析
            seasonal_patterns = df.groupby('season').agg({
                'supply': ['mean', 'std'],
                'demand': ['mean', 'std']
            })
            
            return {
                'hourly_patterns': hourly_patterns,
                'weekday_patterns': weekday_patterns,
                'monthly_patterns': monthly_patterns,
                'seasonal_patterns': seasonal_patterns
            }
            
        except Exception as e:
            self.logger.error(f"分析季节性模式时出错: {str(e)}")
            raise
            
    def _calculate_price_elasticity(self, 
                                  quantity_change: pd.Series, 
                                  price: pd.Series) -> float:
        """计算价格弹性"""
        try:
            # 计算数量和价格的百分比变化
            quantity_pct_change = quantity_change.pct_change()
            price_pct_change = price.pct_change()
            
            # 移除无效值
            valid_mask = (quantity_pct_change != 0) & (price_pct_change != 0)
            
            # 计算弹性
            elasticities = quantity_pct_change[valid_mask] / price_pct_change[valid_mask]
            
            # 返回中位数弹性
            return np.median(elasticities)
            
        except Exception as e:
            self.logger.error(f"计算价格弹性时出错: {str(e)}")
            raise
            
    def _identify_market_states(self, 
                              supply: pd.Series, 
                              demand: pd.Series) -> pd.Series:
        """识别市场状态"""
        try:
            # 标准化供需数据
            supply_norm = (supply - supply.mean()) / supply.std()
            demand_norm = (demand - demand.mean()) / demand.std()
            
            # 定义状态判断函数
            def get_state(s, d):
                if s > self.thresholds['supply']['high']:
                    if d > self.thresholds['demand']['high']:
                        return '供需旺盛'
                    elif d < self.thresholds['demand']['low']:
                        return '供应过剩'
                    else:
                        return '供应主导'
                elif s < self.thresholds['supply']['low']:
                    if d > self.thresholds['demand']['high']:
                        return '需求过剩'
                    elif d < self.thresholds['demand']['low']:
                        return '供需低迷'
                    else:
                        return '供应不足'
                else:
                    if d > self.thresholds['demand']['high']:
                        return '需求主导'
                    elif d < self.thresholds['demand']['low']:
                        return '需求低迷'
                    else:
                        return '供需平衡'
                        
            # 应用状态判断
            market_states = pd.Series(
                [get_state(s, d) for s, d in zip(supply_norm, demand_norm)],
                index=supply.index
            )
            
            return market_states
            
        except Exception as e:
            self.logger.error(f"识别市场状态时出错: {str(e)}")
            raise
            
    def _estimate_equilibrium_price(self, 
                                  supply: pd.Series, 
                                  demand: pd.Series, 
                                  price: pd.Series) -> float:
        """估计均衡价格"""
        try:
            # 计算供需差额的绝对值
            abs_surplus = abs(supply - demand)
            
            # 找到供需最接近的点
            equilibrium_idx = abs_surplus.idxmin()
            
            # 返回对应的价格
            return price[equilibrium_idx]
            
        except Exception as e:
            self.logger.error(f"估计均衡价格时出错: {str(e)}")
            raise
            
    def generate_supply_demand_dashboard(self, data: pd.DataFrame) -> None:
        """生成供需分析仪表板"""
        # 生成图表逻辑
        fig = go.Figure()
        # 添加图表数据
        fig.write_html('trading_system/output/analysis/supply_demand_dashboard.html')
from typing import Dict, List, Optional, Union
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.logger import Logger
from .price_charts import PriceChartGenerator
from .regression_charts import RegressionChartGenerator
from .time_charts import TimeChartGenerator
from .feature_charts import FeatureChartGenerator
from .backtest_charts import BacktestChartGenerator
from .supply_demand import SupplyDemandChartGenerator
from .navigation import NavigationGenerator

class ChartGenerator:
    """主图表生成器"""
    
    def __init__(self, theme: str = 'plotly_dark'):
        self.theme = theme
        self.logger = Logger().get_logger()
        self._data = None
        
        # 初始化子生成器
        self.price_charts = PriceChartGenerator(theme)
        self.regression_charts = RegressionChartGenerator(theme)
        self.time_charts = TimeChartGenerator(theme)
        self.feature_charts = FeatureChartGenerator(theme)
        self.backtest_charts = BacktestChartGenerator(theme)
        self.supply_demand_charts = SupplyDemandChartGenerator(theme)
        self.navigation = NavigationGenerator(theme)
        
        # 使用基础生成器的输出目录
        self.output_dirs = self.price_charts.output_dirs
        
    def set_data(self, data: pd.DataFrame):
        """设置要分析的数据"""
        self._data = data
        
    def _generate_price_analysis(self, output_dir: str) -> Dict[str, str]:
        """生成价格分析图表"""
        if self._data is None:
            raise ValueError("数据未设置")
        return self.price_charts.generate_price_trend_analysis(self._data)
        
    def _generate_time_patterns(self, output_dir: str) -> Dict[str, str]:
        """生成时间模式分析图表"""
        if self._data is None:
            raise ValueError("数据未设置")
        return self.time_charts.generate_time_patterns(self._data)
        
    def _generate_feature_analysis(self, output_dir: str) -> Dict[str, str]:
        """生成特征分析图表"""
        if self._data is None:
            raise ValueError("数据未设置")
        return self.feature_charts.generate_feature_analysis(self._data)
        
    def _generate_regression_analysis(self, output_dir: str) -> Dict[str, str]:
        """生成回归分析图表"""
        if self._data is None:
            raise ValueError("数据未设置")
        return self.regression_charts.generate_regression_analysis(self._data)
        
    def _generate_supply_demand_analysis(self, output_dir: str) -> Dict[str, str]:
        """生成供需分析图表"""
        if self._data is None:
            raise ValueError("数据未设置")
        return self.supply_demand_charts.generate_supply_demand_analysis(self._data)
        
    def _generate_workday_analysis(self, output_dir: str) -> Dict[str, str]:
        """生成工作日分析图表"""
        if self._data is None:
            raise ValueError("数据未设置")
        return self.time_charts.generate_workday_analysis(self._data)
        
    def _generate_peak_valley_analysis(self, output_dir: str) -> Dict[str, str]:
        """生成峰谷分析图表"""
        if self._data is None:
            raise ValueError("数据未设置")
        return self.time_charts.generate_peak_valley_analysis(self._data)
        
    def _generate_prediction_analysis(self, output_dir: str) -> Dict[str, str]:
        """生成预测分析图表"""
        if self._data is None:
            raise ValueError("数据未设置")
        return self.regression_charts.generate_prediction_analysis(self._data)
        
    def generate_navigation_page(self) -> str:
        """生成导航页面"""
        return self.navigation.generate_navigation_page(self.output_dirs)
        
    def generate_all_charts(self, data: pd.DataFrame, analysis_results: Dict):
        """生成所有分析图表"""
        try:
            self.set_data(data)
            results = {}
            
            # 生成各类图表
            results['price'] = self._generate_price_analysis(self.output_dirs['price_analysis'])
            results['time'] = self._generate_time_patterns(self.output_dirs['time_patterns'])
            results['features'] = self._generate_feature_analysis(self.output_dirs['features'])
            results['regression'] = self._generate_regression_analysis(self.output_dirs['regression'])
            results['supply_demand'] = self._generate_supply_demand_analysis(self.output_dirs['supply_demand'])
            results['workday'] = self._generate_workday_analysis(self.output_dirs['workday_analysis'])
            results['peak_valley'] = self._generate_peak_valley_analysis(self.output_dirs['peak_valley'])
            results['prediction'] = self._generate_prediction_analysis(self.output_dirs['predictions'])
            
            # 生成导航页面
            results['navigation'] = self.generate_navigation_page()
            
            return results
            
        except Exception as e:
            self.logger.error(f"生成图表时出错: {str(e)}")
            return None
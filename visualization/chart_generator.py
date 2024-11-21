from typing import Dict, List, Optional, Union
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
from plotlib.subplots import make_subplots
from utils.logger import Logger
from visualization.chart_generator.price_charts import PriceChartGenerator
from visualization.chart_generator.regression_charts import RegressionChartGenerator
from visualization.chart_generator.time_charts import TimeChartGenerator
from visualization.chart_generator.feature_charts import FeatureChartGenerator
from visualization.chart_generator.backtest_charts import BacktestChartGenerator
from visualization.chart_generator.supply_demand import SupplyDemandChartGenerator
from visualization.chart_generator.navigation import NavigationGenerator

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
        try:
            # 确保数据是DataFrame类型
            if not isinstance(data, pd.DataFrame):
                raise ValueError("输入数据必须是pandas DataFrame类型")
            
            # 数据预处理
            if data.index.name == 'date':
                # 如果日期是索引，将其转换为列
                data = data.reset_index()
            elif 'datetime' in data.columns:
                # 如果有datetime列，将其重命名为date
                data = data.rename(columns={'datetime': 'date'})
            elif isinstance(data.index, pd.DatetimeIndex):
                # 如果索引是DatetimeIndex，将其转换为date列
                data = data.reset_index()
                data = data.rename(columns={'index': 'date'})
            
            # 确保date列存在
            if 'date' not in data.columns:
                self.logger.error("数据中缺少date列，当前列有: " + ", ".join(data.columns))
                raise ValueError("数据中缺少date列")
            
            # 确保date列是datetime类型
            data['date'] = pd.to_datetime(data['date'])
            
            self._data = data
            
        except Exception as e:
            self.logger.error(f"设置数据时出错: {str(e)}")
            raise
        
    def _generate_price_analysis(self, output_dir: str) -> Dict[str, str]:
        """生成价格分析图表"""
        if self._data is None:
            raise ValueError("数据未设置")
        self.price_charts.generate_price_distribution(self._data)
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
        
    def _generate_backtest_analysis(self, backtest_results: dict) -> Dict[str, str]:
        """生成回测分析图表"""
        return self.backtest_charts.generate_backtest_results(backtest_results)
        
    def _generate_workday_analysis(self, output_dir: str) -> Dict[str, str]:
        """生成工作日分析图表"""
        if self._data is None:
            raise ValueError("数据未设置")
        return self.time_charts.generate_workday_analysis(self._data)
        
    def _generate_prediction_analysis(self, output_dir: str) -> Dict[str, str]:
        """生成预测分析图表"""
        if self._data is None:
            raise ValueError("数据未设置")
        return self.regression_charts.generate_prediction_analysis(self._data)
    
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
            
            # 如果有回测结果，生成回测图表
            if 'backtest_results' in analysis_results:
                results['backtest'] = self._generate_backtest_analysis(analysis_results['backtest_results'])
            
            # 生成导航页面
            results['navigation'] = self.navigation.generate_navigation_page(self.output_dirs)
            
            return results
            
        except Exception as e:
            self.logger.error(f"生成图表时出错: {str(e)}")
            return None
    
    def plot_regression_analysis(self, regression_results: Dict) -> go.Figure:
        """
        生成回归分析图表
        
        Args:
            regression_results: 回归分析结果字典，包含以下键：
                - feature_importance: 特征重要性字典
                - y_true: 实际值数组
                - y_pred: 预测值数组
                - residuals: 残差数组
                - r2_score: R方值
                
        Returns:
            go.Figure: Plotly图表对象
        """
        try:
            if self._data is None:
                raise ValueError("数据未设置")
                
            # 检查回归结果字典中的必要键
            required_keys = ['feature_importance', 'y_true', 'y_pred', 'residuals', 'r2_score']
            missing_keys = [key for key in required_keys if key not in regression_results]
            if missing_keys:
                raise ValueError(f"回归结果字典缺少必要的键: {missing_keys}")
                
            # 调用回归图表生成器的方法
            return self.regression_charts.plot_regression_analysis(regression_results)
            
        except Exception as e:
            self.logger.error(f"生成回归分析图表时出错: {str(e)}")
            raise

def generate_price_comparison(data: pd.DataFrame) -> go.Figure:
    """生成价格对比图"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['date'], y=data['price'], mode='lines', name='价格'))
    fig.update_layout(title='价格对比图', xaxis_title='日期', yaxis_title='价格')
    return fig

def generate_period_heatmap(data: pd.DataFrame) -> go.Figure:
    """生成时段热力图"""
    # 实现热力图的生成逻辑
    pass

def generate_statistical_tables(data: pd.DataFrame) -> dict:
    """生成统计分析表格"""
    # 实现统计表格的生成逻辑
    pass

def generate_prediction_analysis(data: pd.DataFrame) -> go.Figure:
    """生成预测分析图表"""
    if data.empty:
        print("没有可用的预测数据。")
        raise ValueError("没有可用的预测数据。")
    
    print("预测数据:", data.head())  # 打印预测数据的前几行
    
    fig = go.Figure()
    # 生成图表逻辑
    return fig
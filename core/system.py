from pathlib import Path
import pandas as pd
from datetime import datetime
from time import time
import configparser
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict
import numpy as np

from config.config_manager import ConfigManager
from data_access.db_manager import DBManager, PowerPriceDataAccess
from core.analyzer.seasonal_analyzer import SeasonalAnalyzer
from core.analyzer.volatility_analyzer import VolatilityAnalyzer
from core.analyzer.factor_analyzer import FactorAnalyzer
from core.analyzer.period_pattern_analyzer import PeriodPatternAnalyzer
from visualization.chart_generator import ChartGenerator
from utils.logger import Logger
from utils.helpers import DataHelper
from core.backtest.engine import BacktestEngine
from core.data_processor import DataProcessor
from core.analysis_manager import AnalysisManager
from core.backtest_manager import BacktestManager
from core.analysis.time_pattern_analysis import TimePatternAnalyzer
from visualization.chart_generator import SupplyDemandChartGenerator

class TradingSystem:
    """交易系统主类"""
    
    def __init__(self):
        self.logger = Logger().get_logger()
        self.config, self.config_dict = self._load_config()
        self._init_managers()
        
    def _load_config(self):
        """加载配置"""
        try:
            config = configparser.ConfigParser()
            
            # 使用Path处理路径
            base_dir = Path(__file__).parent.parent  # 获取项目根目录
            config_path = base_dir / 'config' / 'config.ini'
            
            self.logger.info(f"尝试加载配置文件: {config_path}")
            
            if not config_path.exists():
                self.logger.error(f"配置文件不存在: {config_path}")
                raise FileNotFoundError(f"配置文件不存在: {config_path}")
                
            # 指定编码为utf-8
            config.read(str(config_path), encoding='utf-8')
            
            # 转换为字典格式并打印详细信息
            config_dict = {}
            for section in config.sections():
                config_dict[section] = dict(config[section])
           #     self.logger.info(f"配置节[{section}]包含以下配置项:")
                for key, value in config_dict[section].items():
                    self.logger.info(f"  {key}: {value}")
                    
            # 特别检查DATABASE配置
        #    if 'DATABASE' in config_dict:
        #        self.logger.info("数据库配置详情:")
        #        self.logger.info(f"DATABASE配置内容: {config_dict['DATABASE']}")
        #    else:
        #        self.logger.error("配置文件中缺少[DATABASE]节")
                
            return config, config_dict
            
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {str(e)}")
            return configparser.ConfigParser(), {}
            
    def _init_managers(self):
        """初始化各个管理器"""
        try:
            # 初始化数据库管理器，使用字典形式的配置
            self.db_manager = DBManager(self.config_dict.get('DATABASE', {}))
            
            # 初始化数据访问层
            self.data_access = PowerPriceDataAccess(self.db_manager)
            
            # 初始化数据处理器
            self.data_processor = DataProcessor(self.config_dict)
            
            # 初始化图表生成器
            self.chart_generator = ChartGenerator()
            
        except Exception as e:
            self.logger.error(f"初始化管理器失败: {str(e)}")
            raise
            
    def run_analysis(self, start_date, end_date):
        """运行分析"""
        try:
            # 获取数据
            data = self.data_access.get_price_data(start_date, end_date)
            if data is None or data.empty:
                self.logger.error("未获取有效数据")
                raise ValueError("未能获取有效数据")
                
            # 打印获取的数据列
            self.logger.info(f"获取的数据列: {data.columns.tolist()}")
            
            # 处理数据
            processed_data = self.data_processor.process_data(data)
            if processed_data is None:
                raise ValueError("数据处理返回None")
            
            # 打印处理后的数据列
            self.logger.info(f"处理后的数据列: {processed_data.columns.tolist()}")
            

            
            # 检查特征是否存在
            required_features = ['hour_sin', 'price_volatility', 'is_workday', 'returns', 'price_ma24']
            for feature in required_features:
                if feature not in processed_data.columns:
                    self.logger.error(f"处理后的数据中缺少 '{feature}' 特征")
                    raise ValueError(f"处理后的数据中缺少 '{feature}' 特征")
            
            # 添加数据验证
            self.logger.info("验证原始数据...")
            self.logger.info(f"数据形状: {data.shape}")
            self.logger.info(f"数据: {data.columns.tolist()}")
            
            # 确保必要的列存在
            required_columns = ['price', 'trade_hour']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                self.logger.error(f"数据缺少必要列: {missing_columns}")
                raise ValueError(f"数据缺少必要列: {missing_columns}")
                
            # 处理数据前先添加工作日标记
            try:
                # 确保trade_date列存在
                if 'trade_date' not in data.columns:
                    self.logger.warning("数据中缺少trade_date列，尝试从索引创建...")
                    if isinstance(data.index, pd.DatetimeIndex):
                        data['trade_date'] = data.index.date
                    else:
                        raise ValueError("无法创建trade_date列")
                
                # 添加工作日标记
                data['is_workday'] = data['trade_date'].apply(lambda x: 1 if x.weekday() < 5 else 0)
                self.logger.info("已添加工作日标记")
                
            except Exception as e:
                self.logger.error(f"添加工作日标记失败: {str(e)}")
                # 使用默认值
                data['is_workday'] = 1
                self.logger.warning("使用默认工作日标记(全部设为工作日)")
                
            # 处理数据
            try:
                self.logger.info("开始处理数据...")
                processed_data = self.data_processor.process_data(data)
                if processed_data is None:
                    raise ValueError("数据处理返回None")
                    
                # 验证处理后的数据
             #   self.logger.info(f"处理后数据形状: {processed_data.shape}")
             #   self.logger.info(f"处理后数据列: {processed_data.columns.tolist()}")
                
                # 确保索引是datetime类型
                if not isinstance(processed_data.index, pd.DatetimeIndex):
                    self.logger.info("转换索引为datetime类型...")
                    if 'trade_date' in processed_data.columns and 'trade_hour' in processed_data.columns:
                        processed_data['datetime'] = pd.to_datetime(processed_data['trade_date']) + \
                                                     pd.to_timedelta(processed_data['trade_hour'], unit='h')
                        processed_data.set_index('datetime', inplace=True)  # 设置为索引
                    else:
                        # 如果没有必要列，创建一个基础时间索引
                        processed_data.index = pd.date_range(
                            start=start_date,
                            periods=len(processed_data),
                            freq='H'
                        )
                    self.logger.info("索引转换完成")
                self.logger.info(f"处理后数据列: {processed_data.columns.to_list()}")
                #将trade_date转换为时间类型
                processed_data['trade_date'] = pd.to_datetime(processed_data['trade_date'])


                # 添加必要的时间特征
                processed_data['trade_hour'] = processed_data.index.hour
                processed_data['trade_date'] = processed_data.index.date
                processed_data['dayofweek'] = processed_data.index.dayofweek
                processed_data['month'] = processed_data.index.month

                # 确保所有数值列都是float类型
                numeric_columns = ['price', 'price_ma24', 'price_volatility', 'supply_pressure', 'demand_pressure']
                for col in numeric_columns:
                    if col in processed_data.columns:
                        processed_data[col] = processed_data[col].astype(float)

                # 添加供需压力指标
                if 'supply_pressure' not in processed_data.columns:
                    price_change = processed_data['price'].diff()
                    processed_data['supply_pressure'] = price_change.apply(lambda x: abs(min(x, 0)))
                    processed_data['demand_pressure'] = price_change.apply(lambda x: max(x, 0))
                    processed_data['supply_pressure_ma'] = processed_data['supply_pressure'].rolling(24, min_periods=1).mean()
                    processed_data['demand_pressure_ma'] = processed_data['demand_pressure'].rolling(24, min_periods=1).mean()

                # 添加其他必要的特征
                if 'is_workday' not in processed_data.columns:
                    processed_data['is_workday'] = processed_data.index.dayofweek < 5

                # 添加峰谷时段标记
                hour = processed_data.index.hour
                processed_data['is_morning_peak'] = ((hour >= 8) & (hour < 12)).astype(int)
                processed_data['is_evening_peak'] = ((hour >= 17) & (hour < 21)).astype(int)
                processed_data['is_valley'] = ((hour >= 0) & (hour < 6)).astype(int)

                # 检查关键列
                for col in ['price', 'trade_hour', 'is_workday']:
                    if col not in processed_data.columns:
                        self.logger.error(f"处理后数据缺少关键列: {col}")
                        raise ValueError(f"处理后数据缺少关键列: {col}")
                        
                # 检查空
                null_cols = processed_data.columns[processed_data.isnull().any()].tolist()
                if null_cols:
                    self.logger.warning(f"以下列存在空值: {null_cols}")
                    # 对关键列进行填充
                    for col in null_cols:
                        processed_data[col] = processed_data[col].fillna(method='ffill').fillna(method='bfill')
                        self.logger.info(f"已填充 {col} 列的空值")
                        
                # 保存数据供图表生成使用
                self.chart_generator._data = processed_data
                
                # 生成所有类别的图表
                self.logger.info("开始生成所有分析图表...")
                try:
                    # 1. 生成价格分析图表
                    self.logger.info("生成价格分析图表...")
                    self.chart_generator._generate_price_analysis(self.chart_generator.output_dirs['price_analysis'])
                    
                    # 2. 生成时间模式图表
                    self.logger.info("生成时间模式图表...")
                    self.chart_generator._generate_time_patterns(self.chart_generator.output_dirs['time_patterns'])
                    
                    # 3. 生成特征分析图表
                    self.logger.info("生成特征分析图表...")
                    self.chart_generator._generate_feature_analysis(self.chart_generator.output_dirs['features'])
                    
                    # 4. 生成回归分析图表
                    self.logger.info("生成回归分析图表...")
                    self.chart_generator._generate_regression_analysis(self.chart_generator.output_dirs['regression'])
                    
                    # 5. 生成供需分析图表
                    self.logger.info("生成供需分析图表...")
                    self.chart_generator._generate_supply_demand_analysis(self.chart_generator.output_dirs['supply_demand'])
                    
                    # 6. 生成工作日分析图表
                    self.logger.info("生成工作日分析图表...")
                    self.chart_generator._generate_workday_analysis(self.chart_generator.output_dirs['workday_analysis'])
                    
                    # 7. 生成峰谷分析图表
                    self.logger.info("生成峰谷分析图表...")
                    self.chart_generator._generate_peak_valley_analysis(self.chart_generator.output_dirs['peak_valley'])
                    
                    # 8. 生成预测分析图表
                    self.logger.info("生成预测分析图表...")
                    self.chart_generator._generate_prediction_analysis(self.chart_generator.output_dirs['predictions'])
                    
                    # 生成导航页面
                    self.logger.info("生成导航页面...")
                    nav_path = self.chart_generator.generate_navigation_page()
                    if nav_path:
                        self.logger.info(f"导航页面已生成: {nav_path}")
                        
                        # 验证生成的文件
                        nav_file = Path(nav_path)
                        if nav_file.exists():
                            self.logger.info("导航页面文件已确认存在")
                            
                            # 打印所有生成的图表文件
                            self.logger.info("\n=== 已生成的分析文件 ===")
                            base_dir = Path('trading_system/output')
                            if base_dir.exists():
                                for path in base_dir.rglob('*.html'):
                                    self.logger.info(f"- {path.relative_to(base_dir)}")
                        else:
                            self.logger.error(f"导航页面文件未找到: {nav_path}")
                    else:
                        self.logger.error("导航页面生成失败")
                    
                except Exception as e:
                    self.logger.error(f"生成图表时出错: {str(e)}")
                    raise
                    
            except Exception as e:
                self.logger.error(f"数据处理失败: {str(e)}")
                raise ValueError("数据处理失败")
                
        except Exception as e:
            self.logger.error(f"运行分析失败: {str(e)}")
            raise
            
    def _analyze_data(self, data):
        """分析数据"""
        try:
            # 初始化结果字典
            results = {
                'price_stats': None,
                'time_patterns': None,
                'supply_demand_impact': None,  # 添加供需分析结果的占位符
                'regression_results': None,    # 添加回归分析结果的占位符
                'feature_analysis': None       # 添加特征分析结果的占位符
            }
            
            # 基础统计分析
            try:
                price_stats = data['price'].describe()
                results['price_stats'] = price_stats
            except Exception as e:
                self.logger.error(f"计算价格统计信息失败: {str(e)}")
            
            # 添加时间特征分析
            try:
                results['time_patterns'] = {
                    'hourly_avg': data.groupby('trade_hour')['price'].mean(),
                    'workday_avg': data.groupby('is_workday')['price'].mean(),
                    'peak_valley_avg': {
                        'morning_peak': data[data['is_morning_peak'] == 1]['price'].mean(),
                        'evening_peak': data[data['is_evening_peak'] == 1]['price'].mean(),
                        'valley': data[data['is_valley'] == 1]['price'].mean()
                    }
                }
            except Exception as e:
                self.logger.error(f"计算时间模式分析失败: {str(e)}")
            
            # 添加特征分析
            try:
                feature_cols = [col for col in data.columns if col != 'price']
                results['feature_analysis'] = {
                    'correlations': data[feature_cols].corrwith(data['price']).to_dict(),
                    'feature_stats': {
                        col: data[col].describe().to_dict() 
                        for col in feature_cols
                    }
                }
            except Exception as e:
                self.logger.error(f"计算特征分析失败: {str(e)}")
            
            return results  # 始终返回结果字典，即使某些分析失败
            
        except Exception as e:
            self.logger.error(f"数据分析失败: {str(e)}")
            return {  # 返回空结果字典而不是None
                'price_stats': None,
                'time_patterns': None,
                'supply_demand_impact': None,
                'regression_results': None,
                'feature_analysis': None
            }
            
    # 暂时注释掉回测相关方法
    # def run_backtest(self, data, strategy):
    #     """运行回测"""
    #     ...
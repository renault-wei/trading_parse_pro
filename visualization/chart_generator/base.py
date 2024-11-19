from pathlib import Path
from utils.logger import Logger
import pandas as pd

class BaseChartGenerator:
    """基础图表生成器"""
    
    def __init__(self, theme: str = 'plotly_dark', output_dirs: dict = None):
        self.theme = theme
        self.logger = Logger().get_logger()
        self.default_layout = {
            'template': theme,
            'height': 800,
            'width': 1200,
            'showlegend': True,
            'margin': dict(l=50, r=50, t=150, b=50)
        }
        
        if output_dirs is None:
            self.output_dirs = {
                'regression': 'trading_system/output/regression_analysis',
                'analysis': 'trading_system/output/analysis',
                'predictions': 'trading_system/output/predictions',
                'backtest': 'trading_system/output/backtest',
                'features': 'trading_system/output/features',
                'time_patterns': 'trading_system/output/time_patterns',
                'supply_demand': 'trading_system/output/supply_demand',
                'price_analysis': 'trading_system/output/price_analysis',
                'workday_analysis': 'trading_system/output/workday_analysis',
                'peak_valley': 'trading_system/output/peak_valley'
            }
        else:
            self.output_dirs = output_dirs
        
        self._init_output_dirs()
        
    def _init_output_dirs(self):
        """初始化输出目录"""
        for dir_path in self.output_dirs.values():
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            
    def _ensure_datetime_index(self, data: pd.DataFrame) -> pd.DataFrame:
        """确保数据使用datetime索引"""
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        return data.sort_index() 
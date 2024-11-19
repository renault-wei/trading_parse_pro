import pandas as pd
import mysql.connector
from sqlalchemy import create_engine
from datetime import datetime
from typing import Dict, Optional
import logging
from urllib.parse import quote_plus
import numpy as np

class DBManager:
    """数据库管理器"""
    def __init__(self, config: Dict[str, str]):
        self.config = config
        self._engine = None
        self.logger = logging.getLogger('trading_system')
        
        # 检查必要的配置项
        required_fields = ['host', 'port', 'user', 'password', 'database']
        missing_fields = [field for field in required_fields if field not in config]
        if missing_fields:
            error_msg = f"数据库配置缺少必要字段: {missing_fields}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
    
    @property
    def engine(self):
        """获取SQLAlchemy引擎"""
        if self._engine is None:
            try:
                # 确保端口是整数
                port = int(self.config['port'])
                # 对密码进行URL编码
                password = quote_plus(self.config['password'])
                conn_str = (f"mysql+mysqlconnector://{self.config['user']}:{password}"
                           f"@{self.config['host']}:{port}/{self.config['database']}")
                self._engine = create_engine(conn_str)
                self.logger.info(f"成功创建数据库引擎，连接到: {self.config['host']}:{port}")
            except Exception as e:
                self.logger.error(f"创建数据库引擎失败: {str(e)}")
                raise
        return self._engine
    
    def connect(self):
        """建立MySQL连接"""
        try:
            # 确保使用正确的端口类型和密码编码
            config = self.config.copy()
            config['port'] = int(config['port'])
            config['password'] = quote_plus(config['password'])
            conn = mysql.connector.connect(**config)
            self.logger.info("成功建立数据库连接")
            return conn
        except Exception as e:
            self.logger.error(f"数据库连接错误: {str(e)}")
            return None
    
    def load_data(self, start_date, end_date):
        """加载数据"""
        try:
            # 模拟从数据库加载数据
            data = pd.DataFrame({
                'trade_date': pd.date_range(start=start_date, end=end_date, freq='H'),
                'price': np.random.uniform(100, 500, size=len(pd.date_range(start=start_date, end=end_date, freq='H'))),
                'hour': [(x.hour + 1) for x in pd.date_range(start=start_date, end=end_date, freq='H')],  # 1-24
                'day': [x.date() for x in pd.date_range(start=start_date, end=end_date, freq='H')],
                'time': [f"{(x.hour + 1):02d}:00:00" for x in pd.date_range(start=start_date, end=end_date, freq='H')]  # 使用1-24小时制
            })
            
            self.logger.info(f"数据加载成功，形状: {data.shape}")
            self.logger.info(f"数据列: {data.columns.tolist()}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"数据加载失败: {str(e)}")
            return None

class PowerPriceDataAccess:
    """电力价格数据访问类"""
    def __init__(self, db_manager: DBManager):
        self.db_manager = db_manager
        self.logger = logging.getLogger('trading_system')
    
    def get_price_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取价格数据"""
        query = """
        SELECT 
            a.trade_date, 
            a.trade_hour-1  as trade_hour,
            a.price,  # 使用正确的字段名
            COALESCE(b.data_type, 0) as data_type
        FROM trade_table a 
        LEFT JOIN date_info b ON a.trade_date = b.c_date
        WHERE a.trade_date >= %s 
            AND a.trade_date < %s 
        ORDER BY a.trade_date, a.trade_hour
        """
        
        try:
            self.logger.info(f"正在查询从 {start_date} 到 {end_date} 的价格数据")
            
            # 使用SQLAlchemy引擎
            df = pd.read_sql(
                query, 
                self.db_manager.engine, 
                params=(start_date, end_date)
            )
            
            if df.empty:
                self.logger.warning("查询结果为空")
                return pd.DataFrame()
            
            # 打印原始数据的前几行
            self.logger.info(f"原始数据前5行:\n{df.head()}")
            self.logger.info(f"数据类型:\n{df.dtypes}")
            
            # 数据预处理
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df['trade_hour'] = df['trade_hour'].astype(int)
            # price字段已经是正确的名称，不需要重命名
            df['data_type'] = df['data_type'].fillna(0).astype(int)
            
            # 添加数据检查日志
            self.logger.info(f"""
            数据统计信息:
            - 总记录数: {len(df)}
            - 日期范围: {df['trade_date'].min()} 到 {df['trade_date'].max()}
            - 小时范围: {df['trade_hour'].min()} 到 {df['trade_hour'].max()}
            - 价格列名: {df.columns}
            - 价格数据:
              - 非空值数: {df['price'].count()}
              - 空值数: {df['price'].isna().sum()}
              - 均值: {df['price'].mean():.2f}
              - 标准差: {df['price'].std():.2f}
              - 中位数: {df['price'].median():.2f}
              - 最小值: {df['price'].min():.2f}
              - 最大值: {df['price'].max():.2f}
            """)
            
            return df
                
        except Exception as e:
            self.logger.error(f"获取价格数据错误: {str(e)}")
            raise
    
    def get_date_info(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取日期信息"""
        query = """
        SELECT 
            c_date,
            data_type,
            holiday_name
        FROM date_info
        WHERE c_date >= %s AND c_date < %s
        ORDER BY c_date
        """
        
        try:
            df = pd.read_sql(
                query, 
                self.db_manager.engine, 
                params=(start_date, end_date)
            )
            df['c_date'] = pd.to_datetime(df['c_date'])
            return df
            
        except Exception as e:
            print(f"获取日期信息错误: {e}")
            return pd.DataFrame()
    
    def save_analysis_results(self, results: Dict, analysis_date: Optional[str] = None) -> bool:
        """保存分析结果"""
        if analysis_date is None:
            analysis_date = datetime.now().strftime('%Y-%m-%d')
            
        try:
            # 使用SQLAlchemy引擎创建连接
            with self.db_manager.engine.connect() as conn:
                # 保存基础统计结果
                if 'price_stats' in results:
                    stats = results['price_stats']
                    query = """
                    INSERT INTO price_stats 
                    (analysis_date, mean_price, std_price, min_price, max_price, 
                     percentile_25, percentile_50, percentile_75)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    conn.execute(
                        query,
                        (
                            analysis_date,
                            stats.get('mean', 0),
                            stats.get('std', 0),
                            stats.get('min', 0),
                            stats.get('max', 0),
                            stats.get('25%', 0),
                            stats.get('50%', 0),
                            stats.get('75%', 0)
                        )
                    )
                
                conn.commit()
                return True
                
        except Exception as e:
            print(f"保存分析结果错误: {e}")
            return False

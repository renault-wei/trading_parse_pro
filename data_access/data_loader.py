from datetime import datetime
import pandas as pd
from typing import Optional
import logging
from .database import DatabaseManager

class DataLoader:
    """数据加载器"""
    
    def __init__(self):
        self.logger = logging.getLogger('trading_system')
        self.db = DatabaseManager()
        
    def load_price_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = '1d'
    ) -> pd.DataFrame:
        """
        从MySQL加载价格数据
        
        Args:
            symbol: 交易对符号
            start_date: 开始日期
            end_date: 结束日期
            timeframe: 时间周期
            
        Returns:
            DataFrame包含 OHLCV 数据
        """
        try:
            self.db.connect()
            query = """
            SELECT 
                timestamp,
                open,
                high,
                low,
                close,
                volume
            FROM market_data
            WHERE 
                symbol = %s 
                AND timestamp BETWEEN %s AND %s
                AND timeframe = %s
            ORDER BY timestamp
            """
            
            df = self.db.execute_query(
                query,
                (
                    symbol,
                    start_date.strftime('%Y-%m-%d %H:%M:%S'),
                    end_date.strftime('%Y-%m-%d %H:%M:%S'),
                    timeframe
                )
            )
            
            if df is None or df.empty:
                raise ValueError(f"未找到 {symbol} 在指定时间段的数据")
                
            self.logger.info(f"成功加载 {len(df)} 条记录")
            return self._process_dataframe(df)
            
        except Exception as e:
            self.logger.error(f"数据加载失败: {str(e)}")
            raise
        finally:
            self.db.disconnect()
            
    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理数据框"""
        # 确保列名统一
        df.columns = [col.lower() for col in df.columns]
        
        # 确保有所需的所有列
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"数据中缺少必需的列: {col}")
                
        # 确保索引是datetime类型
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        return df
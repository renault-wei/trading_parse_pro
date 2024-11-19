import mysql.connector
import pandas as pd
from typing import Optional
import logging
from pathlib import Path
from config.config_manager import ConfigManager

class DatabaseManager:
    """MySQL数据库管理器"""
    
    def __init__(self):
        self.logger = logging.getLogger('trading_system')
        self.config = ConfigManager()
        self.conn = None
        self.cursor = None
        
    def connect(self):
        """连接到MySQL数据库"""
        try:
            self.conn = mysql.connector.connect(
                host=self.config.get('database', 'host'),
                port=self.config.get_int('database', 'port'),
                database=self.config.get('database', 'database'),
                user=self.config.get('database', 'user'),
                password=self.config.get('database', 'password')
            )
            self.cursor = self.conn.cursor(dictionary=True)
            self.logger.info("成功连接到MySQL数据库")
        except Exception as e:
            self.logger.error(f"数据库连接失败: {str(e)}")
            raise
            
    def disconnect(self):
        """关闭数据库连接"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            self.logger.info("数据库连接已关闭")
            
    def execute_query(self, query: str, params: tuple = None) -> Optional[pd.DataFrame]:
        """执行SQL查询"""
        try:
            if params:
                result = pd.read_sql_query(query, self.conn, params=params)
            else:
                result = pd.read_sql_query(query, self.conn)
            return result
        except Exception as e:
            self.logger.error(f"查询执行失败: {str(e)}")
            raise
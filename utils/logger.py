import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
import shutil

class Logger:
    """日志管理器"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.logger = logging.getLogger('trading_system')
        if not self.logger.handlers:  # 避免重复初始化
            self._clean_old_logs()  # 清理旧日志
            self._setup_logger()
            
    def _clean_old_logs(self):
        """清理旧日志文件"""
        log_dir = Path('logs')
        if log_dir.exists():
            try:
                # 删除整个日志目录
                shutil.rmtree(log_dir)
                print(f"已清理旧日志目录: {log_dir}")
            except Exception as e:
                print(f"清理日志目录失败: {str(e)}")
        
        # 重新创建日志目录
        log_dir.mkdir(exist_ok=True)
            
    def _setup_logger(self):
        """配置日志"""
        self.logger.setLevel(logging.DEBUG)
        
        # 创建日志目录
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        
        # 文件处理器
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_handler = logging.FileHandler(
            log_dir / f'trading_system_{current_time}.log',
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        # 添加处理器
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        
        # 记录启动信息
        self.logger.info("日志系统初始化完成")
        self.logger.info(f"日志文件路径: {log_dir}")
        
    def get_logger(self) -> logging.Logger:
        """获取日志器"""
        return self.logger
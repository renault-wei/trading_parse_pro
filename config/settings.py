import configparser
import os

class Settings:
    """配置管理器"""
    
    def __init__(self, config_path: str = 'config/config.ini'):
        self.config = configparser.ConfigParser()
        self.config_path = config_path
        self.load_config()
        
    def load_config(self):
        """加载配置"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        self.config.read(self.config_path)
        
    def get(self, section: str, key: str, fallback=None):
        """获取配置值"""
        return self.config.get(section, key, fallback=fallback) 
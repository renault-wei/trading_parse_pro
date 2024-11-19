import configparser
from pathlib import Path
from typing import List, Union

class ConfigManager:
    """配置管理器"""
    def __init__(self, config_path: Union[str, Path]):
        self.config = configparser.ConfigParser()
        self.config.read(config_path, encoding='utf-8')
    
    def get(self, section: str, key: str, fallback: str = None) -> str:
        """获取字符串配置"""
        return self.config.get(section, key, fallback=fallback)
    
    def get_int(self, section: str, key: str, fallback: int = None) -> int:
        """获取整数配置"""
        return self.config.getint(section, key, fallback=fallback)
    
    def get_float(self, section: str, key: str, fallback: float = None) -> float:
        """获取浮点数配置"""
        return self.config.getfloat(section, key, fallback=fallback)
    
    def get_bool(self, section: str, key: str, fallback: bool = None) -> bool:
        """获取布尔值配置"""
        return self.config.getboolean(section, key, fallback=fallback)
    
    def get_list(self, section: str, key: str, fallback: List = None) -> List:
        """获取列表配置（以逗号分隔的字符串）"""
        value = self.get(section, key)
        if value is None:
            return fallback or []
        return [item.strip() for item in value.split(',')]

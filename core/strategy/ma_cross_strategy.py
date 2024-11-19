from datetime import datetime
import pandas as pd
from typing import Dict, List
from core.strategy.base_strategy import BaseStrategy

class MACrossStrategy(BaseStrategy):
    """均线交叉策略"""
    
    def __init__(self, params: dict = None):
        super().__init__(params)
        # 定义多个均线周期
        self.ma_periods = {
            'fast': params.get('fast_window', 24),  # 24小时快线
            'medium': params.get('medium_window', 48),  # 48小时中线
            'slow': params.get('slow_window', 72),  # 72小时慢线
            'trend': params.get('trend_window', 168)  # 168小时(一周)趋势线
        }
        self.qty = params.get('qty', 1.0)
        
    def calculate_mas(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算多条均线"""
        mas = pd.DataFrame(index=data.index)
        for name, period in self.ma_periods.items():
            mas[f'ma_{name}'] = data['price'].rolling(window=period).mean()
        return mas
        
    def generate_signals(self, timestamp: datetime, data: pd.DataFrame, positions: Dict) -> List[dict]:
        """生成交易信号"""
        signals = []
        
        # 确保有足够的数据计算均线
        if len(data) < max(self.ma_periods.values()):
            return signals
            
        # 计算所有均线
        mas = self.calculate_mas(data)
        
        # 获取当前和前一个周期的均线值
        current = mas.iloc[-1]
        previous = mas.iloc[-2]
        
        # 检查是否持仓
        current_position = positions.get('default', None)
        position_qty = current_position.quantity if current_position else 0
        
        # 交易信号生成逻辑
        # 1. 快线穿越中线
        fast_cross_medium = (
            previous['ma_fast'] <= previous['ma_medium'] and 
            current['ma_fast'] > current['ma_medium']
        )
        
        # 2. 快线穿越慢线
        fast_cross_slow = (
            previous['ma_fast'] <= previous['ma_slow'] and 
            current['ma_fast'] > current['ma_slow']
        )
        
        # 3. 中线穿越慢线
        medium_cross_slow = (
            previous['ma_medium'] <= previous['ma_slow'] and 
            current['ma_medium'] > current['ma_slow']
        )
        
        # 买入条件：任意两条快线向上穿越较慢线
        if (fast_cross_medium or fast_cross_slow or medium_cross_slow) and position_qty <= 0:
            # 确认趋势：所有均线都在趋势线上方
            if (current['ma_fast'] > current['ma_trend'] and 
                current['ma_medium'] > current['ma_trend'] and 
                current['ma_slow'] > current['ma_trend']):
                signals.append({
                    'direction': 'buy',
                    'quantity': self.qty,
                    'symbol': 'default'
                })
                
        # 卖出条件：反向穿越
        elif ((previous['ma_fast'] >= previous['ma_medium'] and current['ma_fast'] < current['ma_medium']) or
              (previous['ma_fast'] >= previous['ma_slow'] and current['ma_fast'] < current['ma_slow']) or
              (previous['ma_medium'] >= previous['ma_slow'] and current['ma_medium'] < current['ma_slow'])):
            if position_qty > 0:
                signals.append({
                    'direction': 'sell',
                    'quantity': self.qty,
                    'symbol': 'default'
                })
                
        return signals 
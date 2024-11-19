from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from utils.logger import Logger
import logging

@dataclass
class Position:
    """持仓信息"""
    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    direction: str = ''  # 'long' or 'short'
    unrealized_pnl: float = 0.0
    
    def __post_init__(self):
        """确保数值字段为浮点数类型"""
        self.quantity = float(self.quantity)
        self.avg_price = float(self.avg_price)
        self.unrealized_pnl = float(self.unrealized_pnl)
    
@dataclass
class Trade:
    """交易记录"""
    timestamp: datetime
    symbol: str
    direction: str  # 'buy' or 'sell'
    quantity: float
    price: float
    commission: float
    slippage: float
    
class BacktestEngine:
    """回测引擎"""
    
    def __init__(self, data: pd.DataFrame, config: dict):
        """
        初始化回测引擎
        
        Args:
            data: 回测数据，DataFrame格式，必须包含 datetime 索引和 price 列
            config: 回测配置，包含初始资金、手续费率等
        """
        self.data = data
        self.initial_capital = float(config.get('initial_capital', 1000000))
        self.commission_rate = float(config.get('commission_rate', 0.0003))
        self.slippage = float(config.get('slippage', 0.0001))
        self.logger = Logger().get_logger()  # 添加logger
        
        # 回测状态
        self.current_capital = self.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.daily_stats = pd.DataFrame()  # 修改为DataFrame
        
        # 回测结果
        self.performance_metrics = {}
        
    def run_backtest(self, strategy):
        """运行回测"""
        try:
            results = {
                'trades': [],
                'positions': [],
                'equity_curve': [],
                'returns': []
            }
            
            # 使用实际的数据时间范围
            timestamps = self.data.index
            
            for i, timestamp in enumerate(timestamps):
                # 获取历史数据
                historical_data = self.data.iloc[:i+1]
                
                # 获取当前数据
                current_data = self.data.iloc[i:i+1]
                
                # 生成信号
                signal = strategy.generate_signals(
                    timestamp,
                    historical_data,
                    self.positions
                )
                
                # 执行交易
                if signal is not None and len(signal) > 0:
                    trade_result = self._execute_trade(signal.iloc[0], current_data)
                    if trade_result:
                        results['trades'].append(trade_result)
                
                # 更新持仓和资金
                self._update_positions(timestamp)
                equity = self._calculate_equity(timestamp)
                results['equity_curve'].append({
                    'timestamp': timestamp,
                    'equity': equity
                })
            
            # 计算回测结果
            try:
                performance_metrics = self._calculate_performance()
                if performance_metrics is not None:
                    results.update(performance_metrics)
                else:
                    # 如果性能指标计算失败，添加默认值
                    results.update({
                        'total_returns': 0,
                        'annual_returns': 0,
                        'sharpe_ratio': 0,
                        'max_drawdown': 0,
                        'total_trades': len(results['trades']),
                        'trading_days': len(set(t['timestamp'].date() for t in results['trades'])) if results['trades'] else 0,
                        'win_rate': 0,
                        'max_position': 0,
                        'margin': 0
                    })
                    self.logger.warning("使用默认性能指标")
            except Exception as e:
                self.logger.error(f"计算性能指标时出错: {str(e)}")
                # 添加默认值
                results.update({
                    'total_returns': 0,
                    'annual_returns': 0,
                    'sharpe_ratio': 0,
                    'max_drawdown': 0,
                    'total_trades': len(results['trades']),
                    'trading_days': len(set(t['timestamp'].date() for t in results['trades'])) if results['trades'] else 0,
                    'win_rate': 0,
                    'max_position': 0,
                    'margin': 0
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"回测执行出错: {str(e)}")
            raise
            
    def get_performance_metrics(self) -> dict:
        """获取回测性能指标"""
        try:
            # 计算基础指标
            total_returns = (self.current_capital - self.initial_capital) / self.initial_capital
            days = len(self.daily_stats)
            annual_returns = total_returns * (252 / days)  # 年化收益率
            
            # 计算夏普比率
            daily_returns = self.daily_stats['daily_returns']
            risk_free_rate = 0.02  # 假设无风险利率为2%
            excess_returns = daily_returns - risk_free_rate/252
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
            
            # 计算最大回撤
            cumulative_returns = (1 + daily_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = cumulative_returns / rolling_max - 1
            max_drawdown = drawdowns.min()
            
            # 计算其他指标
            winning_trades = sum(1 for t in self.trades if getattr(t, 'pnl', 0) > 0)
            total_trades = len(self.trades)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # 计算最大持仓，处理空持仓情况
            max_position = 0.0
            if self.positions:
                position_sizes = [abs(p.quantity) for p in self.positions.values()]
                if position_sizes:  # 确保列表不为空
                    max_position = max(position_sizes)
            
            # 添加日志记录
            self.logger.debug(f"计算性能指标:")
            self.logger.debug(f"- 总收益率: {total_returns:.2%}")
            self.logger.debug(f"- 年化收益率: {annual_returns:.2%}")
            self.logger.debug(f"- 夏普比率: {sharpe_ratio:.2f}")
            self.logger.debug(f"- 最大回撤: {max_drawdown:.2%}")
            self.logger.debug(f"- 胜率: {win_rate:.2%}")
            self.logger.debug(f"- 最大持仓: {max_position}")
            
            return {
                'initial_capital': self.initial_capital,
                'final_capital': self.current_capital,
                'total_returns': total_returns,
                'annual_returns': annual_returns,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'total_trades': total_trades,
                'trading_days': days,
                'win_rate': win_rate,
                'max_position': max_position,
                'margin': self.current_capital * 0.1,  # 示例：假设保证金为资金的10%
                'daily_stats': self.daily_stats
            }
            
        except Exception as e:
            self.logger.error(f"计算性能指标时出错: {str(e)}")
            raise

    def _update_positions(self, timestamp):
        """
        更新持仓盈亏
        
        Args:
            timestamp: 当前时间戳
        """
        try:
            # 获取当前价格
            try:
                # 检查是否是多列数据
                if isinstance(self.data.loc[timestamp, 'price'], pd.Series):
                    current_price = float(self.data.loc[timestamp, 'price'].iloc[0])
                else:
                    current_price = float(self.data.loc[timestamp, 'price'])
                    
                # 添加调试信息
                self.logger.debug(f"时间戳: {timestamp}, 当前价格: {current_price}")
                
            except Exception as e:
                self.logger.error(f"获取价格数据时出错: {str(e)}")
                self.logger.debug(f"时间戳: {timestamp}")
                self.logger.debug(f"数据类型: {type(self.data.loc[timestamp, 'price'])}")
                self.logger.debug(f"数据列: {self.data.columns.tolist()}")
                return
            
            for symbol, position in self.positions.items():
                if position.quantity != 0:
                    # 确保价格和持仓数量都是数值类型
                    try:
                        position_qty = float(position.quantity)
                        avg_price = float(position.avg_price)
                        
                        # 计算未实现盈亏
                        if position_qty > 0:  # 多仓
                            position.unrealized_pnl = (current_price - avg_price) * position_qty
                        else:  # 空仓
                            position.unrealized_pnl = (avg_price - current_price) * abs(position_qty)
                        
                    except ValueError as e:
                        self.logger.error(f"转换数据类型时出错: {str(e)}")
                        self.logger.debug(f"持仓数量: {position.quantity}, 类型: {type(position.quantity)}")
                        self.logger.debug(f"平均价格: {position.avg_price}, 类型: {type(position.avg_price)}")
                        continue
                        
        except Exception as e:
            self.logger.error(f"更新持仓时出错: {str(e)}")
            self.logger.debug(f"当前时间戳: {timestamp}")
            self.logger.debug(f"持仓信息: {self.positions}")

    def _execute_trade(self, signal, current_data):
        """执行交易"""
        try:
            if not signal:
                return None
                
            # 确保信号包所有必要的字段
            required_fields = ['direction', 'quantity', 'price', 'timestamp']
            if not all(field in signal for field in required_fields):
                self.logger.error(f"信号缺少必要字段: {[f for f in required_fields if f not in signal]}")
                return None
            
            # 添加默认的 symbol
            if 'symbol' not in signal:
                signal['symbol'] = 'default'
                
            trade_price = current_data['price'].iloc[0]
            trade = {
                'timestamp': signal['timestamp'],
                'symbol': signal['symbol'],
                'direction': signal['direction'],
                'quantity': signal['quantity'],
                'price': trade_price,
                'value': trade_price * signal['quantity'],
                'type': 'LONG' if signal['direction'] > 0 else 'SHORT'
            }
            
            # 更新持仓
            if trade['quantity'] > 0:  # 只在有实际交易量时更新持仓
                self._update_position(trade)
            
            return trade
            
        except Exception as e:
            self.logger.error(f"执行交易时出错: {str(e)}")
            return None

    def _calculate_performance(self):
        """计算回测性能指标"""
        try:
            self.logger.info("开始计算性能指标...")
            
            # 计算基础指标
            total_returns = (self.current_capital - self.initial_capital) / self.initial_capital
            days = len(self.daily_stats)
            annual_returns = total_returns * (252 / days)  # 年化收益率
            
            # 计算夏普比率
            daily_returns = self.daily_stats['daily_returns']
            risk_free_rate = 0.02  # 假设无风险利率为2%
            excess_returns = daily_returns - risk_free_rate/252
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
            
            # 计算最大回撤
            cumulative_returns = (1 + daily_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = cumulative_returns / rolling_max - 1
            max_drawdown = drawdowns.min()
            
            # 计算其他指标
            winning_trades = sum(1 for t in self.trades if getattr(t, 'pnl', 0) > 0)
            total_trades = len(self.trades)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # 计算最大持仓，处理空持仓情况
            max_position = 0.0
            if self.positions:
                max_position = max(abs(p.quantity) for p in self.positions.values())
            
            # 保存性能指标
            self.performance_metrics = {
                'initial_capital': self.initial_capital,
                'final_capital': self.current_capital,
                'total_returns': total_returns,
                'annual_returns': annual_returns,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'total_trades': total_trades,
                'trading_days': days,
                'win_rate': win_rate,
                'max_position': max_position,
                'margin': self.current_capital * 0.1,  # 示例：假设保证金为资金的10%
                'daily_stats': self.daily_stats
            }
            
            self.logger.info("性能指标计算完成")
            self.logger.debug(f"性能指标摘要:")
            self.logger.debug(f"- 总收益率: {total_returns:.2%}")
            self.logger.debug(f"- 年化收益率: {annual_returns:.2%}")
            self.logger.debug(f"- 夏普比率: {sharpe_ratio:.2f}")
            self.logger.debug(f"- 最大回撤: {max_drawdown:.2%}")
            self.logger.debug(f"- 胜率: {win_rate:.2%}")
            self.logger.debug(f"- 最大持仓: {max_position}")
            
        except Exception as e:
            self.logger.error(f"计算性能指标时出错: {str(e)}")
            raise

    def get_trades_history(self) -> List[Trade]:
        """获取交易历史"""
        return self.trades

    def _prepare_backtest_data(self, data):
        """准备回测数据"""
        try:
            # 确保数据按时间排序
            data = data.sort_index()
            
            # 确保时间索引是连续的
            start_time = data.index.min()
            end_time = data.index.max()
            expected_index = pd.date_range(start_time, end_time, freq='H')
            
            # 检查是否有缺失的时间点
            missing_times = expected_index.difference(data.index)
            if len(missing_times) > 0:
                self.logger.warning(f"发现 {len(missing_times)} 个缺失的时间点")
                # 可以选择填充或跳过缺失的时间点
                
            # 确保所有必要的列都存在
            required_columns = ['price', 'log_price', 'hour', 'date']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"缺少必要的列: {missing_columns}")
                
            return data
            
        except Exception as e:
            self.logger.error(f"准备回测数据时出错: {str(e)}")
            raise

    def _calculate_equity(self, timestamp):
        """计算当前时点的权益"""
        try:
            # 计算当前持仓的市值
            position_value = 0
            for position in self.positions.values():
                if position.quantity != 0:
                    # 获取当前价格
                    try:
                        if isinstance(self.data.loc[timestamp, 'price'], pd.Series):
                            current_price = float(self.data.loc[timestamp, 'price'].iloc[0])
                        else:
                            current_price = float(self.data.loc[timestamp, 'price'])
                        position_value += position.quantity * current_price
                    except Exception as e:
                        self.logger.error(f"获取价格数据时出错: {str(e)}")
                        continue
            
            # 计算总权益 = 当前资金 + 持仓市值
            total_equity = self.current_capital + position_value
            
            # 记录每日权益数据
            if len(self.daily_stats) == 0 or self.daily_stats.index[-1].date() != timestamp.date():
                # 计算日收益率
                if len(self.daily_stats) > 0:
                    prev_equity = self.daily_stats['equity'].iloc[-1]
                    daily_return = (total_equity - prev_equity) / prev_equity
                else:
                    daily_return = 0
                
                # 使用 concat 替代 append
                new_stats = pd.DataFrame({
                    'equity': [total_equity],
                    'daily_returns': [daily_return],
                    'position_value': [position_value],
                    'cash': [self.current_capital]
                }, index=[timestamp])
                
                self.daily_stats = pd.concat([self.daily_stats, new_stats])
            
            return total_equity
            
        except Exception as e:
            self.logger.error(f"计算权益��出错: {str(e)}")
            return self.current_capital  # 如果出错，返回当前资金

    def _update_position(self, trade):
        """更新持仓信息"""
        try:
            if 'symbol' not in trade:
                trade['symbol'] = 'default'
                
            symbol = trade['symbol']
            if symbol not in self.positions:
                self.positions[symbol] = Position(symbol=symbol)
            
            position = self.positions[symbol]
            trade_quantity = trade['quantity'] * (1 if trade['direction'] > 0 else -1)
            trade_value = abs(trade['value'])
            
            # 更新持仓
            if position.quantity * trade_quantity >= 0:  # 同向交易
                # 计算新的平均价格
                total_value = position.avg_price * abs(position.quantity) + trade_value
                total_quantity = abs(position.quantity) + abs(trade_quantity)
                position.avg_price = total_value / total_quantity if total_quantity > 0 else trade['price']
            else:  # 反向交易（平仓或反手）
                if abs(trade_quantity) >= abs(position.quantity):  # 完全平仓或反手
                    position.avg_price = trade['price']
                # 否则保持原有平均价格
            
            # 更新持仓数量
            position.quantity += trade_quantity
            
            # 更新资金
            commission = trade_value * self.commission_rate
            slippage = trade_value * self.slippage
            self.current_capital -= (trade_value + commission + slippage)
            
            # 记录交易
            self.trades.append(Trade(
                timestamp=trade['timestamp'],
                symbol=symbol,
                direction='buy' if trade['direction'] > 0 else 'sell',
                quantity=abs(trade_quantity),
                price=trade['price'],
                commission=commission,
                slippage=slippage
            ))
            
        except Exception as e:
            self.logger.error(f"更新持仓时出错: {str(e)}")
            raise

class PricePredictor:
    def __init__(self):
        self.analyzer = SupplyDemandAnalyzer()
        self.price_adjustment_rate = {
            '价格上涨趋势': 0.05,
            '价格下跌趋势': -0.05,
            '价格稳定': 0,
            '价格稳定偏强': 0.02,
            '价格稳定偏弱': -0.02
        }
    
    def predict_price_movement(self, current_price, supply_index, demand_index):
        """
        预测价格走势
        """
        market_status, price_trend = self.analyzer.get_market_status(supply_index, demand_index)
        
        # 基础价格调整
        adjustment = self.price_adjustment_rate[price_trend]
        
        # 计算预测价格
        predicted_price = current_price * (1 + adjustment)
        
        return {
            'predicted_price': predicted_price,
            'market_status': market_status,
            'price_trend': price_trend,
            'adjustment_rate': adjustment
        }
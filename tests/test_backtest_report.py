import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from visualization.chart_generator import ChartGenerator
from utils.logger import Logger

def generate_test_backtest_results():
    """生成测试用的回测结果数据"""
    # 生成日期范围
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(24*30)]  # 30天的小时数据
    
    # 生成每日统计数据
    daily_stats = pd.DataFrame({
        'timestamp': dates,
        'total_value': np.linspace(1000000, 1200000, len(dates)) + np.random.normal(0, 1000, len(dates)),
        'daily_returns': np.random.normal(0.001, 0.02, len(dates))
    })
    daily_stats.set_index('timestamp', inplace=True)
    
    # 生成交易记录
    trades = pd.DataFrame({
        'timestamp': dates[::4],  # 每4小时一笔交易
        'direction': np.random.choice(['buy', 'sell'], size=len(dates[::4])),
        'quantity': np.random.randint(1, 10, size=len(dates[::4])),
        'price': np.random.uniform(100, 120, size=len(dates[::4])),
        'pnl': np.random.normal(100, 500, size=len(dates[::4])),
        'commission': np.random.uniform(1, 5, size=len(dates[::4])),
        'slippage': np.random.uniform(0.1, 0.5, size=len(dates[::4]))
    })
    
    # 创建回测结果字典
    backtest_results = {
        'initial_capital': 1000000,
        'final_capital': 1200000,
        'total_returns': 0.2,
        'annual_returns': 0.4,
        'sharpe_ratio': 1.5,
        'max_drawdown': -0.1,
        'total_trades': len(trades),
        'trading_days': 30,
        'win_rate': 0.6,
        'max_position': 10.0,
        'margin': 100000.0,
        'daily_stats': daily_stats,
        'trades': trades
    }
    
    return backtest_results

def test_backtest_report():
    """测试回测报告生成"""
    logger = Logger().get_logger()
    chart_generator = ChartGenerator()
    
    try:
        # 生成测试数据
        logger.info("生成测试数据...")
        backtest_results = generate_test_backtest_results()
        
        # 生成回测报告
        logger.info("开始生成回测报告...")
        chart_generator.generate_backtest_reports(backtest_results)
        
        logger.info("回测报告生成完成")
        
    except Exception as e:
        logger.error(f"测试回测报告生成时出错: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    test_backtest_report() 
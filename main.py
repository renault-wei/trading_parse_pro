import sys
import pandas as pd
import numpy as np
from core.system import TradingSystem
from utils.logger import Logger
from core.strategy.intraday_strategy import IntradayStrategy
from core.strategy.factor_trainer import FactorTrainer
from visualization.workday_analysis import WorkdayAnalyzer
from core.analysis.regression import RegressionAnalyzer
from core.analysis.supply_demand import SupplyDemandAnalyzer
from visualization.supply_demand_charts import SupplyDemandCharts
from core.prediction.price_predictor import ElectricityPricePredictor
from visualization.prediction_charts import PredictionVisualizer
from core.missing_value_analyzer import MissingValueAnalyzer
from utils.pdf_exporter import PDFExporter
from visualization.chart_generator.supply_demand import SupplyDemandChartGenerator

def main():
    """主函数"""
    logger = Logger().get_logger()
    
    try:
        # 初始化系统
        system = TradingSystem()
        
        # 设置分析参数
        start_date = "2024-01-01"
        end_date = "2024-07-01"
        
        # 运行分析
        analysis_results = system.run_analysis(start_date, end_date)
        
        # 初始化供需分析图表生成器
        supply_demand_chart_generator = SupplyDemandChartGenerator(theme='plotly_white')
        
        # 获取处理后的数据
        processed_data = system.data_processor.get_processed_data()
        
        # 生成供需分析图表
        supply_demand_chart_generator.generate_supply_demand_analysis(processed_data)
        
        # 在处理数据后，分析缺失值
        missing_value_analyzer = MissingValueAnalyzer(processed_data)
        missing_value_analyzer.analyze_missing_values()
        
        # 生成 PDF 报告
        pdf_exporter = PDFExporter()
        html_content = pdf_exporter._dataframe_to_html(processed_data, "分析报告")
        pdf_exporter.export_to_pdf(html_content, "分析报告", "analysis_report")
        

        
    except Exception as e:
        logger.error(f"程序运行出错: {str(e)}", exc_info=True)
        sys.exit(1)
        
if __name__ == "__main__":
    main() 
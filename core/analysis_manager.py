import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from core.analyzer.seasonal_analyzer import SeasonalAnalyzer
from core.analyzer.volatility_analyzer import VolatilityAnalyzer
from core.analyzer.factor_analyzer import FactorAnalyzer
from core.analyzer.period_pattern_analyzer import PeriodPatternAnalyzer

class AnalysisManager:
    """分析模块管理器"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.analyzers = self._init_analyzers()
        
    def _init_analyzers(self):
        """初始化分析器"""
        return {
            'seasonal': SeasonalAnalyzer,
            'volatility': VolatilityAnalyzer,
            'factor': FactorAnalyzer,
            'period': PeriodPatternAnalyzer
        }
        
    def run_analysis(self, data: pd.DataFrame) -> dict:
        """运行所有分析"""
        try:
            return self._run_parallel_analysis(data)
        except Exception as e:
            self.logger.error(f"运行分析时出错: {str(e)}")
            raise
            
    def _run_parallel_analysis(self, data: pd.DataFrame) -> dict:
        """并行运行分析"""
        analysis_results = {}
        max_workers = int(self.config.get('PERFORMANCE', 'parallel_workers', 4))
        
        def run_analyzer(name: str, analyzer_class):
            try:
                analyzer = analyzer_class(data)
                analyzer.analyze()
                return name, analyzer.get_results()
            except Exception as e:
                self.logger.error(f"分析器 {name} 执行失败: {str(e)}")
                return name, None
                
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_analyzer = {
                executor.submit(run_analyzer, name, analyzer_class): name 
                for name, analyzer_class in self.analyzers.items()
            }
            
            for future in as_completed(future_to_analyzer):
                try:
                    name, results = future.result()
                    if results is not None:
                        analysis_results[name] = results
                        self.logger.info(f"完成 {name} 分析")
                except Exception as e:
                    self.logger.error(f"获取分析结果时出错: {str(e)}")
                    
        return analysis_results
        
    def validate_results(self, results: dict) -> bool:
        """验证分析结果"""
        try:
            # 检查必需的分析结果是否存在
            required_keys = set(self.analyzers.keys())
            missing_keys = required_keys - set(results.keys())
            
            if missing_keys:
                self.logger.error(f"缺少必需的分析结果: {', '.join(missing_keys)}")
                return False
                
            # 验证数值的合理性
            if 'volatility' in results:
                vol_results = results['volatility']
                if vol_results.get('historical_volatility', {}).get('daily', 0) > float(
                    self.config.get('VALIDATION', 'volatility_warning_threshold', 0.5)
                ):
                    self.logger.warning("日波动率异常高")
                    
            # 验证价格范围
            if 'price_min' in results:
                price_min = float(results['price_min'])
                price_max = float(results['price_max'])
                min_threshold = float(self.config.get('VALIDATION', 'price_min_threshold'))
                max_threshold = float(self.config.get('VALIDATION', 'price_max_threshold'))
                
                if price_min < min_threshold or price_max > max_threshold:
                    self.logger.warning(f"价格超出预期范围: [{price_min}, {price_max}]")
                    
            return True
            
        except Exception as e:
            self.logger.error(f"验证分析结果时出错: {str(e)}")
            return False
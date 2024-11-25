import pandas as pd
import numpy as np
from core.prediction.model_selector import ModelSelector
from core.prediction.enhanced_feature_engineering import EnhancedFeatureEngineer
from core.prediction.enhanced_predictor import EnhancedPredictor
from core.monitoring.model_monitor import ModelMonitor

def test_prediction_pipeline():
    # 1. 准备测试数据
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='H')
    test_data = pd.DataFrame({
        'price': np.random.normal(100, 10, len(dates)),
        'demand': np.random.normal(1000, 100, len(dates))
    }, index=dates)
    
    try:
        # 2. 特征工程测试
        feature_engineer = EnhancedFeatureEngineer()
        features = feature_engineer.create_features(test_data)
        print("特征工程完成，生成的特征:", features.columns.tolist())
        
        # 3. 模型选择测试
        X = features.iloc[168:].values  # 去掉前7天的数据（因为有滞后特征）
        y = test_data['price'].iloc[168:].values
        
        model_selector = ModelSelector()
        best_model, model_name = model_selector.select_best_model(X, y)
        print(f"选择的最佳模型: {model_name}")
        
        # 4. 增强预测测试
        enhanced_predictor = EnhancedPredictor(best_model)
        prediction_results = enhanced_predictor.predict_with_intervals(X[:24])  # 预测一天的数据
        print("预测结果示例:")
        print("- 预测值:", prediction_results['prediction'][:5])
        print("- 下界:", prediction_results['lower_bound'][:5])
        print("- 上界:", prediction_results['upper_bound'][:5])
        
        # 5. 模型监控测试
        monitor = ModelMonitor(best_model)
        needs_retrain = monitor.monitor_performance(y[:100], prediction_results['prediction'][:100])
        
        monitoring_report = monitor.generate_monitoring_report()
        print("\n监控报告:")
        print("- 当前RMSE:", monitoring_report['current_performance']['rmse'])
        print("- 警报:", monitoring_report['alerts'])
        
        return True, "测试完成"
        
    except Exception as e:
        return False, f"测试失败: {str(e)}"

if __name__ == "__main__":
    success, message = test_prediction_pipeline()
    print(f"\n测试结果: {'成功' if success else '失败'}")
    print(f"详细信息: {message}") 
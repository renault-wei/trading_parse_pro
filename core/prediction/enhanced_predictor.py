import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats

class EnhancedPredictor:
    def __init__(self, base_model):
        self.base_model = base_model
        self.confidence_level = 0.95
        
    def predict_with_intervals(self, X):
        """带置信区间的预测"""
        # 基础预测
        y_pred = self.base_model.predict(X)
        
        # 使用bootstrap方法估计预测区间
        predictions = []
        n_iterations = 100
        
        for _ in range(n_iterations):
            # 随机采样训练数据
            indices = np.random.randint(0, len(X), size=len(X))
            X_sample = X[indices]
            
            # 预测
            pred = self.base_model.predict(X_sample)
            predictions.append(pred)
            
        # 计算置信区间
        predictions = np.array(predictions)
        lower = np.percentile(predictions, ((1 - self.confidence_level) / 2) * 100, axis=0)
        upper = np.percentile(predictions, (1 - (1 - self.confidence_level) / 2) * 100, axis=0)
        
        return {
            'prediction': y_pred,
            'lower_bound': lower,
            'upper_bound': upper
        }
        
    def predict_by_period(self, X):
        """分时段预测"""
        predictions = {}
        
        # 按小时分组预测
        for hour in range(24):
            hour_mask = X.index.hour == hour
            if hour_mask.any():
                X_hour = X[hour_mask]
                predictions[hour] = self.predict_with_intervals(X_hour)
                
        return predictions 
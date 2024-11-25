import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class ModelMonitor:
    def __init__(self, model, threshold_rmse=10, threshold_drift=0.1):
        self.model = model
        self.threshold_rmse = threshold_rmse
        self.threshold_drift = threshold_drift
        self.performance_history = []
        self.last_retrain_date = None
        
    def monitor_performance(self, y_true, y_pred):
        """监控模型性能"""
        current_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        current_date = datetime.now()
        
        performance_record = {
            'date': current_date,
            'rmse': current_rmse,
            'sample_size': len(y_true),
            'mean_prediction': np.mean(y_pred),
            'mean_actual': np.mean(y_true)
        }
        
        self.performance_history.append(performance_record)
        
        # 检查是否需要重训练
        if self._check_retrain_needed():
            return True
            
        return False
        
    def _check_retrain_needed(self):
        """检查是否需要重训练模型"""
        if len(self.performance_history) < 2:
            return False
            
        # 检查性能下降
        recent_rmse = self.performance_history[-1]['rmse']
        if recent_rmse > self.threshold_rmse:
            return True
            
        # 检查预测偏移
        recent_mean = self.performance_history[-1]['mean_prediction']
        previous_mean = self.performance_history[-2]['mean_prediction']
        drift = abs(recent_mean - previous_mean) / previous_mean
        
        if drift > self.threshold_drift:
            return True
            
        # 检查上次重训练时间
        if self.last_retrain_date:
            days_since_retrain = (datetime.now() - self.last_retrain_date).days
            if days_since_retrain > 30:  # 每30天强制重训练
                return True
                
        return False
        
    def generate_monitoring_report(self):
        """生成监控报告"""
        df = pd.DataFrame(self.performance_history)
        
        report = {
            'current_performance': {
                'rmse': df.iloc[-1]['rmse'],
                'date': df.iloc[-1]['date']
            },
            'performance_trend': {
                'rmse_trend': df['rmse'].tolist(),
                'dates': df['date'].tolist()
            },
            'alerts': self._generate_alerts(df)
        }
        
        return report
        
    def _generate_alerts(self, df):
        """生成警报信息"""
        alerts = []
        
        # 检查性能突变
        if len(df) >= 2:
            last_rmse = df.iloc[-1]['rmse']
            prev_rmse = df.iloc[-2]['rmse']
            change = (last_rmse - prev_rmse) / prev_rmse
            
            if change > 0.2:  # 性能下降超过20%
                alerts.append({
                    'type': 'performance_degradation',
                    'message': f'Performance degraded by {change*100:.2f}%'
                })
                
        return alerts 
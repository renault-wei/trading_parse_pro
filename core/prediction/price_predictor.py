import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils.logger import Logger

class ElectricityPricePredictor:
    """电力价格预测系统"""
    
    def __init__(self, config=None):
        self.logger = Logger().get_logger()
        # 优化RandomForestRegressor的参数
        self.model = RandomForestRegressor(
            n_estimators=200,          # 增加树的数量以提高稳定性
            max_depth=8,               # 适当增加深度
            min_samples_split=5,       # 控制过拟合
            min_samples_leaf=3,        # 确保叶子节点有足够样本
            max_features='sqrt',       # 特征采样
            n_jobs=-1,                 # 使用所有CPU核心
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def prepare_features(self, data):
        """增强特征工程"""
        self.logger.info("准备特征...")
        features = pd.DataFrame(index=data.index)
        
        # 1. 基础时间特征
        features['is_workday'] = data.index.dayofweek.isin([0,1,2,3,4]).astype(int)
        features['hour_sin'] = np.sin(2 * np.pi * data.index.hour/24)
        features['hour_cos'] = np.cos(2 * np.pi * data.index.hour/24)
        
        # 2. 时段特征
        features['is_morning_peak'] = data.index.hour.isin([6,7,8,9]).astype(int)
        features['is_evening_peak'] = data.index.hour.isin([16,17,18,19,20,21]).astype(int)
        features['is_valley'] = data.index.hour.isin([10,11,12,13,14,15]).astype(int)
        
        # 3. 价格统计特征（如果有历史价格）
        if 'price' in data.columns:
            # 移动平均
            features['price_ma24'] = data['price'].rolling(window=24, min_periods=1).mean()
            # 价格变化率
            features['price_change'] = data['price'].pct_change().fillna(0)
            # 价格波动率
            features['price_volatility'] = data['price'].rolling(window=24, min_periods=1).std()
        
        # 4. 需求特征（如果有）
        if 'demand' in data.columns:
            features['demand'] = data['demand']
            features['demand_ma24'] = data['demand'].rolling(window=24, min_periods=1).mean()
        
        # 填充缺失值
        features = features.fillna(method='ffill').fillna(method='bfill')
        
        self.logger.info(f"特征准备完成，特征列: {features.columns.tolist()}")
        return features
    
    def train(self, train_data):
        """训练模型"""
        self.logger.info("开始训练模型...")
        try:
            # 准备特征
            X = self.prepare_features(train_data)
            y = train_data['price']
            
            # 保存特征名称
            self.feature_names = X.columns.tolist()
            
            # 标准化
            X_scaled = self.scaler.fit_transform(X)
            
            # 训练模型
            self.model.fit(X_scaled, y)
            
            # 返回训练评估指标
            train_pred = self.model.predict(X_scaled)
            metrics = self.evaluate_predictions(y, train_pred)
            
            self.logger.info(f"模型训练完成，评估指标: {metrics}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"模型训练失败: {str(e)}")
            return None
    
    def predict(self, data):
        """预测价格"""
        try:
            # 准备特征
            X = self.prepare_features(data)
            X = X[self.feature_names]  # 确保特征顺序一致
            
            # 标准化
            X_scaled = self.scaler.transform(X)
            
            # 预测
            predictions = self.model.predict(X_scaled)
            
            return pd.Series(predictions, index=data.index)
            
        except Exception as e:
            self.logger.error(f"预测失败: {str(e)}")
            return None
            
    def evaluate_predictions(self, y_true, y_pred):
        """详细评估预测结果"""
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
        
        # 添加分时段评估
        hour_metrics = {}
        for hour in range(24):
            hour_mask = y_true.index.hour == hour
            if hour_mask.any():
                hour_metrics[hour] = {
                    'rmse': np.sqrt(mean_squared_error(y_true[hour_mask], y_pred[hour_mask])),
                    'mae': mean_absolute_error(y_true[hour_mask], y_pred[hour_mask])
                }
        
        metrics['hour_metrics'] = hour_metrics
        return metrics
            
    def get_feature_importance(self):
        """获取特征重要性"""
        if self.model is None or self.feature_names is None:
            return None
            
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        })
        return importance_df.sort_values('importance', ascending=False) 
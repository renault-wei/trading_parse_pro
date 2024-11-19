"""回归分析模块"""
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Dict, Any

class RegressionAnalyzer:
    """回归分析器"""
    def __init__(self, logger):
        self.logger = logger
        
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        执行回归分析
        
        Args:
            data: 包含交易数据的DataFrame
            
        Returns:
            Dict: 包含回归分析结果的字典
        """
        try:
            # 准备特征
            features = self._prepare_features(data)
            
            # 准备目标变量
            y = data['price']
            
            # 分割训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(
                features, y, test_size=0.2, random_state=42
            )
            
            # 标准化特征
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 训练模型
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
            
            # 生成预测值
            y_pred = model.predict(X_test_scaled)
            
            # 计算残差
            residuals = y_test - y_pred
            
            # 计算特征重要性
            feature_importance = pd.Series(dict(zip(features.columns, np.abs(model.coef_))))
            
            # 计算R方值
            r2_score = model.score(X_test_scaled, y_test)
            
            # 返回结果
            return {
                'feature_importance': feature_importance,
                'y_true': y_test.values,  # 确保返回numpy数组
                'y_pred': y_pred,
                'residuals': residuals,
                'r2_score': r2_score,
                'model': model,
                'scaler': scaler,
                'feature_names': features.columns.tolist()
            }
            
        except Exception as e:
            print(f"回归分析失败: {str(e)}")
            # 返回空结果但包含所有必要的键
            return {
                'feature_importance': {},
                'y_true': np.array([]),
                'y_pred': np.array([]),
                'residuals': np.array([]),
                'r2_score': 0.0,
                'model': None,
                'scaler': None,
                'feature_names': []
            }
            
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """准备特征数据"""
        features = pd.DataFrame()
        
        # 时间特征
        if 'trade_hour' in data.columns:
            features['hour'] = data['trade_hour']
            features['hour_sin'] = np.sin(2 * np.pi * data['trade_hour'] / 24)
            features['hour_cos'] = np.cos(2 * np.pi * data['trade_hour'] / 24)
            
        if 'date' in data.columns:
            features['weekday'] = pd.to_datetime(data['date']).dt.dayofweek
            features['month'] = pd.to_datetime(data['date']).dt.month
            
        # 价格特征
        if 'price' in data.columns:
            features['price_ma24'] = data['price'].rolling(window=24, min_periods=1).mean()
            features['price_std24'] = data['price'].rolling(window=24, min_periods=1).std()
            features['price_volatility'] = features['price_std24'] / features['price_ma24']
            
        # 供需特征
        if 'supply_pressure' in data.columns:
            features['supply_pressure'] = data['supply_pressure']
        if 'demand_pressure' in data.columns:
            features['demand_pressure'] = data['demand_pressure']
            
        # 填充缺失值
        features = features.fillna(method='ffill').fillna(method='bfill')
        
        return features
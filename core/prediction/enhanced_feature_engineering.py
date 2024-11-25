import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class EnhancedFeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def create_features(self, df):
        """增强版特征工程"""
        features = pd.DataFrame(index=df.index)
        
        # 基础时间特征
        features = self._add_time_features(features)
        
        # 价格特征
        if 'price' in df.columns:
            features = self._add_price_features(df, features)
            
        # 滞后特征
        features = self._add_lag_features(df, features)
        
        # 统计特征
        features = self._add_statistical_features(df, features)
        
        # 节假日特征
        features = self._add_holiday_features(features)
        
        return features
        
    def _add_time_features(self, df):
        """添加时间特征"""
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year
        
        # 周期性编码
        df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        
        return df
        
    def _add_price_features(self, source_df, df):
        """添加价格相关特征"""
        # 移动平均
        windows = [24, 48, 168]  # 1天, 2天, 1周
        for window in windows:
            df[f'price_ma_{window}'] = source_df['price'].rolling(window=window).mean()
            df[f'price_std_{window}'] = source_df['price'].rolling(window=window).std()
            
        # 价格变化率
        df['price_change'] = source_df['price'].pct_change()
        df['price_change_24h'] = source_df['price'].pct_change(24)
        
        return df
        
    def _add_lag_features(self, source_df, df):
        """添加滞后特征"""
        lags = [1, 24, 48, 168, 336]  # 1小时, 1天, 2天, 1周, 2周
        for lag in lags:
            df[f'price_lag_{lag}'] = source_df['price'].shift(lag)
            
        return df
        
    def _add_statistical_features(self, source_df, df):
        """添加统计特征"""
        # 同时段历史统计
        df['hour_mean'] = source_df.groupby(source_df.index.hour)['price'].transform('mean')
        df['hour_std'] = source_df.groupby(source_df.index.hour)['price'].transform('std')
        
        # 周内同时段统计
        df['weekday_hour_mean'] = source_df.groupby([source_df.index.dayofweek, 
                                                    source_df.index.hour])['price'].transform('mean')
        
        return df
        
    def _add_holiday_features(self, df):
        """添加节假日特征"""
        # 这里需要导入节假日数据
        # df['is_holiday'] = df.index.map(lambda x: x in holidays)
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        return df 
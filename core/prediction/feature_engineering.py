class FeatureEngineer:
    def create_time_features(self, df):
        """创建时间特征"""
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        return df
        
    def create_lag_features(self, df, lags=[1, 24, 48, 168]):
        """创建滞后特征"""
        for lag in lags:
            df[f'price_lag_{lag}'] = df['price'].shift(lag)
        return df 
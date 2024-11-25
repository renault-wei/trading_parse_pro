import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class ModelSelector:
    def __init__(self):
        self.models = {
            'lgb': self._create_lgb_model(),
            'xgb': self._create_xgb_model(),
            'catboost': self._create_catboost_model(),
            'lstm': self._create_lstm_model()
        }
        
    def _create_lgb_model(self):
        return LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=8,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8
        )
        
    def _create_xgb_model(self):
        return XGBRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8
        )
        
    def _create_catboost_model(self):
        return CatBoostRegressor(
            iterations=1000,
            learning_rate=0.01,
            depth=8,
            verbose=False
        )
        
    def _create_lstm_model(self):
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(24, 1)),
            Dropout(0.2),
            LSTM(64),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
        
    def select_best_model(self, X, y, cv=5):
        """使用时间序列交叉验证选择最佳模型"""
        tscv = TimeSeriesSplit(n_splits=cv)
        model_scores = {}
        
        for name, model in self.models.items():
            scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                if name == 'lstm':
                    # LSTM需要特殊处理
                    X_train = X_train.reshape((-1, 24, 1))
                    X_val = X_val.reshape((-1, 24, 1))
                    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
                else:
                    model.fit(X_train, y_train)
                
                y_pred = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                scores.append(rmse)
                
            model_scores[name] = np.mean(scores)
            
        best_model_name = min(model_scores, key=model_scores.get)
        return self.models[best_model_name], best_model_name 
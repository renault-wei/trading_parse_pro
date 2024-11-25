class EnsemblePredictor:
    def __init__(self):
        self.base_models = [
            ('lgb', LGBMRegressor()),
            ('xgb', XGBRegressor()),
            ('cat', CatBoostRegressor())
        ]
        self.meta_model = LinearRegression()
        
    def fit(self, X, y):
        """训练集成模型"""
        self.base_predictions = np.column_stack([
            cross_val_predict(model, X, y, cv=5)
            for name, model in self.base_models
        ])
        self.meta_model.fit(self.base_predictions, y) 
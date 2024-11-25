class ModelEvaluator:
    def evaluate_model(self, y_true, y_pred):
        metrics = {
            'mape': mean_absolute_percentage_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
        return metrics 
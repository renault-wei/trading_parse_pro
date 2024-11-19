import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error
import optuna
import logging

logger = logging.getLogger(__name__)

class FactorTrainer:
    """因子训练和参数优化类"""
    
    def __init__(self, n_trials=100):
        self.n_trials = n_trials
        self.best_params = None
        self.best_score = None
        self.logger = logger
        
    def optimize_parameters(self, data):
        """使用 Optuna 优化因子参数"""
        try:
            def objective(trial):
                # 定义参数搜索空间
                params = {
                    'lookback_window': trial.suggest_int('lookback_window', 12, 48),
                    'workday_threshold': trial.suggest_float('workday_threshold', 0.3, 0.8),
                    'volatility_threshold': trial.suggest_float('volatility_threshold', 0.5, 2.0),
                    'momentum_window': trial.suggest_int('momentum_window', 12, 36)
                }
                
                # 使用这些参数计算因子
                factors = self._calculate_factors_with_params(data, params)
                
                # 计算因子的预测能力
                score = self._evaluate_factors(data, factors)
                
                return score
                
            # 创建和运行优化研究
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=self.n_trials)
            
            # 保存最佳参数
            self.best_params = study.best_params
            self.best_score = study.best_value
            
            logger.info(f"最佳参数: {self.best_params}")
            logger.info(f"最佳得分: {self.best_score:.4f}")
            
            return self.best_params
            
        except Exception as e:
            logger.error(f"参数优化过程出错: {str(e)}")
            return None
            
    def _calculate_factors_with_params(self, data, params):
        """使用给定参数计算因子"""
        try:
            # 检查是否已经计算过因子
            required_factors = ['workday_factor', 'nonworkday_factor', 'hour_factor', 'momentum']
            if all(factor in data.columns for factor in required_factors):
                return data
            
            # 创建副本并确保索引唯一
            df = data.copy()
            if df.index.duplicated().any():
                logger.warning("发现重复的时间索引，保留最后一个值")
                df = df[~df.index.duplicated(keep='last')]
            
            # 确保数据按时间索引排序
            df = df.sort_index()
            
            # 确保索引是时间戳类型
            if not isinstance(df.index, pd.DatetimeIndex):
                logger.info("转换索引为时间戳类型")
                df.index = pd.to_datetime(df.index)
            
            # 工作日和非工作日收益率
            df['is_workday'] = df.index.weekday < 5
            df['returns'] = df['log_price'].diff()
            
            # 分别计算工作日和非工作日的收益率
            workday_mask = df['is_workday']
            
            # 使用参数化的窗口计算波动率
            df['workday_volatility'] = df['returns'].where(workday_mask).rolling(
                window=params['lookback_window'],
                min_periods=1
            ).std().fillna(method='ffill')
            
            df['nonworkday_volatility'] = df['returns'].where(~workday_mask).rolling(
                window=params['lookback_window'],
                min_periods=1
            ).std().fillna(method='ffill')
            
            # 计算工作日因子
            df['workday_factor'] = np.where(
                workday_mask,
                df['returns'] / (df['workday_volatility'] + 1e-10),
                0
            )
            
            # 计算非工作日因子
            df['nonworkday_factor'] = np.where(
                ~workday_mask,
                df['returns'] / (df['nonworkday_volatility'] + 1e-10),
                0
            )
            
            # 计算小时因子 - 修改这部分代码
            df['hour'] = df.index.hour
            
            # 分别计算每个小时的统计数据
            hour_returns = df.groupby('hour')['returns'].mean()
            hour_volatility = df.groupby('hour')['returns'].std()
            
            # 创建小时统计数据框
            hourly_stats = pd.DataFrame({
                'hour': hour_returns.index,
                'hour_return': hour_returns.values,
                'hour_volatility': hour_volatility.values
            })
            
            # 确保小时统计数据中没有空值
            hourly_stats['hour_volatility'] = hourly_stats['hour_volatility'].fillna(
                hourly_stats['hour_volatility'].mean()
            )
            hourly_stats['hour_return'] = hourly_stats['hour_return'].fillna(0)
            
            # 使用merge合并数据
            df = pd.merge(
                df,
                hourly_stats,
                on='hour',
                how='left'
            )
            
            # 计算小时因子
            df['hour_factor'] = df['hour_return'] / (df['hour_volatility'] + 1e-10)
            
            # 添加动量因子
            df['momentum'] = df['log_price'].diff(params['momentum_window'])
            
            # 标准化因子
            factors_to_standardize = ['workday_factor', 'nonworkday_factor', 'hour_factor', 'momentum']
            for factor in factors_to_standardize:
                if factor in df.columns:
                    # 处理无穷大和NaN值
                    df[factor] = df[factor].replace([np.inf, -np.inf], np.nan)
                    # 计算均值和标准差（忽略NaN值）
                    mean = df[factor].mean()
                    std = df[factor].std()
                    if std > 0:
                        df[factor] = (df[factor] - mean) / std
                    else:
                        df[factor] = 0
                    # 填充剩余的NaN值
                    df[factor] = df[factor].fillna(0)
            
            # 添加调试信息（仅在首次计算时输出）
            if not hasattr(self, '_factors_calculated'):
                logger.info("因子计算完成")
                for factor in factors_to_standardize:
                    if factor in df.columns:
                        logger.info(f"{factor} 范围: {df[factor].min():.3f} 到 {df[factor].max():.3f}")
                self._factors_calculated = True
            
            return df
            
        except Exception as e:
            logger.error(f"因子计算出错: {str(e)}")
            # 返回原始数据而不是None，这样后续处理仍然可以继续
            return data
            
    def _evaluate_factors(self, data, factors):
        """评估因子的预测能力"""
        try:
            if factors is None:
                return float('-inf')
                
            # 计算未来收益率
            future_returns = data['log_price'].diff().shift(-1)
            
            # 创建因子组合
            X = factors[['workday_factor', 'nonworkday_factor', 'hour_factor', 'momentum']].fillna(0)
            y = future_returns
            
            # 使用时间序列交叉验证
            tscv = TimeSeriesSplit(n_splits=5)
            scores = []
            
            for train_idx, test_idx in tscv.split(X):
                X_train = X.iloc[train_idx]
                X_test = X.iloc[test_idx]
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]
                
                # 计算每个因子与未来收益的相关性
                correlations = []
                for factor in X_train.columns:
                    corr = np.corrcoef(X_train[factor], y_train.shift(-1).fillna(0))[0, 1]
                    correlations.append(abs(corr))
                
                # 使用相关性作为权重
                weights = np.array(correlations) / sum(correlations)
                
                # 计算加权组合因子
                pred_train = (X_train * weights).sum(axis=1)
                pred_test = (X_test * weights).sum(axis=1)
                
                # 计算预测得分
                score = r2_score(y_test.fillna(0), pred_test)
                scores.append(score)
            
            return np.mean(scores)
            
        except Exception as e:
            logger.error(f"因子评估出错: {str(e)}")
            return float('-inf')
            
    def train_and_save_model(self, data, save_path):
        """训练并保存模型"""
        try:
            # 优化参数
            best_params = self.optimize_parameters(data)
            
            if best_params is None:
                return None
                
            # 使用最佳参数计算因子
            final_factors = self._calculate_factors_with_params(data, best_params)
            
            # 保存模型参数和因子
            model_data = {
                'parameters': best_params,
                'performance': self.best_score,
                'factors': final_factors
            }
            
            # 保存到文件
            pd.to_pickle(model_data, save_path)
            logger.info(f"模型已保存到: {save_path}")
            
            return model_data
            
        except Exception as e:
            logger.error(f"模型训练和保存过程出错: {str(e)}")
            return None 
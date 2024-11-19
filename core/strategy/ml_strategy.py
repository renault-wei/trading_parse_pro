import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from utils.logger import Logger

class MLStrategy:
    def __init__(self):
        # 初始化logger
        self.logger = Logger().get_logger()
        
        # 确保特征列表与训练模型时使用的特征完全一致
        self.features = [
            'is_workday',
            'hour_sin',
            'hour_cos',
            'is_morning_peak',
            'is_evening_peak',
            'is_valley',
            'price_ma24',
            'price_change',
            'price_volatility'
        ]  # 这9个特征需要与训练模型时使用的特征完全一致
        
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        self.scaler = StandardScaler()
        
        # 调整交易参数
        self.min_holding_period = 24    # 最小持仓时间
        self.entry_threshold = 0.25     # 进场阈值
        self.position_size = 0.2        # 仓位大小
        
        self.logger.info(f"MLStrategy 初始化完成，使用特征: {self.features}")
        
    def update_model(self, new_model):
        """更新策略使用的预测模型"""
        try:
            self.model = new_model
            self.logger.info("策略模型已更新")
            
            # 获取模型特征重要性（如果是随机森林模型）
            if hasattr(self.model, 'feature_importances_'):
                importances = pd.Series(
                    self.model.feature_importances_,
                    index=self.features
                ).sort_values(ascending=False)
                
                self.logger.info("\n特征重要性:")
                for feature, importance in importances.items():
                    self.logger.info(f"{feature}: {importance:.4f}")
                    
            return True
            
        except Exception as e:
            self.logger.error(f"更新模型时出错: {str(e)}")
            return False
            
    def calculate_price_patterns(self, data):
        """计算电力价格特征模式"""
        try:
            patterns = pd.DataFrame(index=data.index)
            
            # 计算同时段历史价格平均值和标准差
            patterns['hour_avg'] = data.groupby('trade_hour')['price'].transform('mean')
            patterns['hour_std'] = data.groupby('trade_hour')['price'].transform('std')
            
            # 计算价格偏离度
            patterns['price_deviation'] = (data['price'] - patterns['hour_avg']) / patterns['hour_std']
            
            self.logger.info("价格模式计算完成")
            return patterns
            
        except Exception as e:
            self.logger.error(f"计算价格模式时出错: {str(e)}")
            return None
            
    def generate_signals(self, data):
        """生成交易信号"""
        try:
            # 计算价格模式
            price_patterns = self.calculate_price_patterns(data)
            if price_patterns is None:
                raise ValueError("价格模式计算失败")
            
            # 准备特征数据
            features = self.prepare_features(data)
            
            # 首先对所有数据进行fit
            self.scaler.fit(features)
            
            # 生成信号
            signals = pd.Series(index=data.index, data=0)
            current_position = 0
            
            for i, (idx, row) in enumerate(data.iterrows()):
                # 检查是否是合适的交易时机
                is_suitable_time = (
                    not row['is_morning_peak'] and  # 避开早高峰
                    not row['is_evening_peak'] and  # 避开晚高峰
                    abs(price_patterns.loc[idx, 'price_deviation']) < 1.5  # 价格不过分偏离
                )
                
                if not is_suitable_time:
                    signals[idx] = current_position
                    continue
                
                # 准备当前时点的特征
                current_features = features.loc[idx:idx]
                if len(current_features) > 0:
                    # 使用已经fit过的scaler进行transform
                    X_scaled = self.scaler.transform(current_features)
                    pred = self.model.predict(X_scaled)[0]
                    
                    # 生成交易信号
                    if pred > self.entry_threshold and current_position <= 0:
                        signals[idx] = self.position_size
                        current_position = 1
                    elif pred < -self.entry_threshold and current_position >= 0:
                        signals[idx] = -self.position_size
                        current_position = -1
                    else:
                        signals[idx] = current_position
            
            # 计算收益
            price_returns = data['price'].pct_change()
            returns = signals.shift(1) * price_returns
            
            self.logger.info("交易信号生成完成")
            return {
                'signals': signals,
                'returns': returns,
                'positions': signals.copy(),
                'cumulative_returns': (1 + returns).cumprod() - 1,
                'drawdown': (1 + returns).cumprod() / (1 + returns).cumprod().expanding().max() - 1,
                'total_returns': returns.sum(),
                'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252),
                'max_drawdown': ((1 + returns).cumprod() / (1 + returns).cumprod().expanding().max() - 1).min(),
                'total_trades': (signals.diff() != 0).sum() // 2
            }
            
        except Exception as e:
            self.logger.error(f"生成交易信号时出错: {str(e)}")
            raise
        
    def prepare_features(self, data):
        """准备特征数据"""
        try:
            # 首先验证输入数据
            if data is None or len(data) == 0:
                raise ValueError("输入数据为空")
            
            required_columns = ['trade_hour', 'price']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"输入数据缺少必要列: {missing_columns}")

            # 创建特征DataFrame
            features = pd.DataFrame(index=data.index)
            
            # 基础时间特征
            features['trade_hour'] = data['trade_hour']
            features['hour_sin'] = np.sin(2 * np.pi * data['trade_hour'] / 24)
            features['hour_cos'] = np.cos(2 * np.pi * data['trade_hour'] / 24)
            
            # 价格特征
            features['price_change'] = data['price'].pct_change()
            features['price_ma24'] = data['price'].rolling(window=24, min_periods=1).mean()
            features['price_volatility'] = data['price'].rolling(window=24, min_periods=1).std()
            
            # 工作日特征 - 添加默认值处理
            features['is_workday'] = data.get('is_workday', 1)  # 如果没有is_workday列，默认为工作日
            
            # 时段特征 - 添加默认值处理
            features['is_morning_peak'] = data.get('is_morning_peak', 0)
            features['is_evening_peak'] = data.get('is_evening_peak', 0)
            features['is_valley'] = data.get('is_valley', 0)
            
            # 供需特征（可选）
            if 'supply_pressure' in data.columns and 'demand_pressure' in data.columns:
                features['supply_pressure'] = data['supply_pressure']
                features['demand_pressure'] = data['demand_pressure']
            
            # 填充缺失值
            features = features.fillna(method='ffill').fillna(0)  # 改用0填充
            
            # 检查特征完整性并处理
            for feature in self.features:
                if feature not in features.columns:
                    self.logger.warning(f"缺少特征 {feature}，使用默认值0填充")
                    features[feature] = 0
            
            # 只选择策略需要的特征
            selected_features = features[self.features]
            
            # 验证最终特征数据
            if selected_features.isnull().any().any():
                self.logger.warning("特征数据中存在空值，已用0填充")
                selected_features = selected_features.fillna(0)
            
            self.logger.info(f"特征准备完成，特征维度: {selected_features.shape}")
            return selected_features
            
        except Exception as e:
            self.logger.error(f"准备特征时出错: {str(e)}")
            self.logger.error(f"输入数据列: {data.columns.tolist() if data is not None and hasattr(data, 'columns') else 'None'}")
            raise
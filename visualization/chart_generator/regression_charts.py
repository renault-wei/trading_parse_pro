import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from scipy import stats
from .base import BaseChartGenerator
from typing import Dict
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from utils.pdf_exporter import PDFExporter

class RegressionChartGenerator(BaseChartGenerator):
    """回归分析图表生成器"""
    
    def generate_regression_analysis(self, data: pd.DataFrame):
        """生成回归分析图表"""
        try:
            # 计算相关系数
            feature_cols = [col for col in data.columns if col != 'price']
            
            # 打印 feature_cols 的内容和数据类型
            self.logger.info("Feature Columns: %s", feature_cols)
            self.logger.info("Data Types:\n%s", data[feature_cols].dtypes)
            
            # 过滤掉非数值类型的列
            feature_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(data[col])]
            
            # 记录过滤后的特征列
            self.logger.info("Filtered Feature Columns: %s", feature_cols)
            
            correlations = data[feature_cols].corrwith(data['price']).sort_values(ascending=True)
            
            # 1. 回归系数分析
            fig_coef = self._create_coefficient_plot(correlations, feature_cols)
            
            # 保存图表为 HTML
            output_path_html = Path(self.output_dirs['regression']) / 'price_factors_regression.html'
            fig_coef.write_html(str(output_path_html))
            
            # 将 HTML 内容导出为 PDF
            pdf_exporter = PDFExporter()
            html_content = fig_coef.to_html()  # 获取图表的 HTML 内容
            
            # 检查 HTML 内容
            if html_content is None or not isinstance(html_content, str):
                self.logger.error("生成的 HTML 内容无效，无法导出为 PDF。")
                raise ValueError("生成的 HTML 内容无效，无法导出为 PDF。")
            
            pdf_exporter.export_to_pdf(html_content, "回归分析图表", "price_factors_regression")
            
            # 2. 回归诊断图
            fig_diag = self._create_diagnostic_plots(data, feature_cols)
            diag_path = Path(self.output_dirs['regression']) / 'regression_diagnostics.html'
            fig_diag.write_html(str(diag_path))
            
            # 3. 特征重要性分析
            fig_imp = self._create_importance_plot(data, feature_cols)
            imp_path = Path(self.output_dirs['regression']) / 'feature_importance.html'
            fig_imp.write_html(str(imp_path))
            
            # 生成图表
           # fig = self._create_coefficient_plot(correlations, feature_cols)
            
            # 保存图表为 HTML
         #   output_path = Path(self.output_dirs['regression']) / 'regression_analysis.html'
          #  fig.write_html(str(output_path))
            
          #  print(f"图表已保存为静态 HTML: {output_path}")
            
            return {
                'coefficient_plot': str(output_path_html),
                'diagnostic_plot': str(diag_path),
                'importance_plot': str(imp_path)
            }
            
        except Exception as e:
            self.logger.error(f"生成回归分析图表时出错: {str(e)}")
            raise  # 确保这里有代码
    
    def _create_coefficient_plot(self, correlations: pd.Series, feature_cols: list) -> go.Figure:
        """创建回归系数图"""
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=correlations.values,
            y=correlations.index,
            orientation='h',
            name='回归系数'
        ))
        
        fig.update_layout(
            title='价格影响因素回归分析',
            xaxis_title='相关系数',
            yaxis_title='特征',
            height=max(400, len(feature_cols) * 30),
            template=self.theme
        )
        
        return fig
        
    def _create_diagnostic_plots(self, data: pd.DataFrame, feature_cols: list) -> go.Figure:
        """创建回归诊断图"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '残差分布',
                '实际值vs预测值',
                '残差vs预测值',
                'Q-Q图'
            )
        )
        
        # 拟合线性回归模型
        X = data[feature_cols]
        y = data['price']
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        residuals = y - y_pred
        
        # 残差分布
        fig.add_trace(
            go.Histogram(x=residuals, name='残差分'),
            row=1, col=1
        )
        
        # 实际值vs预测值
        fig.add_trace(
            go.Scatter(
                x=y, y=y_pred,
                mode='markers',
                name='实际vs预测'
            ),
            row=1, col=2
        )
        
        # 残差vs预测值
        fig.add_trace(
            go.Scatter(
                x=y_pred,
                y=residuals,
                mode='markers',
                name='残差vs预测'
            ),
            row=2, col=1
        )
        
        # Q-Q图
        qq = stats.probplot(residuals)
        fig.add_trace(
            go.Scatter(
                x=qq[0][0],
                y=qq[0][1],
                mode='markers',
                name='Q-Q图'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text='回归诊断分析',
            showlegend=True,
            template=self.theme
        )
        
        return fig
        
    def _create_importance_plot(self, data: pd.DataFrame, feature_cols: list) -> go.Figure:
        """创建特征重要性图"""
        # 拟合模型获取特征重要性
        X = data[feature_cols]
        y = data['price']
        
        # 使用 HistGradientBoostingRegressor
        model = HistGradientBoostingRegressor()
        self.logger.info("开始训练模型...")
        model.fit(X, y)
        
        # 确保模型训练成功
        self.logger.info("模型训练完成")
        
        # 使用 permutation_importance 计算特征重要性
        result = permutation_importance(model, X, y, n_repeats=30, random_state=42)
        
        # 获取特征重要性
        importance = pd.Series(result.importances_mean, index=feature_cols).sort_values(ascending=True)
        
        self.logger.info("特征重要性: %s", importance)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=importance.values,
            y=importance.index,
            orientation='h',
            name='特征重要性'
        ))
        
        fig.update_layout(
            title='特征重要性分析',
            xaxis_title='重要性系数',
            yaxis_title='特征',
            height=max(400, len(feature_cols) * 30),
            template=self.theme
        )
        
        return fig
        
    def plot_regression_analysis(self, regression_results: Dict) -> go.Figure:
        """生成回归分析图表"""
        try:
            # 记录输入数据的类型和内容
            self.logger.info(f"回归分析输入数据列: {regression_results.keys()}")
            
            # 确保处理日期的部分没有错误
            if 'date' in regression_results:
                self.logger.info(f"日期列数据类型: {type(regression_results['date'])}")
                self.logger.info(f"日期列样本数据: {regression_results['date'].head()}")
                regression_results['date'] = pd.to_datetime(regression_results['date'])  # 确保转换为 datetime 类型
            
            # 创建图表对象
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    '特征重要性',
                    '预测值 vs 实际值',
                    '残差分布',
                    '残差时间序列'
                )
            )
            
            # 添加特征重要性图
            self._add_feature_importance(fig, regression_results['feature_importance'], 1, 1)
            
            # 添加预测值vs实际值散点图
            self._add_prediction_scatter(fig, regression_results['y_true'], 
                                          regression_results['y_pred'], 1, 2)
            
            # 添加残差分布图
            self._add_residuals_distribution(fig, regression_results['residuals'], 2, 1)
            
            # 添加残差时间序列图
            self._add_residuals_timeseries(fig, regression_results['residuals'], 2, 2)
            
            # 更新布局
            fig.update_layout(
                height=800,
                width=1200,
                showlegend=True,
                title_text=f"回分析结果 (R² = {regression_results['r2_score']:.3f})"
            )
            
            return fig  # 确保返回一个有效的图表对象
            
        except Exception as e:
            self.logger.error(f"生成回归分析图表时出错: {str(e)}")
            raise  # 确保这里有代码
    
    def generate_prediction_analysis(self, data: pd.DataFrame) -> str:
        """
        生成预测分析图表
        
        Args:
            data: 包含交易数据的DataFrame
            
        Returns:
            str: 生成的图表文件路径
        """
        try:
            # 创建子图
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    '价格预测vs实际',
                    '预测误差分布',
                    '预测准确度随时间变化',
                    '预测区间分析'
                )
            )
            
            # 生成预测值
            if 'predicted_price' not in data.columns:
                self.logger.info("数据中没有预测值，使用简单模型生成预测...")
                try:
                    # 1. 准备特征
                    X = pd.DataFrame({
                        'hour': data.index.hour if isinstance(data.index, pd.DatetimeIndex) else data['trade_hour'],
                        'weekday': data.index.dayofweek if isinstance(data.index, pd.DatetimeIndex) else data['date'].dt.dayofweek
                    })
                    if 'price_ma24' in data.columns:
                        X['price_ma24'] = data['price_ma24']
                    if 'price_volatility' in data.columns:
                        X['price_volatility'] = data['price_volatility']
                    
                    # 2. 准备目标变量
                    y = data['price']
                    
                    # 3. 拟合简单线性回归模型
                    from sklearn.linear_model import LinearRegression
                    model = LinearRegression()
                    model.fit(X, y)
                    
                    # 4. 生成预测值
                    data['predicted_price'] = model.predict(X)
                    
                except Exception as e:
                    self.logger.warning(f"使用模型生成预测值失败: {str(e)}，将使用移动平均...")
                    # 如果模型预测失败，使用简单的移动平均
                    data['predicted_price'] = data['price'].rolling(window=24, min_periods=1).mean()
            
            # 1. 价格预测vs实际
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['price'],
                    mode='lines',
                    name='实际价格'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['predicted_price'],
                    mode='lines',
                    name='预测价格',
                    line=dict(dash='dash')
                ),
                row=1, col=1
            )
            
            # 2. 预测误差分布
            prediction_error = data['price'] - data['predicted_price']
            fig.add_trace(
                go.Histogram(
                    x=prediction_error,
                    name='预测误差分布'
                ),
                row=1, col=2
            )
            
            # 3. 预测准确随时间变化
            mape = abs(prediction_error / data['price']) * 100
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=mape,
                    mode='lines',
                    name='预测准确度(MAPE)'
                ),
                row=2, col=1
            )
            
            # 4. 预测区间分析
            std = prediction_error.std()
            upper_bound = data['predicted_price'] + 2*std
            lower_bound = data['predicted_price'] - 2*std
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['price'],
                    mode='lines',
                    name='实际价格',
                    line=dict(color='blue')
                ),
                row=2, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=upper_bound,
                    mode='lines',
                    name='预测区间上限',
                    line=dict(dash='dash', color='red')
                ),
                row=2, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=lower_bound,
                    mode='lines',
                    name='预测区间下限',
                    line=dict(dash='dash', color='red'),
                    fill='tonexty'
                ),
                row=2, col=2
            )
            
            # 更新布局
            fig.update_layout(
                height=1000,
                width=1400,
                title_text="预测分析",
                showlegend=True,
                template=self.theme
            )
            
            # 更新标轴标签
            fig.update_xaxes(title_text="时间", row=1, col=1)
            fig.update_xaxes(title_text="预测误差", row=1, col=2)
            fig.update_xaxes(title_text="时间", row=2, col=1)
            fig.update_xaxes(title_text="时间", row=2, col=2)
            
            fig.update_yaxes(title_text="价格", row=1, col=1)
            fig.update_yaxes(title_text="频次", row=1, col=2)
            fig.update_yaxes(title_text="MAPE (%)", row=2, col=1)
            fig.update_yaxes(title_text="价格", row=2, col=2)
            
            # 保存图表
            output_path = Path(self.output_dirs['predictions']) / 'prediction_analysis.html'
            fig.write_html(str(output_path))
            
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"生成预测分析图表时出错: {str(e)}")
            raise
    
    def _add_feature_importance(self, fig: go.Figure, feature_importance: dict, row: int, col: int):
        """
        添加特征重要性图
        
        Args:
            fig: Plotly图表对象
            feature_importance: 特征重要性字典
            row: 子图行号
            col: 子图列号
        """
        try:
            # 查feature_importance的类型并行相应处理
            if isinstance(feature_importance, pd.DataFrame):
                # 如果是DataFrame，转换为Series
                importance_series = feature_importance.iloc[:, 0]
            elif isinstance(feature_importance, pd.Series):
                # 如果已经是Series，直接使用
                importance_series = feature_importance
            elif isinstance(feature_importance, dict):
                # 如果是字典，转换为Series
                importance_series = pd.Series(feature_importance)
            else:
                # 如果是其他类型，记录错误并使用空Series
                self.logger.error(f"不支持的feature_importance类型: {type(feature_importance)}")
                importance_series = pd.Series()
            
            # 确保有数据
            if importance_series.empty:
                self.logger.warning("特征重要性为空，使用示例数据")
                importance_series = pd.Series({
                    '示例特征1': 0.5,
                    '示例特征2': 0.3,
                    '示例特征3': 0.2
                })
            
            # 排序
            importance_series = importance_series.sort_values(ascending=True)
            
            # 添加图表
            fig.add_trace(
                go.Bar(
                    x=importance_series.values,
                    y=importance_series.index,
                    orientation='h',
                    name='特征重要性'
                ),
                row=row, col=col
            )
            
            # 更新坐标轴
            fig.update_xaxes(title_text="重要性", row=row, col=col)
            fig.update_yaxes(title_text="特征", row=row, col=col)
            
        except Exception as e:
            self.logger.error(f"添加特征重要性图时出错: {str(e)}")
            # 添加一个空的图表，避免完全失败
            fig.add_trace(
                go.Bar(
                    x=[0],
                    y=['无数据'],
                    orientation='h',
                    name='特征重要性'
                ),
                row=row, col=col
            )
    
    def _add_prediction_scatter(self, fig: go.Figure, y_true: np.ndarray, y_pred: np.ndarray, row: int, col: int):
        """
        添加预测值vs实际散点图
        
        Args:
            fig: Plotly图表对象
            y_true: 实际值数组
            y_pred: 预测值数组
            row: 子图行号
            col: 子图列号
        """
        fig.add_trace(
            go.Scatter(
                x=y_true,
                y=y_pred,
                mode='markers',
                name='预测vs实际',
                marker=dict(
                    color='blue',
                    size=8,
                    opacity=0.6
                )
            ),
            row=row, col=col
        )
        
        # 添加对角线
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='理想预测线',
                line=dict(
                    color='red',
                    dash='dash'
                )
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="际值", row=row, col=col)
        fig.update_yaxes(title_text="预测值", row=row, col=col)
    
    def _add_residuals_distribution(self, fig: go.Figure, residuals: np.ndarray, row: int, col: int):
        """
        添加残差分布图
        
        Args:
            fig: Plotly图表对象
            residuals: 残差数组
            row: 子图行号
            col: 子图列号
        """
        fig.add_trace(
            go.Histogram(
                x=residuals,
                name='残差分布',
                nbinsx=30,
                marker_color='blue',
                opacity=0.7
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="残差", row=row, col=col)
        fig.update_yaxes(title_text="频次", row=row, col=col)
    
    def _add_residuals_timeseries(self, fig: go.Figure, residuals: np.ndarray, row: int, col: int):
        """
        添加残差时间序列图
        
        Args:
            fig: Plotly图表对象
            residuals: 残差数组
            row: 子图行号
            col: 子图列号
        """
        fig.add_trace(
            go.Scatter(
                x=list(range(len(residuals))),
                y=residuals,
                mode='lines',
                name='残差时间序列',
                line=dict(color='blue')
            ),
            row=row, col=col
        )
        
        # 添加零线
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="red",
            row=row,
            col=col
        )
        
        fig.update_xaxes(title_text="样本序号", row=row, col=col)
        fig.update_yaxes(title_text="残差", row=row, col=col)
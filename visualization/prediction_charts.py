import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import os

class PredictionVisualizer:
    def __init__(self):
        self.output_dir = "out/prediction"
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
    def create_prediction_chart(self, actual, predicted):
        """创建预测结果对比图"""
        fig = go.Figure()
        
        # 添加实际值
        fig.add_trace(go.Scatter(
            x=actual.index,
            y=actual,
            name='实际值',
            line=dict(color='blue')
        ))
        
        # 添加预测值
        fig.add_trace(go.Scatter(
            x=predicted.index,
            y=predicted,
            name='预测值',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title='价格预测结果对比',
            xaxis_title='时间',
            yaxis_title='价格',
            showlegend=True
        )
        
        return fig
        
    def create_feature_importance_chart(self, feature_importance_df):
        """创建特征重要性图表"""
        if feature_importance_df is None or len(feature_importance_df) == 0:
            return None
            
        fig = px.bar(
            feature_importance_df,
            x='feature',
            y='importance',
            title='特征重要性分析'
        )
        
        fig.update_layout(
            xaxis_title='特征',
            yaxis_title='重要性',
            showlegend=False
        )
        
        return fig
        
    def create_error_analysis_chart(self, actual, predicted):
        """创建预测误差分析图"""
        errors = predicted - actual
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('预测误差分布', '预测误差随时间变化')
        )
        
        # 误差分布直方图
        fig.add_trace(
            go.Histogram(x=errors, name='误差分布'),
            row=1, col=1
        )
        
        # 误差随时间变化
        fig.add_trace(
            go.Scatter(x=actual.index, y=errors, name='误差'),
            row=2, col=1
        )
        
        fig.update_layout(
            height=800,
            title_text='预测误差分析'
        )
        
        return fig
        
    def save_prediction_analysis(self, actual, predicted, feature_importance_df=None):
        """保存所有预测分析图表"""
        output_paths = {}
        
        # 保存预测结果对比图
        prediction_fig = self.create_prediction_chart(actual, predicted)
        prediction_path = f"{self.output_dir}/prediction_comparison.html"
        prediction_fig.write_html(prediction_path)
        output_paths['prediction_comparison'] = prediction_path
        
        # 保存特征重要性图表
        if feature_importance_df is not None:
            importance_fig = self.create_feature_importance_chart(feature_importance_df)
            if importance_fig is not None:
                importance_path = f"{self.output_dir}/feature_importance.html"
                importance_fig.write_html(importance_path)
                output_paths['feature_importance'] = importance_path
        
        # 保存误差分析图表
        error_fig = self.create_error_analysis_chart(actual, predicted)
        error_path = f"{self.output_dir}/error_analysis.html"
        error_fig.write_html(error_path)
        output_paths['error_analysis'] = error_path
        
        return output_paths
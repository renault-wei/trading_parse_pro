import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from pathlib import Path
from .base import BaseChartGenerator

class TimeChartGenerator(BaseChartGenerator):
    """时间模式图表生成器"""
    
    def generate_time_patterns(self, data: pd.DataFrame):
        """生成时间模式分析图表"""
        try:
            # 创建子图
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    '每小时平均价格', 
                    '工作日vs周末价格',
                    '高峰时段价格分布', 
                    '日内价格模式'
                )
            )
            
            # 1. 每小时平均价格
            hourly_avg = data.groupby('trade_hour')['price'].mean()
            fig.add_trace(
                go.Scatter(
                    x=hourly_avg.index, 
                    y=hourly_avg.values,
                    mode='lines+markers', 
                    name='每小时平均价格'
                ),
                row=1, col=1
            )
            
            # 2. 工作日vs周末价格
            fig.add_trace(
                go.Box(
                    x=data['is_workday'].map({1: '工作日', 0: '周末'}),
                    y=data['price'], 
                    name='工作日价格分布'
                ),
                row=1, col=2
            )
            
            # 3. 高峰时段价格分布
            peak_data = pd.DataFrame({
                '时段': ['早高峰', '晚高峰', '低谷'],
                '平均价格': [
                    data[data['is_morning_peak'] == 1]['price'].mean(),
                    data[data['is_evening_peak'] == 1]['price'].mean(),
                    data[data['is_valley'] == 1]['price'].mean()
                ]
            })
            
            fig.add_trace(
                go.Bar(
                    x=peak_data['时段'], 
                    y=peak_data['平均价格'],
                    name='时段平均价格'
                ),
                row=2, col=1
            )
            
            # 4. 日内价格模式
            fig.add_trace(
                go.Scatter(
                    x=data['hour_sin'], 
                    y=data['hour_cos'],
                    mode='markers',
                    marker=dict(
                        color=data['price'], 
                        colorscale='Viridis'
                    ),
                    name='日内价格模式'
                ),
                row=2, col=2
            )
            
            # 更新布局
            fig.update_layout(
                height=1000,
                width=1400,
                title_text="时间模式分析",
                showlegend=True,
                template=self.theme
            )
            
            # 保存图表
            output_path = Path(self.output_dirs['time_patterns']) / 'time_patterns.html'
            fig.write_html(str(output_path))
            
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"生成时间模式分析图表时出错: {str(e)}")
            return None
            
    def generate_season_period_table(self):
        """生成季节峰谷时段划分表"""
        try:
            # 定义季节
            seasons = ['春季(3月-5月)', '夏季(6月-8月)', '秋季(9月-11月)', '冬季(12月-2月)']
            
            # 定义24小时段
            hours = [f"{str(i).zfill(2)}:00-{str(i+1).zfill(2)}:00" for i in range(24)]
            
            # 定义���段类型及其对应的颜色
            period_colors = {
                '尖峰': '#FF4500',  # 深橙色
                '峰': '#FFB6C1',    # 浅粉色
                '平': '#FFFFFF',    # 白色
                '谷': '#ADD8E6',    # 浅蓝色
                '深谷': '#4169E1'    # 皇家蓝
            }
            
            # 创建时段数据
            period_data = self._get_period_data()
            
            # 创建表格
            fig = self._create_period_table(hours, seasons, period_data, period_colors)
            
            # 保存图表
            output_path = Path(self.output_dirs['time_patterns']) / 'season_period_table.html'
            fig.write_html(
                str(output_path),
                config={
                    'displayModeBar': False,
                    'scrollZoom': False
                }
            )
            
            self.logger.info(f"季节峰谷时段划分表已生成: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"生成季节峰谷时段划分表时出错: {str(e)}")
            return None
            
    def _get_period_data(self) -> dict:
        """获取时段数据"""
        return {
            '春季(3月-5月)': [
                '平', '平', '平', '平', '平', '平',  # 0-6
                '', '平', '谷', '谷', '深', '谷',  # 6-12
                '深谷', '深谷', '谷', '平',  # 12-16
                '平', '尖峰', '尖峰', '尖峰', '峰', '峰',  # 16-22
                '平', '平'  # 22-24
            ],
            '夏季(6月-8月)': [
                '谷', '谷', '谷', '谷', '谷', '谷',  # 0-6
                '平', '平', '平', '平', '平', '平',  # 6-12
                '平', '平', '', '峰',  # 12-16
                '峰', '尖峰', '峰', '尖峰', '尖峰', '平',  # 16-22
                '平', '平'  # 22-24
            ],
            '秋季(9月-11月)': [
                '平', '平', '平', '平', '平', '平',  # 0-6
                '平', '平', '谷', '谷', '深谷', '谷',  # 6-12
                '深谷', '深谷', '谷', '峰',  # 12-16
                '峰', '尖峰', '尖峰', '峰', '峰', '平',  # 16-22
                '峰', '尖峰'  # 22-24
            ],
            '冬季(12月-2月)': [
                '平', '平', '平', '平', '平', '平',  # 0-6
                '平', '平', '谷', '谷', '深谷', '谷',  # 6-12
                '深谷', '深谷', '谷', '尖峰',  # 12-16
                '尖峰', '尖峰', '尖峰', '峰', '峰', '尖峰',  # 16-22
                '峰', '峰'  # 22-24
            ]
        }
        
    def _create_period_table(self, hours, seasons, period_data, period_colors) -> go.Figure:
        """创建时段表格"""
        def get_cell_colors(data):
            """获取单元格颜色"""
            return [period_colors.get(period, '#FFFFFF') for period in data]
            
        fig = go.Figure(data=[
            go.Table(
                header=dict(
                    values=['时段'] + seasons,
                    fill_color='grey',
                    align='center',
                    font=dict(color='white', size=12),
                    height=40
                ),
                cells=dict(
                    values=[hours] + [period_data[season] for season in seasons],
                    fill_color=[['white'] * len(hours)] + 
                              [get_cell_colors(period_data[season]) for season in seasons],
                    align='center',
                    font=dict(color='black', size=12),
                    height=30
                ),
                columnwidth=[0.8, 1, 1, 1, 1]
            )
        ])
        
        # 更新布局
        fig.update_layout(
            title=dict(
                text='2024年季节及峰谷时段划分表',
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top',
                font=dict(size=16)
            ),
            width=1500,
            height=1000,
            margin=dict(t=80, b=20, l=20, r=200)
        )
        
        # 添加图例
        legend_items = [
            ('尖峰', '#FF4500'),
            ('峰', '#FFB6C1'),
            ('平', '#FFFFFF'),
            ('谷', '#ADD8E6'),
            ('深谷', '#4169E1')
        ]
        
        for i, (name, color) in enumerate(legend_items):
            fig.add_annotation(
                x=1.1,
                y=0.9 - i*0.1,
                xref="paper",
                yref="paper",
                text=name,
                showarrow=False,
                font=dict(size=12),
                bgcolor=color,
                bordercolor='black',
                borderwidth=1,
                align="left"
            )
            
        return fig 
    
    def generate_workday_analysis(self, data):
        # 确保使用 trade_date 列
        if 'trade_date' not in data.columns:
            raise ValueError("数据中缺少trade_date列")
        
        # 使用 trade_date 进行分析
        # 例如，您可以将原来的 date 列替换为 trade_date
        # 这里是示例代码，具体实现可能需要根据您的逻辑进行调整
        workday_data = data.groupby('trade_date').agg({
            'price': 'mean',  # 示例聚合
            # 其他需要的聚合操作
        }).reset_index()

        # 继续进行工作日分析的逻辑
        ...
    
    def generate_peak_valley_analysis(self, data: pd.DataFrame) -> str:
        """生成峰谷时段分析图表"""
        try:
            # 数据预处理
            if 'trade_date' in data.columns:
                self.logger.info(f"处理前的trade_date列样本: {data['trade_date'].head()}")
                data['date'] = pd.to_datetime(data['trade_date'])  # 确保转换为 datetime 类型
            else:
                self.logger.error(f"数据中缺少trade_date列，当前列有: {data.columns.tolist()}")
                raise ValueError("数据中缺少trade_date列")
            
            # 确保有必要的列
            if 'trade_hour' not in data.columns and isinstance(data.index, pd.DatetimeIndex):
                data['trade_hour'] = data.index.hour
            elif 'trade_hour' not in data.columns:
                data['trade_hour'] = data['date'].dt.hour
            
            # 记录处理后的数据类型
            self.logger.info(f"处理后的数据列: {data.columns.tolist()}")
            
            if 'is_morning_peak' not in data.columns:
                data['is_morning_peak'] = ((data['trade_hour'] >= 8) & (data['trade_hour'] < 12)).astype(int)
            if 'is_evening_peak' not in data.columns:
                data['is_evening_peak'] = ((data['trade_hour'] >= 17) & (data['trade_hour'] < 21)).astype(int)
            if 'is_valley' not in data.columns:
                data['is_valley'] = ((data['trade_hour'] >= 0) & (data['trade_hour'] < 6)).astype(int)
            
            # 创建子图
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    '峰谷时段平均价格',
                    '峰谷时段价格箱线图',
                    '日内峰谷时段分布',
                    '峰谷时段价格趋势'
                )
            )
            
            # 1. 峰谷时段平均价格
            period_avg = {
                '早高峰': data[data['is_morning_peak'] == 1]['price'].mean(),
                '晚高峰': data[data['is_evening_peak'] == 1]['price'].mean(),
                '低谷': data[data['is_valley'] == 1]['price'].mean(),
                '平段': data[(data['is_morning_peak'] == 0) & 
                          (data['is_evening_peak'] == 0) & 
                          (data['is_valley'] == 0)]['price'].mean()
            }
            
            fig.add_trace(
                go.Bar(
                    x=list(period_avg.keys()),
                    y=list(period_avg.values()),
                    name='平均价格'
                ),
                row=1, col=1
            )
            
            # 2. 峰谷时段价格箱线图
            period_data = []
            period_names = []
            for period in ['早高峰', '晚高峰', '低谷', '平段']:
                if period == '早高峰':
                    mask = data['is_morning_peak'] == 1
                elif period == '晚高峰':
                    mask = data['is_evening_peak'] == 1
                elif period == '低谷':
                    mask = data['is_valley'] == 1
                else:
                    mask = ((data['is_morning_peak'] == 0) & 
                           (data['is_evening_peak'] == 0) & 
                           (data['is_valley'] == 0))
                period_data.extend(data[mask]['price'].tolist())
                period_names.extend([period] * len(data[mask]))
            
            fig.add_trace(
                go.Box(
                    x=period_names,
                    y=period_data,
                    name='价格分布'
                ),
                row=1, col=2
            )
            
            # 3. 日内峰谷时段分布
            hourly_avg = data.groupby('trade_hour')['price'].mean()
            fig.add_trace(
                go.Scatter(
                    x=hourly_avg.index,
                    y=hourly_avg.values,
                    mode='lines+markers',
                    name='小时均价'
                ),
                row=2, col=1
            )
            
            # 添加峰谷时段背景
            fig.add_vrect(
                x0=8, x1=12,
                fillcolor="red", opacity=0.2,
                layer="below", line_width=0,
                annotation_text="早高峰",
                row=2, col=1
            )
            fig.add_vrect(
                x0=17, x1=21,
                fillcolor="red", opacity=0.2,
                layer="below", line_width=0,
                annotation_text="晚高峰",
                row=2, col=1
            )
            fig.add_vrect(
                x0=0, x1=6,
                fillcolor="blue", opacity=0.2,
                layer="below", line_width=0,
                annotation_text="低谷",
                row=2, col=1
            )
            
            # 4. 峰谷时段价格趋势
            # 使用日期的日期部分进行分组
            data['date_only'] = data['date'].dt.date  # 这里确保 date 列是 datetime 类型
            daily_peak_valley = {
                '早高峰': data[data['is_morning_peak'] == 1].groupby('date_only')['price'].mean(),
                '晚高峰': data[data['is_evening_peak'] == 1].groupby('date_only')['price'].mean(),
                '低谷': data[data['is_valley'] == 1].groupby('date_only')['price'].mean()
            }
            
            for period, prices in daily_peak_valley.items():
                fig.add_trace(
                    go.Scatter(
                        x=prices.index,
                        y=prices.values,
                        mode='lines',
                        name=period
                    ),
                    row=2, col=2
                )
            
            # 更新布局
            fig.update_layout(
                height=1000,
                width=1400,
                title_text="峰谷时段分析",
                showlegend=True,
                template=self.theme
            )
            
            # 更新x轴标签
            fig.update_xaxes(title_text="时段", row=1, col=1)
            fig.update_xaxes(title_text="时段", row=1, col=2)
            fig.update_xaxes(title_text="小时", row=2, col=1)
            fig.update_xaxes(title_text="日期", row=2, col=2)
            
            # 更新y轴标签
            fig.update_yaxes(title_text="价格", row=1, col=1)
            fig.update_yaxes(title_text="价格", row=1, col=2)
            fig.update_yaxes(title_text="价格", row=2, col=1)
            fig.update_yaxes(title_text="价格", row=2, col=2)
            
            # 保存图表
            output_path = Path(self.output_dirs['peak_valley']) / 'peak_valley_analysis.html'
            fig.write_html(str(output_path))
            
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"生成峰谷时段分析图表时出错: {str(e)}")
            raise
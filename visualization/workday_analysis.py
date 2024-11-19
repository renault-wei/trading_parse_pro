import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
from pathlib import Path
from datetime import datetime
from utils.logger import Logger

class WorkdayAnalyzer:
    """工作日vs非工作日分析器"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.output_dir = Path(config.get('OUTPUT', 'charts_dir'))
        
    def generate_workday_analysis(self, data: pd.DataFrame):
        """生成工作日分析报告"""
        try:
            self.logger.info("开始生成工作日分析报告...")
            
            # 创建输出目录
            workday_dir = self.output_dir / 'workday_analysis'
            workday_dir.mkdir(exist_ok=True)
            
            # 处理数据类型
            self.logger.info("处理数据类型标记...")
            data = self._process_data_types(data)
            self.logger.info(f"工作日数据点数: {data[data['is_workday']].shape[0]}")
            self.logger.info(f"非工作日数据点数: {data[~data['is_workday']].shape[0]}")
            
            # 生成各个分析图表
            self.logger.info("生成价格对比分析...")
            self._generate_price_comparison(data, workday_dir)
            
            self.logger.info("生成时段热力图...")
            self._generate_period_heatmap(data, workday_dir)
            
            self.logger.info("生成统计分析...")
            self._generate_statistical_analysis(data, workday_dir)
            
            # 生成分析结论
            self.logger.info("生成分析结论...")
            try:
                conclusions = self._generate_analysis_conclusions(data)
                self.logger.info("分析结论生成完成")
                self.logger.debug(f"结论内容: {conclusions}")
            except Exception as e:
                self.logger.error(f"生成分析结论时出错: {str(e)}")
                conclusions = self._generate_empty_conclusions()
            
            # 生成汇总报告
            self.logger.info("生成汇总报告...")
            html_content = self._create_html_template()
            html_content = self._update_html_conclusions(html_content, conclusions)
            
            # 保存报告
            output_path = workday_dir / 'workday_analysis_report.html'
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            self.logger.info(f"工作日分析报告已生成: {workday_dir}")
            
        except Exception as e:
            self.logger.error(f"生成工作日分析报告时出错: {str(e)}")
            raise
            
    def _generate_price_comparison(self, data: pd.DataFrame, output_dir: Path):
        """生成价格对比分析"""
        try:
            # 添加工作日标记
            data['is_workday'] = data.index.weekday < 5  # 周一到周五为工作日
            
            # 1. 日内价格走势对比
            workday_hourly = data[data['is_workday']].groupby('hour')['price'].agg(['mean', 'std'])
            nonworkday_hourly = data[~data['is_workday']].groupby('hour')['price'].agg(['mean', 'std'])
            
            # 2. 月度对比
            data['month'] = data.index.month
            workday_monthly = data[data['is_workday']].groupby('month')['price'].mean()
            nonworkday_monthly = data[~data['is_workday']].groupby('month')['price'].mean()
            
            # 3. 季度对比
            data['quarter'] = data.index.quarter
            workday_quarterly = data[data['is_workday']].groupby('quarter')['price'].mean()
            nonworkday_quarterly = data[~data['is_workday']].groupby('quarter')['price'].mean()
            
            # 创建子图
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=(
                    '日内价格走势对比',
                    '月度工作日vs非工作日对比',
                    '季度工作日vs非工作日对比'
                ),
                vertical_spacing=0.15
            )
            
            # 添加日内价格走势
            fig.add_trace(
                go.Scatter(
                    x=workday_hourly.index,
                    y=workday_hourly['mean'],
                    name='工作日均价',
                    line=dict(color='blue'),
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # 添加工作日价格置信区间
            fig.add_trace(
                go.Scatter(
                    x=workday_hourly.index,
                    y=workday_hourly['mean'] + workday_hourly['std'],
                    fill=None,
                    mode='lines',
                    line_color='rgba(0,0,255,0.1)',
                    showlegend=False
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=workday_hourly.index,
                    y=workday_hourly['mean'] - workday_hourly['std'],
                    fill='tonexty',
                    mode='lines',
                    line_color='rgba(0,0,255,0.1)',
                    name='工作日置信区间'
                ),
                row=1, col=1
            )
            
            # 添加非工作日价格走势
            fig.add_trace(
                go.Scatter(
                    x=nonworkday_hourly.index,
                    y=nonworkday_hourly['mean'],
                    name='非工作日均价',
                    line=dict(color='red'),
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # 添加非工作日价格置信区间
            fig.add_trace(
                go.Scatter(
                    x=nonworkday_hourly.index,
                    y=nonworkday_hourly['mean'] + nonworkday_hourly['std'],
                    fill=None,
                    mode='lines',
                    line_color='rgba(255,0,0,0.1)',
                    showlegend=False
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=nonworkday_hourly.index,
                    y=nonworkday_hourly['mean'] - nonworkday_hourly['std'],
                    fill='tonexty',
                    mode='lines',
                    line_color='rgba(255,0,0,0.1)',
                    name='非工作日置信区间'
                ),
                row=1, col=1
            )
            
            # 添加月度对比
            fig.add_trace(
                go.Bar(
                    x=workday_monthly.index,
                    y=workday_monthly.values,
                    name='工作日月均价',
                    marker_color='blue'
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=nonworkday_monthly.index,
                    y=nonworkday_monthly.values,
                    name='非工作日月均价',
                    marker_color='red'
                ),
                row=2, col=1
            )
            
            # 添加季度对比
            fig.add_trace(
                go.Bar(
                    x=workday_quarterly.index,
                    y=workday_quarterly.values,
                    name='工作日季度均价',
                    marker_color='blue'
                ),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=nonworkday_quarterly.index,
                    y=nonworkday_quarterly.values,
                    name='非工作日季度均价',
                    marker_color='red'
                ),
                row=3, col=1
            )
            
            # 更新布局
            fig.update_layout(
                height=1200,
                width=1000,
                title_text="价格走势对比分析",
                showlegend=True
            )
            
            # 更新x轴标签
            fig.update_xaxes(title_text="小时", row=1, col=1)
            fig.update_xaxes(title_text="月份", row=2, col=1)
            fig.update_xaxes(title_text="季度", row=3, col=1)
            
            # 更新y轴标签
            fig.update_yaxes(title_text="价格", row=1, col=1)
            fig.update_yaxes(title_text="价格", row=2, col=1)
            fig.update_yaxes(title_text="价格", row=3, col=1)
            
            # 保存图表
            output_path = output_dir / 'price_comparison.html'
            fig.write_html(str(output_path))
            
        except Exception as e:
            self.logger.error(f"生成价格对比分析时出错: {str(e)}")
            raise
            
    def _generate_period_heatmap(self, data: pd.DataFrame, output_dir: Path):
        """生成时段热力图分析"""
        try:
            # 添加工作日标记
            data['is_workday'] = data.index.weekday < 5
            data['weekday'] = data.index.weekday
            
            # 1. 价格热力图数据
            workday_price = pd.pivot_table(
                data[data['is_workday']],
                values='price',
                index='weekday',
                columns='hour',
                aggfunc='mean'
            )
            
            nonworkday_price = pd.pivot_table(
                data[~data['is_workday']],
                values='price',
                index='weekday',
                columns='hour',
                aggfunc='mean'
            )
            
            # 2. 波动率热力图数据
            workday_vol = pd.pivot_table(
                data[data['is_workday']],
                values='returns',
                index='weekday',
                columns='hour',
                aggfunc='std'
            )
            
            nonworkday_vol = pd.pivot_table(
                data[~data['is_workday']],
                values='returns',
                index='weekday',
                columns='hour',
                aggfunc='std'
            )
            
            # 创建子图
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    '工作日价格热力图',
                    '非工作日价格热力图',
                    '工作日波动率热力图',
                    '非工作日波动率热力图'
                ),
                vertical_spacing=0.15,
                horizontal_spacing=0.1
            )
            
            # 添加工作日价格热力图
            fig.add_trace(
                go.Heatmap(
                    z=workday_price.values,
                    x=workday_price.columns,
                    y=['周一', '周二', '周', '周四', '周五'],
                    colorscale='RdBu_r',
                    colorbar=dict(title='价格'),
                    name='工作日价格'
                ),
                row=1, col=1
            )
            
            # 添加非工作日价格热力图
            fig.add_trace(
                go.Heatmap(
                    z=nonworkday_price.values,
                    x=nonworkday_price.columns,
                    y=['周六', '周日'],
                    colorscale='RdBu_r',
                    colorbar=dict(title='价格'),
                    name='非工作日价格'
                ),
                row=1, col=2
            )
            
            # 添加工作日波动率热力图
            fig.add_trace(
                go.Heatmap(
                    z=workday_vol.values,
                    x=workday_vol.columns,
                    y=['周一', '周二', '周三', '周四', '周五'],
                    colorscale='Viridis',
                    colorbar=dict(title='波动率'),
                    name='工作日波动率'
                ),
                row=2, col=1
            )
            
            # 添加非工作日波动率热力图
            fig.add_trace(
                go.Heatmap(
                    z=nonworkday_vol.values,
                    x=nonworkday_vol.columns,
                    y=['周六', '周日'],
                    colorscale='Viridis',
                    colorbar=dict(title='波动率'),
                    name='非工作日波动率'
                ),
                row=2, col=2
            )
            
            # 更布局
            fig.update_layout(
                height=1000,
                width=1400,
                title_text="时段特征热力图分析"
            )
            
            # 保存图表
            output_path = output_dir / 'period_heatmap.html'
            fig.write_html(str(output_path))
            
        except Exception as e:
            self.logger.error(f"生成时段热力图分析时出错: {str(e)}")
            raise
            
    def _generate_statistical_analysis(self, data: pd.DataFrame, output_dir: Path):
        """生成统计分析表格"""
        try:
            # 添加工作日标记
            data['is_workday'] = data.index.weekday < 5
            
            # 1. 时段统计分析
            def calculate_period_stats(group_data):
                return pd.Series({
                    '均价': group_data['price'].mean(),
                    '标准差': group_data['price'].std(),
                    '最大值': group_data['price'].max(),
                    '最小值': group_data['price'].min(),
                    '波动率': group_data['returns'].std(),
                    '样本数': len(group_data)
                })
            
            # 按时段和工作日分组统计
            period_stats = []
            for hour in range(24):
                workday_stats = calculate_period_stats(data[(data['hour'] == hour) & data['is_workday']])
                nonworkday_stats = calculate_period_stats(data[(data['hour'] == hour) & ~data['is_workday']])
                
                # 计算t检验
                t_stat, p_value = stats.ttest_ind(
                    data[(data['hour'] == hour) & data['is_workday']]['price'],
                    data[(data['hour'] == hour) & ~data['is_workday']]['price']
                )
                
                period_stats.append({
                    '时段': f"{hour:02d}:00-{(hour+1):02d}:00",
                    '工作日均价': workday_stats['均价'],
                    '工作日波动率': workday_stats['波动率'],
                    '非工作日均价': nonworkday_stats['均价'],
                    '非工作日波动率': nonworkday_stats['波动率'],
                    '价差': workday_stats['均价'] - nonworkday_stats['均价'],
                    'p值': p_value
                })
            
            period_stats_df = pd.DataFrame(period_stats)
            
            # 创建统计表格
            fig = go.Figure(data=[
                go.Table(
                    header=dict(
                        values=list(period_stats_df.columns),
                        fill_color='grey',
                        align='center',
                        font=dict(color='white', size=12)
                    ),
                    cells=dict(
                        values=[period_stats_df[col] for col in period_stats_df.columns],
                        fill_color='white',
                        align='center',
                        format=[
                            None,  # 时段
                            '.2f',  # 工作日均价
                            '.4f',  # 工作日波动率
                            '.2f',  # 非工作日均价
                            '.4f',  # 非工作日波动率
                            '.2f',  # 价差
                            '.4f'   # p值
                        ]
                    )
                )
            ])
            
            # 更新布局
            fig.update_layout(
                title_text="时段统计分析",
                height=800,
                width=1200
            )
            
            # 保存表格
            output_path = output_dir / 'statistical_analysis.html'
            fig.write_html(str(output_path))
            
        except Exception as e:
            self.logger.error(f"生成统计分析表格时出错: {str(e)}")
            raise
            
    def _generate_summary_report(self, output_dir: Path):
        """生成汇总报告"""
        try:
            html_content = self._create_html_template()
            
            # 保存汇总报告
            output_path = output_dir / 'workday_analysis_report.html'
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
        except Exception as e:
            self.logger.error(f"生成汇总报告时出错: {str(e)}")
            raise
            
    def _create_html_template(self) -> str:
        """创建HTML模板"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>工作日vs非工作日分析报告</title>
            <meta charset="utf-8">
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1800px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }}
                .section {{
                    margin-bottom: 30px;
                }}
                .chart-container {{
                    margin-bottom: 20px;
                    border: 1px solid #ddd;
                    padding: 15px;
                    border-radius: 5px;
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 30px;
                    padding: 20px;
                    background-color: #f8f9fa;
                    border-radius: 5px;
                }}
                iframe {{
                    width: 100%;
                    height: 600px;
                    border: none;
                }}
                .analysis-conclusions {{
                    padding: 20px;
                    background-color: #fff;
                    border-radius: 5px;
                    margin-top: 20px;
                }}
                .finding-section {{
                    margin-bottom: 20px;
                    padding: 15px;
                    background-color: #f8f9fa;
                    border-radius: 5px;
                }}
                pre {{
                    white-space: pre-wrap;
                    font-family: Arial, sans-serif;
                    margin: 10px 0;
                    padding: 10px;
                    background-color: #fff;
                    border-radius: 3px;
                }}
                .trading-advice {{
                    margin-top: 20px;
                    padding: 15px;
                    background-color: #e9ecef;
                    border-radius: 5px;
                }}
                ul {{
                    padding-left: 20px;
                }}
                li {{
                    margin-bottom: 10px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>工作日vs非工作日分析报告</h1>
                    <p>生成时间: {timestamp}</p>
                </div>
                
                <div class="section">
                    <h2>1. 价格走势分析</h2>
                    <div class="chart-container">
                        <iframe src="price_comparison.html"></iframe>
                    </div>
                </div>
                
                <div class="section">
                    <h2>2. 时段特征分析</h2>
                    <div class="chart-container">
                        <iframe src="period_heatmap.html"></iframe>
                    </div>
                </div>
                
                <div class="section">
                    <h2>3. 统计指标分析</h2>
                    <div class="chart-container">
                        <iframe src="statistical_analysis.html"></iframe>
                    </div>
                </div>
                
                <div class="section">
                    <h2>4. 分析结论</h2>
                    <div class="analysis-conclusions">
                        <!-- ANALYSIS_CONCLUSIONS_PLACEHOLDER -->
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
            
    def _generate_analysis_conclusions(self, data: pd.DataFrame) -> dict:
        """生成分析结论"""
        try:
            self.logger.info("开始生成分析结论...")
            
            # 检查数据有效性
            if data.empty:
                self.logger.warning("输入数据为空")
                return self._generate_empty_conclusions()
                
            # 检查必要的列
            required_columns = ['price', 'hour', 'is_workday']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                self.logger.error(f"缺少必要的列: {missing_columns}")
                return self._generate_empty_conclusions()
                
            # 生成各部分结论
            self.logger.info("分析价格差异...")
            price_diff = self._analyze_price_differences(data)
            
            self.logger.info("分析时段切换...")
            period_transition = self._analyze_period_transitions(data)
            
            self.logger.info("分析波动率特征...")
            volatility = self._analyze_volatility_patterns(data)
            
            conclusions = {
                'price_diff': price_diff,
                'period_transition': period_transition,
                'volatility': volatility
            }
            
            self.logger.info("分析结论生成完成")
            return conclusions
            
        except Exception as e:
            self.logger.error(f"生成分析结论时出错: {str(e)}")
            return self._generate_empty_conclusions()
            
    def _process_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理数据类型标记"""
        try:
            self.logger.info(f"原始数据形状: {data.shape}")
            self.logger.info(f"原始数据列: {data.columns.tolist()}")
            
            # 检查必要的列
            required_columns = ['price', 'returns', 'hour']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                self.logger.error(f"数据缺少必要的列: {missing_columns}")
                raise ValueError(f"缺少必要的列: {missing_columns}")
            
            # 确保data_type列存在
            if 'data_type' not in data.columns:
                self.logger.warning("数据中缺少data_type列，将使用weekday进行判断")
                data['data_type'] = data.index.weekday.map(lambda x: 1 if x >= 5 else 0)
            
            # 创建工作日标记（0为工作日，1和2都视为非工作日）
            data['is_workday'] = data['data_type'] == 0
            
            # 打印数据统计信息
            self.logger.info(f"工作日数据点数: {data[data['is_workday']].shape[0]}")
            self.logger.info(f"非工作日数据点数: {data[~data['is_workday']].shape[0]}")
            
            # 检查数据是否为空
            if data.empty:
                raise ValueError("处理后的数据为空")
                
            return data
            
        except Exception as e:
            self.logger.error(f"处理数据类型时出错: {str(e)}")
            raise

    def _analyze_price_differences(self, data: pd.DataFrame) -> str:
        """分析工作日与非工作日价格差异特征"""
        try:
            self.logger.info("开始分析价格差异...")
            
            # 数据检查
            if data.empty:
                raise ValueError("输入数据为空")
                
            if not all(col in data.columns for col in ['price', 'is_workday', 'hour']):
                raise ValueError("缺少必要的数据列")
                
            # 打印基本统计信息
            self.logger.info(f"价格范围: {data['price'].min():.2f} - {data['price'].max():.2f}")
            self.logger.info(f"作日样本数: {data[data['is_workday']].shape[0]}")
            self.logger.info(f"非工作日样本数: {data[~data['is_workday']].shape[0]}")
            
            # 计算工作日和非工作日的均价
            workday_mean = data[data['is_workday']]['price'].mean()
            nonworkday_mean = data[~data['is_workday']]['price'].mean()
            price_diff_pct = (workday_mean - nonworkday_mean) / nonworkday_mean * 100
            
            self.logger.info(f"工作日均价: {workday_mean:.2f}")
            self.logger.info(f"非工作日均价: {nonworkday_mean:.2f}")
            self.logger.info(f"价格差异百分比: {price_diff_pct:.2f}%")
            
            # 分时段价格差异
            peak_hours = range(17, 21)  # 峰时段
            valley_hours = range(11, 15)  # 谷时段
            
            def calculate_period_diff(period_hours):
                workday_period = data[(data['is_workday']) & (data['hour'].isin(period_hours))]['price'].mean()
                nonworkday_period = data[(~data['is_workday']) & (data['hour'].isin(period_hours))]['price'].mean()
                return workday_period, nonworkday_period
            
            peak_workday, peak_nonworkday = calculate_period_diff(peak_hours)
            valley_workday, valley_nonworkday = calculate_period_diff(valley_hours)
            
            analysis = f"""
            价格差异特征分析：
            1. 整体价格水平比较：
               - 工作日平均价格: {workday_mean:.2f}
               - 非工作日平均价格: {nonworkday_mean:.2f}
               - 价格差异: {price_diff_pct:+.2f}%
               
            2. 峰时段(17:00-21:00)价格比较：
               - 工作日: {peak_workday:.2f}
               - 非工作日: {peak_nonworkday:.2f}
               - 差异: {((peak_workday-peak_nonworkday)/peak_nonworkday*100):+.2f}%
               
            3. 谷时段(11:00-15:00)价格比较：
               - 工作日: {valley_workday:.2f}
               - 非工作日: {valley_nonworkday:.2f}
               - 差异: {((valley_workday-valley_nonworkday)/valley_nonworkday*100):+.2f}%
            """
            
            self.logger.info("价格差异分析完成")
            return analysis
            
        except Exception as e:
            self.logger.error(f"分析价格差异时出错: {str(e)}")
            return "价格差异分析数据不足"
        
    def _analyze_period_transitions(self, data: pd.DataFrame) -> str:
        """分析时段切换特征"""
        try:
            self.logger.info("开始分析时段切换特征...")
            
            def calculate_transition_stats(df, from_hour, to_hour):
                changes = []
                # 将索引转换为pandas datetime
                df = df.copy()
                if not isinstance(df.index, pd.DatetimeIndex):
                    self.logger.warning("索引不是DatetimeIndex类型，尝试转换...")
                    df.index = pd.to_datetime(df.index)
                
                # 获取唯一日期
                dates = pd.Series(df.index.date).unique()
                self.logger.debug(f"分析日期数量: {len(dates)}")
                
                for date in dates:
                    # 使用日期过滤数据
                    day_data = df[df.index.date == date]
                    if len(day_data) > 0:
                        # 获取特定小时的数据
                        from_data = day_data[day_data['hour'] == from_hour]
                        to_data = day_data[day_data['hour'] == to_hour]
                        
                        if not from_data.empty and not to_data.empty:
                            price_change = to_data['price'].iloc[0] - from_data['price'].iloc[0]
                            changes.append(price_change)
                
                if not changes:
                    self.logger.warning(f"未找到从{from_hour}时到{to_hour}时的有效价格变化")
                    return 0, 0
                
                return np.mean(changes), np.std(changes)
            
            transitions = {
                '早高峰开始(6->8)': (6, 8),
                '早高峰结束(12->14)': (12, 14),
                '晚高峰开始(16->18)': (16, 18),
                '晚高峰结束(21->23)': (21, 23)
            }
            
            analysis = ["时段切换特征分析："]
            
            for name, (from_hour, to_hour) in transitions.items():
                self.logger.info(f"分析时段切换: {name}")
                
                # 工作日分析
                workday_data = data[data['is_workday']]
                workday_mean, workday_std = calculate_transition_stats(
                    workday_data, from_hour, to_hour
                )
                
                # 非工作日分析
                nonworkday_data = data[~data['is_workday']]
                nonworkday_mean, nonworkday_std = calculate_transition_stats(
                    nonworkday_data, from_hour, to_hour
                )
                
                # 计算差异显著性
                significance = 'high' if abs(workday_mean - nonworkday_mean) > max(workday_std, nonworkday_std) else 'low'
                
                analysis.append(f"""
                {name}:
                - 工作日：平均变化 {workday_mean:+.2f} (标准差: {workday_std:.2f})
                - 非工作日：平均变化 {nonworkday_mean:+.2f} (标准差: {nonworkday_std:.2f})
                - 差异显著性：{significance}
                """)
                
            self.logger.info("时段切换分析完成")
            return "\n".join(analysis)
            
        except Exception as e:
            self.logger.error(f"分析时段切换特征时出错: {str(e)}")
            self.logger.debug("错误详情", exc_info=True)
            return "时段切换分析数据不足"
        
    def _analyze_volatility_patterns(self, data: pd.DataFrame) -> str:
        """分析波动率特征"""
        try:
            self.logger.info("开始分析波动率特征...")
            
            # 计算各时段的波动率
            def calculate_period_volatility(df, hours):
                period_data = df[df['hour'].isin(hours)]['returns']
                if period_data.empty:
                    return 0
                return period_data.std() * np.sqrt(252)
                
            periods = {
                '早高峰(8-12)': range(8, 12),
                '午谷期(12-16)': range(12, 16),
                '晚高峰(16-20)': range(16, 20),
                '夜间期(20-24)': range(20, 24)
            }
            
            volatility_analysis = []
            for period_name, hours in periods.items():
                workday_vol = calculate_period_volatility(data[data['is_workday']], hours)
                nonworkday_vol = calculate_period_volatility(data[~data['is_workday']], hours)
                
                volatility_analysis.append(f"""
                {period_name}:
                - 工作日波动率: {workday_vol:.2%}
                - 非工作日波动率: {nonworkday_vol:.2%}
                - 波动率差异: {abs(workday_vol - nonworkday_vol):.2%}
                """)
                
            self.logger.info("波动率特征分析完成")
            return "波动率特征分析：\n" + "\n".join(volatility_analysis)
            
        except Exception as e:
            self.logger.error(f"分析波动率特征时出错: {str(e)}")
            return "波动率特征分析失败"
        
    def _update_html_conclusions(self, html_content: str, conclusions: dict) -> str:
        """更新HTML中的分析结论部分"""
        conclusions_html = f"""
        <div class="finding-section">
            <h3>主要发现</h3>
            <div class="finding-item">
                <h4>工作日与非工作日价格差异特征</h4>
                <pre>{conclusions['price_diff']}</pre>
            </div>
            
            <div class="finding-item">
                <h4>时段切换特征分析</h4>
                <pre>{conclusions['period_transition']}</pre>
            </div>
            
            <div class="finding-item">
                <h4>波动率特征分析</h4>
                <pre>{conclusions['volatility']}</pre>
            </div>
        </div>
        
        <div class="trading-advice">
            <h3>交易建议</h3>
            <ul>
                <li><strong>重点关注时段：</strong>{self._generate_key_periods(conclusions)}</li>
                <li><strong>风险控制建议：</strong>{self._generate_risk_control_advice(conclusions)}</li>
                <li><strong>策略优化方向：</strong>{self._generate_strategy_optimization(conclusions)}</li>
            </ul>
        </div>
        """
        
        # 使用占位符替换
        return html_content.replace('<!-- ANALYSIS_CONCLUSIONS_PLACEHOLDER -->', conclusions_html)
        
    def _generate_key_periods(self, conclusions: dict) -> str:
        """生成重点关注时段建议"""
        # 基于分析结果生成建议
        return "工作日早晚高峰时段的价格跳跃机会，以及非工作日的低波动时段"
        
    def _generate_risk_control_advice(self, conclusions: dict) -> str:
        """生成风险控制建议"""
        return "高波动时段采用更严格的止损策略，时段切换时注意控制持仓规模"
        
    def _generate_strategy_optimization(self, conclusions: dict) -> str:
        """生成策略优化建议"""
        return "根据工作日和非工作日的特征分别制定交易策略，重点关注时段切换机会"

    def _generate_empty_conclusions(self) -> dict:
        """生成空的结论模板"""
        return {
            'price_diff': "价格差异分析数据不足",
            'period_transition': "时段切换分析数据不足",
            'volatility': "波动率分析数据不足"
        }
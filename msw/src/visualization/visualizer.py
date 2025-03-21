import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from src.config.config import Config

class Visualizer:
    def __init__(self):
        # 初始化可视化器
        self.output_dir = Path(Config.PATH_CONFIG['visualization_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # 设置绘图样式
        sns.set_style('whitegrid')
        plt.rcParams['font.size'] = 12
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        # 从配置中获取目标列名和转换方法
        self.target_column = Config.DATA_CONFIG['target_column']
        self.transform_method = Config.FEATURE_CONFIG['target_transform_method']
        self.transformed_column = f'{self.target_column}_{self.transform_method}'
        self.target_column_pred = f'{self.target_column}_pred'

    def plot_country_predictions(self, country: str, test_data: pd.DataFrame):
        """绘制单个国家的预测结果

        参数:
            country: 国家名称
            test_data: 包含'Year'、目标列和预测列的测试数据
        """
        plt.figure(figsize=(12, 6))
        
        # Plot actual and predicted values
        plt.plot(test_data['Year'], test_data[self.target_column], 'o-', color='#2ecc71', 
                 label='实际值', linewidth=2, markersize=8)
        plt.plot(test_data['Year'], test_data[self.target_column_pred], 'x--', color='#e74c3c',
                 label='预测值', linewidth=2, markersize=8)
        
        plt.title(f'{self.target_column} 固体废物产生量预测 - {country}', pad=20)
        plt.xlabel('年份', labelpad=10)
        plt.ylabel(self.target_column, labelpad=10)
        plt.legend(frameon=True, fancybox=True, shadow=True)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(self.output_dir / f'{country}_country_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_time_predictions(self, country: str, train_data: pd.DataFrame, test_data: pd.DataFrame):
        """绘制单个国家的时间序列预测结果

        参数:
            country: 国家名称
            train_data: 包含'Year'和目标列的训练数据
            test_data: 包含'Year'、目标列和预测列的测试数据
        """
        plt.figure(figsize=(12, 6))
        
        # Plot historical data
        plt.plot(train_data['Year'], train_data[self.target_column], 'o-', color='#3498db',
                 label='历史数据', linewidth=2, markersize=8)
        
        # 绘制实际值和预测值
        plt.plot(test_data['Year'], test_data[self.target_column], 'o-', color='#2ecc71',
                 label='实际值', linewidth=2, markersize=8)
        plt.plot(test_data['Year'], test_data[self.target_column_pred], 'x--', color='#e74c3c',
                 label='预测值', linewidth=2, markersize=8)
        
        plt.title(f'{self.target_column} 固体废物产生量时间序列 - {country}', pad=20)
        plt.xlabel('年份', labelpad=10)
        plt.ylabel(f'{self.target_column} (吨)', labelpad=10)
        plt.legend(frameon=True, fancybox=True, shadow=True)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(self.output_dir / f'{country}_time_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_scatter_comparison(self, test_data: pd.DataFrame):
        """国家外样本测试集 - 实际值与预测值对比"""
        plt.figure(figsize=(10, 8))
        plt.scatter(test_data[self.target_column], test_data[self.target_column_pred], alpha=0.6, color='#3498db')
        min_val = test_data[[self.target_column, self.target_column_pred]].min().min()
        max_val = test_data[[self.target_column, self.target_column_pred]].max().max()
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        plt.xlabel(f'实际{self.target_column}值', labelpad=10)
        plt.ylabel(f'预测{self.target_column}值', labelpad=10)
        plt.title('国家外样本测试集 - 实际值与预测值对比', pad=20)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'scatter_comparison.png', dpi=300)
        plt.show()

    def plot_grouped_error(self, test_data: pd.DataFrame):
        """分组误差分析（地区/收入）"""
        plt.figure(figsize=(16, 6))
        
        # 计算误差百分比
        if 'Error_percent' not in test_data.columns:
            test_data['Error_percent'] = ((test_data[target_column_pred] - test_data[self.target_column]) 
                                         / test_data[self.target_column] * 100)
        
        # 地区分组
        plt.subplot(1, 2, 1)
        sns.boxplot(x='Region', y='Error_percent', data=test_data, palette='viridis')
        plt.axhline(y=0, color='#e74c3c', linestyle='--', linewidth=2)
        plt.xlabel('地区', labelpad=10)
        plt.ylabel('预测误差百分比 (%)', labelpad=10)
        plt.title('地区分组误差分布', pad=15)
        plt.xticks(rotation=45, ha='right')

        # 收入分组
        plt.subplot(1, 2, 2)
        sns.boxplot(x='Income Group', y='Error_percent', data=test_data, palette='viridis')
        plt.axhline(y=0, color='#e74c3c', linestyle='--', linewidth=2)
        plt.xlabel('收入组', labelpad=10)
        plt.ylabel('预测误差百分比 (%)', labelpad=10)
        plt.title('收入分组误差分布', pad=15)
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()

        # 新增收入分组趋势分析
        plt.figure(figsize=(16, 8))
        
        # 误差趋势分析
        plt.subplot(2, 2, 1)
        sns.lineplot(x='Year', y='Error_percent', hue='Income Group', 
                    data=test_data, palette='Set2', ci=None)
        plt.axhline(0, color='#e74c3c', linestyle='--')
        plt.title('不同收入组误差年度趋势')
        
        # 误差分布密度
        plt.subplot(2, 2, 3)
        for income_group in test_data['Income Group'].unique():
            subset = test_data[test_data['Income Group']==income_group]
            sns.kdeplot(subset['Error_percent'], label=income_group)
        plt.legend()
        plt.title('误差分布密度')
        
        # 累积误差分布
        plt.subplot(2, 2, 4)
        for income_group in test_data['Income Group'].unique():
            subset = test_data[test_data['Income Group']==income_group]
            subset['Error_percent'].hist(cumulative=True, density=True, 
                                        histtype='step', bins=50, 
                                        label=income_group)
        plt.legend()
        plt.title('累积误差分布')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'enhanced_income_group_analysis.png', dpi=300)
        plt.show()
        
        # 检查缺失数据
        if test_data['Income Group'].isnull().any():
            print('发现缺失的Income Group数据，可能影响分析结果')

    def plot_overall_performance(self, test_data: pd.DataFrame):
        """绘制整体预测性能，包括误差分布和按年份的误差

        参数:
            test_data: 包含'Year'、目标列和预测列的测试数据
        """
        # 绘制国家外样本测试集的误差趋势
        self.plot_country_error_trend(test_data)

        # 绘制时间外样本测试集的误差分析
        self.plot_time_error_analysis(test_data)
        
        # 绘制散点对比图和分组误差分析
        self.plot_scatter_comparison(test_data)
        self.plot_grouped_error(test_data)


    def plot_all_country_predictions(self, test_data: pd.DataFrame):
        """绘制国家外样本测试集中所有国家的预测结果

        参数:
            test_data: 包含'Country Name'、'Year'、目标列和预测列的测试数据
        """
        # 获取所有唯一的国家
        countries = test_data['Country Name'].unique()
        
        # 为每个国家绘制预测结果
        for country in countries:
            country_data = test_data[test_data['Country Name'] == country].copy()
            self.plot_country_predictions(country, country_data)
    
    def plot_top_time_predictions(self, train_data: pd.DataFrame, test_data: pd.DataFrame, n_top=10):
        """绘制表现最好的国家的时间序列预测结果

        参数:
            train_data: 包含'Year'和目标列的训练数据
            test_data: 包含'Year'、目标列和预测列的测试数据
            n_top: 要绘制的表现最好的国家数量
        """
        # 合并训练和测试数据以检查数据完整性
        full_data = pd.concat([train_data[['Country Name', 'Year', self.target_column]], 
                              test_data[['Country Name', 'Year', self.target_column]]])
        
        # 获取1990到2022年的所有年份
        expected_years = set(range(1990, 2023))
        
        # 找出具有完整数据的国家
        complete_countries = []
        for country in full_data['Country Name'].unique():
            country_years = set(full_data[full_data['Country Name'] == country]['Year'])
            if country_years.issuperset(expected_years):
                complete_countries.append(country)
        
        # 计算具有完整数据的国家的平均绝对百分比误差
        test_data['Error_percent'] = abs((test_data[self.target_column_pred] - test_data[self.target_column]) / test_data[self.target_column] * 100)
        country_errors = test_data[test_data['Country Name'].isin(complete_countries)]\
                        .groupby('Country Name')['Error_percent'].mean()
        
        # Get top performing countries among those with complete data
        top_countries = country_errors.nsmallest(n_top).index
        
        # Plot for each top country
        for country in top_countries:
            # Get training and test data for the country and sort by year
            country_train = train_data[train_data['Country Name'] == country].copy()
            country_test = test_data[test_data['Country Name'] == country].copy()
            
            # Sort data by year to ensure continuity
            country_train = country_train.sort_values('Year')
            country_test = country_test.sort_values('Year')
            
            # Only proceed if we have both training and test data
            if not country_train.empty and not country_test.empty:
                self.plot_time_predictions(country, country_train, country_test)

    def plot_country_error_trend(self, test_data: pd.DataFrame):
        """国家外样本测试集 - 预测误差随时间变化"""
        plt.figure(figsize=(14, 8))
        test_data['Error_percent'] = ((test_data[self.target_column_pred] - test_data[self.target_column]) / test_data[self.target_column]) * 100
        yearly_error = test_data.groupby('Year')['Error_percent'].agg(['mean', 'std']).reset_index()
        
        plt.errorbar(yearly_error['Year'], yearly_error['mean'], yerr=yearly_error['std'], 
                    fmt='o-', capsize=5, ecolor='gray', alpha=0.7)
        plt.axhline(y=0, color='#e74c3c', linestyle='--', linewidth=2)
        plt.title('国家外样本测试集 - 预测误差随时间变化', pad=20)
        plt.xlabel('年份', labelpad=10)
        plt.ylabel('平均预测误差百分比 (%)', labelpad=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'country_error_trend.png', dpi=300)
        plt.show()

    def plot_time_error_analysis(self, test_data: pd.DataFrame):
        """时间外样本测试集 - 预测误差分析"""
        plt.figure(figsize=(12, 6))
        test_data['Error_percent'] = ((test_data[self.target_column_pred] - test_data[self.target_column]) / test_data[self.target_column]) * 100
        
        # 误差分布直方图
        plt.subplot(1, 2, 1)
        sns.histplot(test_data['Error_percent'], bins=20, color='#3498db', alpha=0.7)
        plt.axvline(x=0, color='#e74c3c', linestyle='--', linewidth=2)
        plt.title('预测误差分布')
        plt.xlabel('预测误差百分比 (%)')
        plt.ylabel('频数')

        # 误差与年份关系
        plt.subplot(1, 2, 2)
        sns.regplot(x='Year', y='Error_percent', data=test_data, 
                  scatter_kws={'alpha':0.6, 'color':'#2ecc71'},
                  line_kws={'color':'#e74c3c', 'linestyle':'--'})
        plt.title('误差与年份关系')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'time_error_analysis.png', dpi=300)
        plt.show()

    def plot_country_multiple_metrics(self, country: str, data: pd.DataFrame):
        """国家多指标趋势分析"""
        fig, axes = plt.subplots(3, 2, figsize=(18, 15))
        fig.suptitle(f"{country} 趋势分析", fontsize=16)

        # 目标变量实际值与预测值对比
        axes[0,0].plot(data['Year'], data[self.target_column], 'o-', color='#3498db', label='实际值')
        axes[0,0].plot(data['Year'], data[self.target_column_pred], 'x--', color='#e74c3c', label='预测值')
        axes[0,0].set_title(f'{self.target_column}对比')
        axes[0,0].legend()

        # 人口趋势
        axes[0,1].plot(data['Year'], data['Population'], 'o-', color='#2ecc71')
        axes[0,1].set_title('人口趋势')

        # GDP PPP趋势
        axes[1,0].plot(data['Year'], data['GDP PPP 2017'], 'o-', color='#9b59b6')
        axes[1,0].set_title('GDP PPP趋势')

        # 人均GDP
        axes[1,1].plot(data['Year'], data['GDP PPP/capita 2017'], 'o-', color='#e67e22')
        axes[1,1].set_title('人均GDP')

        # 计算预测误差百分比
        if self.target_column in data.columns and self.target_column_pred in data.columns:
            # 处理除零异常
            denominator = data[self.target_column].replace(0, np.nan)
            data['Error_percent'] = np.where(
                denominator.notna(),
                (data[self.target_column_pred] - data[self.target_column]) / denominator * 100,
                (data[self.target_column_pred] - data[self.target_column])  # 绝对误差
            )

        # 预测误差
        axes[2,0].bar(data['Year'], data['Error_percent'], color='#f1c40f')
        axes[2,0].axhline(0, color='#e74c3c', linestyle='--')
        axes[2,0].set_title('预测误差')

        # 目标变量与人均GDP关系
        axes[2,1].scatter(data['GDP PPP/capita 2017'], data[self.target_column], color='#3498db')
        axes[2,1].set_title(f'{self.target_column}与人均GDP关系')

        plt.tight_layout()
        plt.savefig(self.output_dir / f'{country}_multi_metrics.png', dpi=300)
        plt.show()
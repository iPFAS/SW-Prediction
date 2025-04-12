import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Dict, Optional, Tuple, Set
from ..config.config import Config

class FeatureEngineering:
    def __init__(self):
        """初始化特征工程类"""
        self.required_columns = Config.FEATURE_CONFIG['usecols']
        self.base_year = Config.FEATURE_CONFIG['base_year']
        self.global_stats = {}  # 存储全球统计指标
        
    def save_params(self, params_path: str) -> None:
        """保存特征工程参数到文件"""
        params = {
            'global_stats': self.global_stats  # 保存全球统计指标
        }
        pd.to_pickle(params, params_path)
        
    def load_params(self, params_path: str) -> None:
        """从文件加载特征工程参数"""
        params = pd.read_pickle(params_path)
        self.global_stats = params.get('global_stats', {})  # 加载全球统计指标

    def validate_columns(self, df: pd.DataFrame) -> None:
        """验证输入数据是否包含所需列"""
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def transform_target(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """转换目标变量
        
        Args:
            df: 输入数据
            target_column: 目标列名
            
        Returns:
            转换后的DataFrame
        """
        df = df.copy()
        method = Config.FEATURE_CONFIG['target_transform_method']
        transformed_column = f'{target_column}_{method}'

        if method == 'log':
            df[transformed_column] = np.log1p(df[target_column])
        elif method == 'boxcox':
            df[transformed_column], _ = stats.boxcox(df[target_column] + 1)
        else:
            df[transformed_column] = df[target_column]

        return df

    def fit(self, df: pd.DataFrame) -> None:
        """拟合训练数据并保存全球统计参数
        
        计算并保存全球经济和人口发展趋势指标，使用分位数等统计方法避免数据泄露
        """
        # 为了公平计算全球分位数，每个国家只取一个代表值（基准年或平均值）
        country_stats = df.groupby('Country Name').agg({
            'GDP PPP/capita 2017': 'mean',
            'Population': 'mean',
            'GDP PPP 2017': 'mean'
        })
        
        # 计算全球经济发展阶段（使用分位数）- 基于国家平均值
        gdp_quantiles = country_stats['GDP PPP/capita 2017'].quantile([0.2, 0.4, 0.6, 0.8]).to_dict()
        population_quantiles = country_stats['Population'].quantile([0.2, 0.4, 0.6, 0.8]).to_dict()
        
        # 计算全球年度趋势指标 - 这个仍然按年份分组
        yearly_stats = df.groupby('Year').agg({
            'GDP PPP/capita 2017': ['mean', 'std', 'median'],
            'Population': ['mean', 'std', 'median'],
            'GDP PPP 2017': ['mean', 'std', 'median']
        })
        
        # 存储全球统计指标
        self.global_stats = {
            'gdp_per_capita_quantiles': gdp_quantiles,
            'population_quantiles': population_quantiles,
            'yearly_stats': yearly_stats.to_dict(),
            'base_year_stats': yearly_stats.loc[self.base_year].to_dict() if self.base_year in yearly_stats.index else None
        }
            
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """使用已拟合的参数转换数据，生成特征
        
        Args:
            df: 输入数据

        Returns:
            处理后的DataFrame副本，包含所有生成的特征
        """
        self.validate_columns(df)
        df = df.copy()
        
        # 1. 基础指标的非线性特征
        for metric in ['GDP PPP 2017', 'GDP PPP/capita 2017', 'Population']:
            metric_name = metric.lower().replace(' ', '_').replace('/', '_per_')
            
            # 对数变换
            df[f'{metric_name}_log'] = np.log1p(df[metric])
            
            # 二次项特征 - 可以捕捉倒U型关系
            df[f'{metric_name}_squared'] = np.square(df[f'{metric_name}_log'])
            
            # 三次项特征 - 更好地捕捉环境库兹涅茨曲线关系
            df[f'{metric_name}_cubic'] = np.power(df[f'{metric_name}_log'], 3)
            
            # 指数变换 - 捕捉快速增长阶段
            df[f'{metric_name}_exp'] = np.exp(df[f'{metric_name}_log'] / 10) - 1  # 除以10避免溢出
            
            # 增长率（基于历史数据）
            df[f'{metric_name}_growth'] = df.groupby('Country Name')[metric].pct_change().fillna(0)
            
            # 相对变化率（避免使用未来数据）
            df[f'{metric_name}_relative_change'] = df.groupby('Country Name')[metric].transform(
                lambda x: (x - x.expanding().mean()) / x.expanding().std()
            ).fillna(0)
            
            # 库兹涅茨曲线特征 - 针对人均GDP特别有意义
            if 'capita' in metric_name:
                # 倒U型变换 - 模拟环境库兹涅茨曲线
                x = df[f'{metric_name}_log']
                peak_point = x.quantile(0.7)  # 假设在70%分位点达到峰值
                df[f'{metric_name}_ekc'] = -(x - peak_point) ** 2
                
                # 分段线性特征 - 捕捉不同发展阶段的不同影响
                df[f'{metric_name}_early_stage'] = np.where(x < x.quantile(0.3), x, x.quantile(0.3))
                df[f'{metric_name}_mid_stage'] = np.where((x >= x.quantile(0.3)) & (x < x.quantile(0.7)), 
                                                        x - x.quantile(0.3), 0)
                df[f'{metric_name}_late_stage'] = np.where(x >= x.quantile(0.7), 
                                                         x - x.quantile(0.7), 0)
        
        # 2. 收入组特征
        df['income_group_ordinal'] = df['Income Group'].map({
            'Low income': 1, 
            'Lower middle income': 2, 
            'Upper middle income': 3, 
            'High income': 4
        })

        # 区域序数编码
        # 根据区域的经济发展水平进行编码
        df['region_ordinal'] = df['Region'].map({
            'Sub-Saharan Africa': 1,
            'South Asia': 2,
            'Middle East & North Africa': 3,
            'Latin America & Caribbean': 4,
            'East Asia & Pacific': 5,
            'Europe & Central Asia': 6,
            'North America': 7
        })

        # 3. 全球发展阶段特征
        # 使用预先计算的全球统计指标
        if self.global_stats:
            # 使用预先计算的GDP分位数
            gdp_quantiles = self.global_stats.get('gdp_per_capita_quantiles', {})
            if gdp_quantiles:
                df['global_economic_stage'] = pd.cut(
                    df['GDP PPP/capita 2017'],
                    bins=[-np.inf] + list(gdp_quantiles.values()) + [np.inf],
                    labels=range(5)
                ).fillna(0).astype(int)
            
            # 使用预先计算的人口分位数
            pop_quantiles = self.global_stats.get('population_quantiles', {})
            if pop_quantiles:
                df['global_population_stage'] = pd.cut(
                    df['Population'],
                    bins=[-np.inf] + list(pop_quantiles.values()) + [np.inf],
                    labels=range(5)
                ).fillna(0).astype(int)
            
            # 使用预先计算的年度趋势指标
            yearly_stats = self.global_stats.get('yearly_stats', {})
            
            # 计算与全球基准的相对位置
            for metric in ['GDP PPP/capita 2017', 'Population', 'GDP PPP 2017']:
                metric_name = metric.lower().replace(' ', '_').replace('/', '_per_')
                
                # 相对于全球年度中位数的位置
                if yearly_stats:
                    df[f'{metric_name}_global_position'] = df.apply(
                        lambda row: (row[metric] - yearly_stats[(metric, 'median')][row['Year']]) / \
                                   yearly_stats[(metric, 'std')][row['Year']] \
                        if row['Year'] in yearly_stats[(metric, 'median')] else 0,
                        axis=1
                    )
        else:
            # 如果没有预先计算的分位数，则按年份计算
            df['global_economic_stage'] = df.groupby('Year')['GDP PPP/capita 2017'].transform(
                lambda x: pd.qcut(x, 4, labels=False, duplicates='drop')
            ).fillna(0).astype(int)
            
            df['global_population_stage'] = df.groupby('Year')['Population'].transform(
                lambda x: pd.qcut(x, 4, labels=False, duplicates='drop')
            ).fillna(0).astype(int)
        
        # 发展速度指标（基于历史数据）
        for metric in ['GDP PPP/capita 2017', 'Population', 'GDP PPP 2017']:
            metric_name = metric.lower().replace(' ', '_').replace('/', '_per_')
            df[f'{metric_name}_momentum'] = df.groupby('Country Name')[metric].transform(
                lambda x: x.pct_change().rolling(3, min_periods=1).mean()
            ).fillna(0)
        
        # 区域和收入组特征
        # 区域内GDP占比
        df['region_gdp_share'] = df.groupby(['Region', 'Year'])['GDP PPP 2017'].transform(
            lambda x: x / x.sum()
        )
        
        # 收入组内GDP占比
        df['income_group_gdp_share'] = df.groupby(['Income Group', 'Year'])['GDP PPP 2017'].transform(
            lambda x: x / x.sum()
        )
        
        # 区域内人口占比
        df['region_population_share'] = df.groupby(['Region', 'Year'])['Population'].transform(
            lambda x: x / x.sum()
        )

        # 区域内人口占比
        df['region_population_share'] = df.groupby(['Region', 'Year'])['Population'].transform(
            lambda x: x / x.sum()
        )
        
        # 添加MSW与GDP相关的交互特征
        # GDP与收入组的交互特征
        df['gdp_income_interaction'] = df['GDP PPP/capita 2017'] * df['income_group_ordinal']
        
        # GDP与区域发展水平的交互特征
        df['gdp_region_interaction'] = df['GDP PPP/capita 2017'] * df['region_ordinal']
        
        # GDP增长与经济发展阶段的交互
        df['gdp_growth_stage_interaction'] = df['gdp_ppp_per_capita_2017_growth'] * df['global_economic_stage']
        
        # 人均GDP的立方项 (捕捉环境库兹涅茨曲线关系)
        df['gdp_per_capita_cubic'] = np.power(df['gdp_ppp_per_capita_2017_log'], 3)

        # 处理异常值和缺失值
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # 先将所有数值列转换为float64类型
        for col in numeric_cols:
            df[col] = df[col].astype(np.float64)
        
        # 直接使用fillna处理无穷值和NaN值
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # 对数值进行裁剪，确保在浮点数的有效范围内
        df[numeric_cols] = df[numeric_cols].apply(
            lambda x: np.clip(x, np.finfo(np.float64).min, np.finfo(np.float64).max)
        )
        
        return df
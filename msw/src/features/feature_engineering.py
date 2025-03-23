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
        self.high_error_regions = ['South Asia', 'Sub-Saharan Africa', 'Middle East & North Africa']
        self.region_stats = {}  # 存储区域统计指标
        
        
    def save_params(self, params_path: str) -> None:
        """保存特征工程参数到文件"""
        params = {
            'region_stats': self.region_stats  # 新增：保存区域统计指标
        }
        pd.to_pickle(params, params_path)
        
    def load_params(self, params_path: str) -> None:
        """从文件加载特征工程参数"""
        params = pd.read_pickle(params_path)
        self.region_stats = params.get('region_stats', {})  # 新增：加载区域统计指标

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
        """拟合训练数据并保存区域统计参数
        
        计算并保存区域级别的经济和人口发展趋势指标，使用分位数等统计方法避免数据泄露
        """
        # 计算并保存区域基准指标
        for region in df['Region'].unique():
            region_data = df[df['Region'] == region]
            
            # 计算区域经济发展阶段（使用分位数）
            gdp_quantiles = region_data['GDP PPP/capita 2017'].quantile([0.2, 0.4, 0.6, 0.8]).to_dict()
            population_quantiles = region_data['Population'].quantile([0.2, 0.4, 0.6, 0.8]).to_dict()
            
            # 计算区域年度趋势指标
            yearly_stats = region_data.groupby('Year').agg({
                'GDP PPP/capita 2017': ['mean', 'std', 'median'],
                'Population': ['mean', 'std', 'median'],
                'GDP PPP 2017': ['mean', 'std', 'median']
            })
            
            # 存储区域统计指标
            self.region_stats[region] = {
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
            
            # 二次项特征
            df[f'{metric_name}_squared'] = np.square(df[f'{metric_name}_log'])
            
            # 增长率（基于历史数据）
            df[f'{metric_name}_growth'] = df.groupby('Country Name')[metric].pct_change().fillna(0)
            
            # 相对变化率（避免使用未来数据）
            df[f'{metric_name}_relative_change'] = df.groupby('Country Name')[metric].transform(
                lambda x: (x - x.expanding().mean()) / x.expanding().std()
            ).fillna(0)
        
        # 2. 收入组特征
        df['income_group_ordinal'] = df['Income Group'].map({
            'Low income': 1, 
            'Lower middle income': 2, 
            'Upper middle income': 3, 
            'High income': 4
        })
        
        # 3. 区域发展阶段特征
        # 为所有区域计算基本统计指标
        for region in df['Region'].unique():
            region_data = df[df['Region'] == region]
            region_mask = (df['Region'] == region)
            
            # 使用预先计算的统计指标
            region_stats = self.region_stats.get(region, {})
            if region_stats:
                # 使用预先计算的GDP分位数
                gdp_quantiles = region_stats.get('gdp_per_capita_quantiles', {})
                if gdp_quantiles:
                    df.loc[region_mask, 'economic_stage'] = pd.cut(
                        df.loc[region_mask, 'GDP PPP/capita 2017'],
                        bins=[-np.inf] + list(gdp_quantiles.values()) + [np.inf],
                        labels=range(5)
                    ).fillna(-1)
                
                # 使用预先计算的人口分位数
                pop_quantiles = region_stats.get('population_quantiles', {})
                if pop_quantiles:
                    df.loc[region_mask, 'population_stage'] = pd.cut(
                        df.loc[region_mask, 'Population'],
                        bins=[-np.inf] + list(pop_quantiles.values()) + [np.inf],
                        labels=range(5)
                    ).fillna(-1)
                
            # 使用预先计算的年度趋势指标
            yearly_stats = region_stats.get('yearly_stats', {})
            
            # 计算与区域基准的相对位置
            for metric in ['GDP PPP/capita 2017', 'Population', 'GDP PPP 2017']:
                metric_name = metric.lower().replace(' ', '_').replace('/', '_per_')
                
                # 相对于区域年度中位数的位置
                if yearly_stats:
                    df.loc[region_mask, f'{metric_name}_relative_position'] = \
                        df.loc[region_mask].apply(
                            lambda row: (row[metric] - yearly_stats[(metric, 'median')][row['Year']]) / \
                                       yearly_stats[(metric, 'std')][row['Year']] \
                            if row['Year'] in yearly_stats[(metric, 'median')] else 0,
                            axis=1
                        )
                
                # 发展速度指标（基于历史数据）
                df.loc[region_mask, f'{metric_name}_momentum'] = \
                    df.loc[region_mask].groupby('Country Name')[metric].transform(
                        lambda x: x.pct_change().rolling(3, min_periods=1).mean()
                    ).fillna(0)
                
            # 为高误差区域添加额外的特征
            if region in self.high_error_regions:
                region_stats = self.region_stats.get(region, {})
                if region_stats:
                    # 使用预先计算的统计数据进行更精细的特征工程
                    df.loc[region_mask, 'high_error_economic_volatility'] = \
                        df.loc[region_mask].groupby('Country Name')['GDP PPP/capita 2017'].transform(
                            lambda x: x.rolling(5, min_periods=2).std() / x.rolling(5, min_periods=2).mean()
                        ).fillna(0)
                    
                    df.loc[region_mask, 'high_error_population_growth_stability'] = \
                        df.loc[region_mask].groupby('Country Name')['Population'].transform(
                            lambda x: 1 / (1 + x.pct_change().rolling(3, min_periods=2).std())
                        ).fillna(0)
        
        # 4. 高误差区域特定特征
        df['high_error_region'] = df['Region'].isin(self.high_error_regions).astype(int)
        
        # 时间趋势特征
        df['high_error_region_time'] = df['high_error_region'] * (df['Year'] - self.base_year)
        df['high_error_region_time_squared'] = df['high_error_region_time'] ** 2
        
        # 发展阶段转换特征
        df['high_error_development_phase'] = df['high_error_region'] * (
            df.groupby('Region')['GDP PPP/capita 2017'].transform(
                lambda x: pd.qcut(x, q=4, labels=False, duplicates='drop')
            )
        ).fillna(-1)
        
        # 区域稳定性指标
        df['high_error_region_stability'] = df['high_error_region'] * (
            1 / (1 + df.groupby('Country Name')['GDP PPP/capita 2017'].transform(
                lambda x: x.rolling(5, min_periods=2).std() / x.rolling(5, min_periods=2).mean()
            ))
        ).fillna(0)
        
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
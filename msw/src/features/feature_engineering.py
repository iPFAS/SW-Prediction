import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Dict, Optional, Tuple, Set
from ..config.config import Config

class FeatureEngineering:
    def __init__(self):
        """初始化特征工程类"""
        self.required_columns = Config.FEATURE_CONFIG['usecols']
        self.base_year = Config.FEATURE_CONFIG.get('base_year', 2015)  # 添加缺失的base_year
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
        """拟合训练数据并保存统计参数"""
        # 计算并保存区域基准指标
        for region in self.high_error_regions:
            region_name = region.lower().replace(' & ', '_').replace(' ', '_')
            region_data = df[df['Region'] == region]
            
            # 计算区域基准统计量
            self.region_stats[region] = {
                'gdp_per_capita_mean': region_data.groupby('Year')['GDP PPP/capita 2017'].mean().to_dict(),
                'gdp_per_capita_std': region_data.groupby('Year')['GDP PPP/capita 2017'].std().to_dict(),
                'inequality_index': (
                    region_data.groupby('Year')['GDP PPP/capita 2017'].agg(['std', 'mean'])
                    .apply(lambda x: x['std'] / x['mean'], axis=1)
                    .to_dict()
                ),
                'growth_rate': region_data.groupby('Year')['GDP PPP/capita 2017'].mean().pct_change().to_dict(),
                'global_position': (
                    df.groupby('Year')['GDP PPP/capita 2017'].median().to_dict()
                )
            }
            
            # 计算区域发展阶段基准
            self.region_stats[region]['development_quantiles'] = (
                region_data['GDP PPP/capita 2017'].quantile(q=[0.25, 0.5, 0.75]).to_dict()
            )
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """使用已拟合的参数转换数据

        Args:
            df: 输入数据

        Returns:
            处理后的DataFrame副本
        """
        self.validate_columns(df)
        df = df.copy()
        
        # 按国家分组进行特征计算
        df = df.sort_values(['Country Name', 'Year'])
        
        # 新增GDP/capita增长率计算
        df['gdp_per_capita_growth'] = df.groupby('Country Name')['GDP PPP/capita 2017'].pct_change().fillna(0)
        
        # 添加区域经济发展阶段加权指标
        region_economic_level = df.groupby('Region')['GDP PPP/capita 2017'].transform(
            lambda x: np.log1p(x.rolling(5, min_periods=1).mean())
        )
        df['region_economic_weight'] = region_economic_level / region_economic_level.max()
        
        # 计算每个国家自身的GDP和人口趋势特征
        for metric in ['GDP PPP 2017', 'GDP PPP/capita 2017', 'Population']:
            metric_name = metric.lower().replace(' ', '_')
            
            # 基础特征变换
            df[f'{metric_name}_log'] = np.log1p(df[metric])
            df[f'{metric_name}_squared'] = np.square(df[metric])
            
            # 计算增长率（只考虑国家自身的历史数据）
            df[f'{metric_name}_growth'] = df.groupby('Country Name')[metric].transform(
                lambda x: x.pct_change().shift(1)
            ).fillna(0)
            
            # 计算相对于国家自身历史的统计特征
            df[f'{metric_name}_rolling_mean_3y'] = df.groupby('Country Name')[metric].transform(
                lambda x: x.rolling(window=3, min_periods=1).mean()
            )
            df[f'{metric_name}_rolling_std_3y'] = df.groupby('Country Name')[metric].transform(
                lambda x: x.rolling(window=3, min_periods=1).std()
            )
            
            # 计算与历史均值的偏差
            df[f'{metric_name}_deviation'] = df.groupby('Country Name')[metric].transform(
                lambda x: (x - x.expanding().mean()) / x.expanding().std()
            ).fillna(0)
        
        # 计算人均GDP的二阶导数（加速度）
        df['gdp_per_capita_acceleration'] = df.groupby('Country Name')['gdp_per_capita_growth'].transform(
            lambda x: x.pct_change().shift(1)
        ).fillna(0)

        # 1. 添加South Asia特定特征
        df['is_south_asia'] = df['Region'].apply(lambda x: 1 if x == 'South Asia' else 0)
        df['south_asia_gdp_interaction'] = df['is_south_asia'] * df['GDP PPP/capita 2017']
        df['south_asia_population_density'] = df['is_south_asia'] * (df['Population'] / df.groupby('Country Name')['Population'].transform('mean'))
        
        # 2. 添加收入组序数特征
        df['income_group_ordinal'] = df['Income Group'].map({
            'Low income': 1, 
            'Lower middle income': 2, 
            'Upper middle income': 3, 
            'High income': 4
        })
        
        # 3. 添加收入组与GDP交互特征
        df['income_gdp_interaction'] = df['income_group_ordinal'] * np.log1p(df['GDP PPP/capita 2017'])
        
        # 4. 添加区域-收入组交叉特征
        df['region_income_interaction'] = df.apply(
            lambda row: f"{row['Region']}_{row['Income Group']}", axis=1
        ).astype('category').cat.codes
        
        # 5. 针对高误差区域的特征
        df['high_error_region'] = df['Region'].apply(lambda x: 1 if x in self.high_error_regions else 0)
        # 时间趋势特征增强
        df['high_error_region_time_trend'] = df['high_error_region'] * (df['Year'] - self.base_year)
        df['high_error_region_time_squared'] = df['high_error_region_time_trend'] ** 2  # 添加二次项

        # 区域内部结构特征
        for region in self.high_error_regions:
            region_name = region.lower().replace(' & ', '_').replace(' ', '_')
            df[f'{region_name}_flag'] = (df['Region'] == region).astype(int)
            
            # 使用基准统计量计算特征
            base_stats = self.region_stats.get(region, {})
            
            # 1. 区域内部发展不平衡指数（使用基准值）
            df[f'{region_name}_inequality_index'] = df[f'{region_name}_flag'] * df['Year'].map(
                base_stats.get('inequality_index', {})
            ).fillna(df[f'{region_name}_flag'] * base_stats.get('inequality_index', {}).get(self.base_year, 0))
            
            # 2. 区域内部相对位置（基于历史分位数）
            df[f'{region_name}_relative_position'] = df[f'{region_name}_flag'] * (
                pd.qcut(df[df['Region'] == region]['GDP PPP/capita 2017'], 
                       q=10, labels=False, duplicates='drop')
            ).fillna(-1)
            
            # 3. 区域发展速度差异（使用滚动统计）
            df[f'{region_name}_growth_deviation'] = df[f'{region_name}_flag'] * (
                df.groupby('Country Name')['GDP PPP/capita 2017'].pct_change() -
                df['Year'].map(base_stats.get('growth_rate', {})).fillna(0)
            )
            
            # 4. 区域经济结构转型指标（使用滚动窗口）
            df[f'{region_name}_transformation_index'] = df[f'{region_name}_flag'] * (
                df.groupby('Country Name')['GDP PPP/capita 2017'].transform(
                    lambda x: x.rolling(5, min_periods=1).mean() / x.rolling(5, min_periods=1).std()
                )
            ).fillna(0)
            
            # 5. 区域追赶指数（相对于全球中位数）
            df[f'{region_name}_catchup_index'] = df[f'{region_name}_flag'] * (
                df['GDP PPP/capita 2017'] / df['Year'].map(base_stats.get('global_position', {})).fillna(1) - 1
            )
            
            # 6. 区域韧性指标（基于历史最大值）
            df[f'{region_name}_resilience'] = df[f'{region_name}_flag'] * (
                df.groupby('Country Name')['GDP PPP/capita 2017'].transform(
                    lambda x: x / x.expanding().max()
                )
            ).fillna(1)
            
            # 7. 区域发展稳定性（使用扩展窗口）
            df[f'{region_name}_stability'] = df[f'{region_name}_flag'] * (
                df.groupby('Country Name')['GDP PPP/capita 2017'].transform(
                    lambda x: 1 / (1 + x.expanding().std() / x.expanding().mean())
                )
            ).fillna(0)
            
            # 8. 区域发展阶段（基于固定分位数）
            quantiles = base_stats.get('development_quantiles', {})
            df[f'{region_name}_development_stage'] = df[f'{region_name}_flag'] * pd.cut(
                df['GDP PPP/capita 2017'],
                bins=[-np.inf] + list(quantiles.values()) + [np.inf],
                labels=False
            ).fillna(-1)
            
        # 6. 收入组时间趋势特征
        for income_group in ['Upper middle income', 'Low income', 'Lower middle income']:
            col_name = income_group.lower().replace(' ', '_')
            df[f'{col_name}_flag'] = (df['Income Group'] == income_group).astype(int)
            df[f'{col_name}_gdp_trend'] = df[f'{col_name}_flag'] * df.groupby('Country Name')['GDP PPP/capita 2017'].transform(
                lambda x: x.pct_change().rolling(3, min_periods=1).mean()
            ).fillna(0)
        
        # 7. 增强区域经济发展阶段特征
        df['region_development_stage'] = df.groupby('Region')['GDP PPP/capita 2017'].transform(
            lambda x: pd.qcut(x, q=5, labels=False, duplicates='drop')
        ).fillna(-1)
        
        # 8. 添加人口密度相关特征
        df['population_density_trend'] = df.groupby('Country Name')['Population'].transform(
            lambda x: x.diff() / x.shift(1)
        ).fillna(0)
        
        # 9. GDP波动性特征
        df['gdp_volatility'] = df.groupby('Country Name')['GDP PPP 2017'].transform(
            lambda x: x.rolling(5, min_periods=2).std() / x.rolling(5, min_periods=2).mean()
        ).fillna(0)
        

        # 10.区域发展阶段转换特征
        df['high_error_region_development_phase'] = df['high_error_region'] * (
            df.groupby('Region')['GDP PPP/capita 2017'].transform(
                lambda x: pd.qcut(x, q=4, labels=False, duplicates='drop')
            )
        ).fillna(-1)

        # 区域稳定性指标
        df['high_error_region_stability'] = df['high_error_region'] * (
            1 / (1 + df.groupby('Country Name')['GDP PPP/capita 2017'].transform(
                lambda x: x.rolling(5, min_periods=2).std() / x.rolling(5, min_periods=2).mean()
            ))
        )

        # 区域间相对发展水平
        df['high_error_region_relative_development'] = df['high_error_region'] * (
            df['GDP PPP/capita 2017'] / df.groupby('Year')['GDP PPP/capita 2017'].transform('mean')
        )

        # 处理无限值和异常值
        df = df.replace([np.inf, -np.inf], np.nan)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].apply(lambda x: np.clip(x, np.finfo(np.float64).min, np.finfo(np.float64).max))

        # 填充缺失值
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        return df
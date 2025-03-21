import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Dict, Optional, Tuple, Set
from ..config.config import Config

class FeatureEngineering:
    def __init__(self):
        """初始化特征工程类"""
        self.base_year = Config.FEATURE_CONFIG['base_year']
        self.required_columns = Config.FEATURE_CONFIG['usecols']
        
    def save_params(self, params_path: str):
        """保存特征工程参数到文件"""
        # 参数完整性验证
        required_keys = ['group_stats', 'country_trend_params']
        missing_keys = [k for k in required_keys if not hasattr(self, k)]
        if missing_keys:
            raise ValueError(f"缺失关键参数: {missing_keys}")

        # 记录参数结构信息
        param_stats = {
            'group_stats_keys': [k for k in self.group_stats.keys()],
            'country_count': len(self.country_trend_params),
            'group_stats_entries': sum(len(v) for v in self.group_stats.values())
        }
        print(f"保存参数摘要: {param_stats}")

        params = {
            'group_stats': self.group_stats,
            'country_trend_params': self.country_trend_params
        }
        pd.to_pickle(params, params_path)
        
    def load_params(self, params_path: str):
        """从文件加载特征工程参数"""
        params = pd.read_pickle(params_path)
        
        # 参数完整性检查
        required_keys = ['group_stats', 'country_trend_params']
        missing_keys = [k for k in required_keys if k not in params]
        if missing_keys:
            raise ValueError(f"参数文件缺失关键字段: {missing_keys}")

        # 记录加载参数摘要
        param_stats = {
            'loaded_group_stats_keys': [k for k in params['group_stats'].keys()],
            'country_count': len(params['country_trend_params']),
            'group_stats_entries': sum(len(v) for v in params['group_stats'].values())
        }
        print(f"加载参数摘要: {param_stats}")

        self.group_stats = params['group_stats']
        self.country_trend_params = params['country_trend_params']
        
        # 验证参数类型
        if not isinstance(self.group_stats, dict):
            raise TypeError("group_stats参数类型错误，应为字典")
        if not isinstance(self.country_trend_params, dict):
            raise TypeError("country_trend_params参数类型错误，应为字典")

    def validate_columns(self, df: pd.DataFrame) -> None:
        """验证输入数据是否包含所需列"""
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def transform_target(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, str]:
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

    def fit(self, df: pd.DataFrame):
        """拟合训练数据并保存统计参数"""
        # 计算收入组的基本统计特征
                # 新增国家趋势参数计算
        # 计算区域和收入组的长期基准参数
        group_stats = {}
        for group_type in ['Region', 'Income Group']:
            stats = df.groupby([group_type, 'Year']).agg(
                gdp_mean=('GDP PPP 2017', 'mean'),
                gdp_median=('GDP PPP 2017', 'median'),
                gdp_std=('GDP PPP 2017', 'std'),
                gdp_per_capita_mean=('GDP PPP/capita 2017', 'mean'),
                gdp_per_capita_median=('GDP PPP/capita 2017', 'median'),
                gdp_per_capita_std=('GDP PPP/capita 2017', 'std'),
                population_mean=('Population', 'mean'),
                population_median=('Population', 'median'),
                population_std=('Population', 'std')
            ).reset_index()
            
            # 计算历史基准（使用5年窗口均值）
            stats['gdp_5y_ma'] = stats.groupby(group_type)['gdp_mean'].transform(lambda x: x.rolling(5, min_periods=1).mean())
            stats['gdp_per_capita_5y_ma'] = stats.groupby(group_type)['gdp_per_capita_mean'].transform(lambda x: x.rolling(5, min_periods=1).mean())
            stats['population_5y_ma'] = stats.groupby(group_type)['population_mean'].transform(lambda x: x.rolling(5, min_periods=1).mean())
            
            group_stats[group_type] = stats.set_index([group_type, 'Year']).to_dict()
        
        # 动态获取组统计参数并设置默认值
        self.group_stats = group_stats
        
        # 计算国家级的GDP/capita趋势参数（确保不包含未来年份）
        self.country_trend_params = {
            country: np.polyfit(
                country_df[country_df['Year'] <= self.base_year]['Year'] - self.base_year,
                country_df[country_df['Year'] <= self.base_year]['GDP PPP/capita 2017'], 1
            )[0] if len(country_df[country_df['Year'] <= self.base_year]) > 1 else 0.0
            for country, country_df in df.groupby('Country Name')
        }

        # 新增GDP/capita增长率计算
        df['gdp_per_capita_growth'] = df.groupby('Country Name')['GDP PPP/capita 2017'].pct_change().fillna(0)
        
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
        
        # 计算动态组特征与预存统计参数的交互项
        for group_col in ['Region', 'Income Group']:
            group_name = group_col.lower().replace(' ', '_')
            
            for metric in ['GDP PPP 2017', 'GDP PPP/capita 2017', 'Population']:
                metric_name = metric.lower().replace(' ', '_')
                metric_map = {
                    'GDP PPP 2017': 'gdp',
                    'GDP PPP/capita 2017': 'gdp_per_capita',
                    'Population': 'population'
                }
                
                # 创建临时索引列用于对齐
                df['_temp_index'] = df.index
                
                # 获取统计量并保持索引对齐
                stats_df = df.groupby([group_col, 'Year']).apply(
                    lambda g: pd.Series({
                        'stat_mean': self.group_stats.get(group_col, {})
                                       .get((g.name[0], g.name[1]), {})
                                       .get(f'{metric_map[metric]}_mean', 1)
                    })
                ).reset_index()
                
                # 合并统计量到原始数据
                df = pd.merge(df, stats_df, on=[group_col, 'Year'], how='left')
                ma_ratio = df[metric] / (df['stat_mean'] + 1e-6)
                df[f'{metric_name}_{group_name}_ma_ratio'] = np.log1p(ma_ratio)
                
                # 清理临时列
                df = df.drop(columns=['_temp_index', 'stat_mean'])
                # ========================

                # 保留标准化偏差特征
                df[f'{metric_name}_{group_name}_deviation'] = df.groupby(group_col)[metric].transform(
                    lambda x: (x - x.mean()) / x.std()
                ).fillna(0)

        
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
        high_error_regions = ['South Asia', 'Sub-Saharan Africa', 'Middle East & North Africa']
        df['high_error_region'] = df['Region'].apply(lambda x: 1 if x in high_error_regions else 0)
        df['high_error_region_time_trend'] = df['high_error_region'] * (df['Year'] - df['Year'].min())
        
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
        
        # 10. 增强现有MA特征的稳定性
        for col in df.filter(regex='_ma_ratio$').columns:
            df[f'{col}_smoothed'] = df.groupby('Country Name')[col].transform(
                lambda x: x.rolling(3, min_periods=1).mean()
            )
        
        # 处理无限值和异常值
        df = df.replace([np.inf, -np.inf], np.nan)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].apply(lambda x: np.clip(x, np.finfo(np.float64).min, np.finfo(np.float64).max))

        # 填充缺失值
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        return df
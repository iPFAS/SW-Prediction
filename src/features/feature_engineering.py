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
        self.development_stage_bins = None
        self.similar_countries_ref = None
        
    def save_params(self, params_path: str):
        """保存特征工程参数到文件"""
        params = {
            'development_stage_bins': self.development_stage_bins,
            'similar_countries_ref': self.similar_countries_ref
        }
        pd.to_pickle(params, params_path)
        
    def load_params(self, params_path: str):
        """从文件加载特征工程参数"""
        params = pd.read_pickle(params_path)
        self.development_stage_bins = params['development_stage_bins']
        self.similar_countries_ref = params['similar_countries_ref']

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
        # 计算发展阶段的全局分位数
        self.development_stage_bins = pd.qcut(
            df['GDP PPP/capita 2017'], 
            q=5, 
            retbins=True,
            duplicates='drop'
        )[1]
        
        # 计算区域平均GDP
        df['development_stage'] = pd.cut(
            df['GDP PPP/capita 2017'], 
            bins=self.development_stage_bins, 
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        )
        
        region_stats = df.groupby(['Region', 'Year']).agg({
            'GDP PPP 2017': 'mean',
            'GDP PPP/capita 2017': 'mean'
        }).reset_index()
        
        self.region_stats = region_stats.set_index(['Region', 'Year']).to_dict()
        
        # 预计算所有国家的GDP增长率
        gdp_growth_rates = df.groupby('Country Name', group_keys=True)['GDP PPP 2017'].apply(
            lambda x: np.clip(x.pct_change(), -0.5, 0.5)
        ).fillna(0)

        # 按年份和发展阶段分组预处理数据
        df_grouped = df.groupby(['Year', 'development_stage', 'Region'])
        similar_countries_data = []

        # 使用向量化操作计算相似国家
        for (year, stage, region), group in df_grouped:
            countries = group['Country Name'].unique()
            gdp_values = group['GDP PPP/capita 2017'].values
            
            for i, country in enumerate(countries):
                country_gdp = gdp_values[i]
                
                # 使用numpy操作计算GDP比率
                gdp_ratios = gdp_values / country_gdp
                similar_mask = (gdp_ratios >= 0.5) & (gdp_ratios <= 2.0)
                
                # 获取相似国家
                similar_countries = countries[similar_mask & (countries != country)]
                
                if len(similar_countries) > 0:
                    # 计算相似国家的平均GDP增长率
                    similar_growth_rates = gdp_growth_rates[gdp_growth_rates.index.get_level_values('Country Name').isin(similar_countries)]
                    similar_gdp_growth = similar_growth_rates.mean()
                    
                    # 如果没有有效的增长率，使用全局平均值
                    if pd.isna(similar_gdp_growth):
                        similar_gdp_growth = gdp_growth_rates.mean()
                    
                    similar_countries_data.append({
                        'Year': year,
                        'Country': country,
                        'similar_gdp_growth': similar_gdp_growth
                    })
        
        # 将结果转换为字典格式
        self.similar_countries_ref = pd.DataFrame(similar_countries_data).set_index(['Year', 'Country']).to_dict()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """使用已拟合的参数转换数据

        Args:
            df: 输入数据

        Returns:
            处理后的DataFrame副本
        """
        self.validate_columns(df)
        df = df.copy()
        
        # 生成时间特征（基于配置的基准年份）
        # 按国家分组进行特征计算，避免数据泄露
        df = df.sort_values(['Country Name', 'Year'])
        
        # 计算基础时间趋势特征
        df['year_trend'] = np.clip(df['Year'] - self.base_year, 0, None)
        df['year_trend_squared'] = df['year_trend'] ** 2
        df['year_trend_log'] = np.log1p(np.clip(df['year_trend'], 0, None))
        
        # 计算GDP趋势特征
        df['gdp_5y_ma'] = df.groupby('Country Name')['GDP PPP 2017'].transform(
            lambda s: s.rolling(window=5, min_periods=1).mean().shift(1))
        df['gdp_10y_ma'] = df.groupby('Country Name')['GDP PPP 2017'].transform(
            lambda s: s.rolling(window=10, min_periods=1).mean().shift(1))
        df['gdp_growth_rate'] = df.groupby('Country Name')['GDP PPP 2017'].transform(
            lambda s: s.pct_change().shift(1))
        
        # 计算人口动态特征
        df['pop_growth_rate'] = df.groupby('Country Name')['Population'].transform(
            lambda s: s.pct_change().shift(1))
        df['pop_density_trend'] = df.groupby('Country Name')['Population'].transform(
            lambda s: s.rolling(window=5, min_periods=1).mean().shift(1))
        
        # 计算经济-人口交互特征
        df['gdp_pop_interaction'] = np.log1p(df['GDP PPP 2017']) * np.log1p(df['Population'])
        df['gdp_per_capita_growth'] = df.groupby('Country Name')['GDP PPP/capita 2017'].transform(
            lambda s: s.pct_change().shift(1))
        
        # 计算中长期历史趋势
        df['gdp_trend'] = df.groupby('Country Name')['GDP PPP 2017'].transform(
            lambda s: (s - s.shift(5)) / s.shift(5)).shift(1)
        df['pop_trend'] = df.groupby('Country Name')['Population'].transform(
            lambda s: (s - s.shift(5)) / s.shift(5)).shift(1)
        
        # 计算经济发展阶段特征
        df['gdp_per_capita_ma'] = df.groupby('Country Name')['GDP PPP/capita 2017'].transform(
            lambda s: s.rolling(window=5, min_periods=1).mean().shift(1))
        df['gdp_acceleration'] = df.groupby('Country Name')['GDP PPP 2017'].transform(
            lambda s: s.pct_change().diff().shift(1))
        
        # 计算发展阶段特征
        df['development_stage'] = pd.cut(df['GDP PPP/capita 2017'], 
                                       bins=self.development_stage_bins, 
                                       labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        
        # 计算区域经济特征
        df['region_avg_gdp'] = df.apply(lambda row: self.region_stats['GDP PPP 2017'].get((row['Region'], row['Year']), 0), axis=1)
        df['region_gdp_per_capita'] = df.apply(lambda row: self.region_stats['GDP PPP/capita 2017'].get((row['Region'], row['Year']), 0), axis=1)
        
        # 计算相似国家特征
        df['similar_gdp_growth'] = df.apply(lambda row: self.similar_countries_ref['similar_gdp_growth'].get((row['Year'], row['Country Name']), 0), axis=1)
        
        # 计算发展阶段动态权重特征（增加低收入组权重）
        df['stage_weight'] = df['development_stage'].map({'Very Low': 1.5, 'Low': 1.3, 'Medium': 1.0, 'High': 0.8, 'Very High': 0.7}).astype(float)
        df['weighted_gdp'] = df['GDP PPP 2017'] * df['stage_weight']
        df['weighted_pop'] = df['Population'] * df['stage_weight']
        
        # 计算发展阶段平均特征
        df['stage_avg_gdp_growth'] = df.groupby(['development_stage', 'Year'])['GDP PPP 2017'].transform(lambda s: s.pct_change().mean()).shift(1)
        
        # 添加区域和发展阶段的交互特征
        df['region_stage_gdp'] = df.groupby(['Region', 'development_stage', 'Year'])['GDP PPP 2017'].transform('mean')
        df['region_stage_pop'] = df.groupby(['Region', 'development_stage', 'Year'])['Population'].transform('mean')
        df['region_stage_gdp_growth'] = df.groupby(['Region', 'development_stage'])['GDP PPP 2017'].transform(lambda x: x.pct_change().mean())
        
        # 添加区域特定的发展阶段权重
        region_weights = {
            'South Asia': {'Very Low': 1.8, 'Low': 1.5, 'Medium': 1.2, 'High': 0.9, 'Very High': 0.7},
            'Sub-Saharan Africa': {'Very Low': 1.7, 'Low': 1.4, 'Medium': 1.1, 'High': 0.9, 'Very High': 0.7}
        }
        df['region_stage_weight'] = df.apply(lambda row: region_weights.get(row['Region'], {}).get(row['development_stage'], row['stage_weight']), axis=1)
        df['region_weighted_gdp'] = df['GDP PPP 2017'] * df['region_stage_weight']
        
        # 填充缺失值
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        return df
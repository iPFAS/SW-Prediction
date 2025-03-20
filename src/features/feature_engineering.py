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
        
        # 计算相似国家参考基准
        # 对每个年份和国家，选取人均GDP在±20%范围内的国家作为参照组
        similar_countries_data = []
        for year in df['Year'].unique():
            year_data = df[df['Year'] == year]
            for country in year_data['Country Name'].unique():
                country_gdp = year_data[year_data['Country Name'] == country]['GDP PPP/capita 2017'].iloc[0]
                # 计算GDP范围
                gdp_lower = country_gdp * 0.8
                gdp_upper = country_gdp * 1.2
                # 筛选相似国家
                similar_countries = year_data[
                    (year_data['GDP PPP/capita 2017'] >= gdp_lower) &
                    (year_data['GDP PPP/capita 2017'] <= gdp_upper) &
                    (year_data['Country Name'] != country)
                ]
                if not similar_countries.empty:
                    # 计算相似国家的GDP增长率均值
                    similar_gdp_growth = similar_countries.groupby('Country Name')['GDP PPP 2017'].pct_change().mean()
                    similar_countries_data.append({
                        'Year': year,
                        'Country': country,
                        'similar_gdp_growth': similar_gdp_growth
                    })
        
        # 将结果转换为字典格式
        self.similar_countries_ref = pd.DataFrame(similar_countries_data).set_index(['Year', 'Country'])['similar_gdp_growth'].to_dict()

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
        
        df = df.assign(
            # 基础时间趋势特征（带1年滞后）
            # year_trend: 线性时间趋势，基准年份后的年份差，反映时间效应，用于捕捉长期发展趋势
            # year_trend_squared: 二次时间趋势，捕捉非线性变化，体现加速或减速效应，用于识别发展阶段转换
            # year_trend_log: 对数化时间趋势，降低量纲影响，平滑长期趋势，适合处理长期增长模式
            year_trend = np.clip(df['Year'] - self.base_year, 0, None),
            year_trend_squared = lambda x: x.year_trend ** 2,
            year_trend_log = lambda x: np.log1p(np.clip(x.year_trend, 0, None)),
            
            # GDP趋势特征（国家维度分组计算，考虑时间滞后避免数据泄露）
            # gdp_5y_ma: 5年移动平均，通过计算过去5年的GDP均值，平滑短期波动，突出中期趋势
            # gdp_10y_ma: 10年移动平均，计算过去10年GDP均值，识别长期经济周期和结构性变化
            # gdp_growth_rate: 年增长率，计算年度GDP变化率，衡量经济活跃度和发展动能
            gdp_5y_ma = lambda x: x.groupby('Country Name')['GDP PPP 2017'].transform(
                lambda s: s.rolling(window=5, min_periods=1).mean().shift(1)),
            gdp_10y_ma = lambda x: x.groupby('Country Name')['GDP PPP 2017'].transform(
                lambda s: s.rolling(window=10, min_periods=1).mean().shift(1)),
            gdp_growth_rate = lambda x: x.groupby('Country Name')['GDP PPP 2017'].transform(
                lambda s: s.pct_change().shift(1)),
            
            # 人口动态特征（国家维度分组计算，反映人口结构变迁）
            # pop_growth_rate: 人口年增长率，计算年度人口变化百分比，反映人口变化速度和人口红利
            # pop_density_trend: 5年人口密度趋势，计算5年移动平均，体现城市化和人口集聚进程
            pop_growth_rate = lambda x: x.groupby('Country Name')['Population'].transform(
                lambda s: s.pct_change().shift(1)),
            pop_density_trend = lambda x: x.groupby('Country Name')['Population'].transform(
                lambda s: s.rolling(window=5, min_periods=1).mean().shift(1)),
            
            # 经济-人口交互特征（探索经济和人口的协同效应）
            # gdp_pop_interaction: GDP与人口的对数乘积，通过对数变换降低量纲差异，衡量综合规模效应
            # gdp_per_capita_growth: 人均GDP增长率，计算人均GDP年度变化，反映生活水平提升速度
            gdp_pop_interaction = lambda x: np.log1p(x['GDP PPP 2017']) * np.log1p(x['Population']),
            gdp_per_capita_growth = lambda x: x.groupby('Country Name')['GDP PPP/capita 2017'].transform(
                lambda s: s.pct_change().shift(1)),
            # 中长期历史趋势（5年跨度，反映结构性变化）
            # gdp_trend: 5年GDP变化率，计算5年期GDP总体变化百分比，反映经济周期和结构转型
            # pop_trend: 5年人口变化率，计算5年期人口总体变化百分比，反映人口结构转变趋势
            gdp_trend = lambda x: x.groupby('Country Name')['GDP PPP 2017'].transform(
                lambda s: (s - s.shift(5)) / s.shift(5)).shift(1),
            pop_trend = lambda x: x.groupby('Country Name')['Population'].transform(
                lambda s: (s - s.shift(5)) / s.shift(5)).shift(1),
            # 经济发展阶段特征（刻画经济发展的质变过程）
            # gdp_per_capita_ma: 人均GDP 5年移动平均，平滑短期波动，反映实际发展水平和生活质量
            # gdp_acceleration: GDP增长率的变化率，计算GDP增速的二阶导数，捕捉经济转折点
            gdp_per_capita_ma = lambda x: x.groupby('Country Name')['GDP PPP/capita 2017'].transform(
                lambda s: s.rolling(window=5, min_periods=1).mean().shift(1)),
            gdp_acceleration = lambda x: x.groupby('Country Name')['GDP PPP 2017'].transform(
                lambda s: s.pct_change().diff().shift(1)),
            # 相似国家特征（基于经济发展水平的参照系）
            # 使用fit阶段计算的similar_countries_ref作为参考基准
            # similar_countries_gdp_growth = lambda x: x.apply(lambda row: self.similar_countries_ref.get((row['Year'], row['Country Name']), 0), axis=1),
            # 发展水平分组特征（基于人均GDP的五档分位数分组）
            development_stage = lambda x: pd.cut(x['GDP PPP/capita 2017'], bins=self.development_stage_bins, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']),
            stage_avg_gdp_growth = lambda x: x.groupby(['development_stage', 'Year'])['GDP PPP 2017'].transform(
                lambda s: s.pct_change().mean()).shift(1)
        )
        
        # 填充缺失值
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        return df
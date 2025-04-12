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
        
        计算并保存全球经济和人口发展趋势指标，考虑区域和收入组差异
        """
        # 检查必要的分组列是否存在
        group_cols = ['Region', 'Income Group']
        for col in group_cols:
            if col not in df.columns:
                print(f"警告: 列 '{col}' 不存在，部分特征将无法生成")
        
        # 1. 计算全球基本统计量
        global_stats = {
            'gdp_pc_mean': df['GDP PPP/capita 2017'].mean(),
            'gdp_pc_median': df['GDP PPP/capita 2017'].median(),
            'gdp_pc_std': df['GDP PPP/capita 2017'].std(),
            'gdp_pc_min': df['GDP PPP/capita 2017'].min(),
            'gdp_pc_max': df['GDP PPP/capita 2017'].max(),
            'population_mean': df['Population'].mean(),
            'population_median': df['Population'].median(),
            'population_std': df['Population'].std(),
            'gdp_mean': df['GDP PPP 2017'].mean(),
            'gdp_median': df['GDP PPP 2017'].median(),
            'gdp_std': df['GDP PPP 2017'].std()
        }
        
        # 2. 按收入组计算统计量
        income_group_stats = {}
        if 'Income Group' in df.columns:
            for income_group in df['Income Group'].unique():
                group_data = df[df['Income Group'] == income_group]
                
                # 计算该收入组的统计量
                income_group_stats[income_group] = {
                    'gdp_pc_mean': group_data['GDP PPP/capita 2017'].mean(),
                    'gdp_pc_median': group_data['GDP PPP/capita 2017'].median(),
                    'gdp_pc_std': group_data['GDP PPP/capita 2017'].std(),
                    'gdp_pc_min': group_data['GDP PPP/capita 2017'].min(),
                    'gdp_pc_max': group_data['GDP PPP/capita 2017'].max(),
                    'population_mean': group_data['Population'].mean(),
                    'gdp_mean': group_data['GDP PPP 2017'].mean(),
                    'country_count': len(group_data['Country Name'].unique())
                }
                
                # 特别关注High income组，计算更多统计量
                if income_group == 'High income':
                    # 计算High income组的人均GDP分位数，用于确定发展阶段
                    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
                    for q in quantiles:
                        income_group_stats[income_group][f'gdp_pc_q{int(q*100)}'] = \
                            group_data['GDP PPP/capita 2017'].quantile(q)
                    
                    # 计算High income组的人均GDP峰值点（假设在75%分位点）
                    income_group_stats[income_group]['gdp_pc_peak'] = \
                        group_data['GDP PPP/capita 2017'].quantile(0.75)
        
        # 3. 按区域计算统计量
        region_stats = {}
        if 'Region' in df.columns:
            for region in df['Region'].unique():
                region_data = df[df['Region'] == region]
                
                region_stats[region] = {
                    'gdp_pc_mean': region_data['GDP PPP/capita 2017'].mean(),
                    'gdp_pc_median': region_data['GDP PPP/capita 2017'].median(),
                    'gdp_pc_std': region_data['GDP PPP/capita 2017'].std(),
                    'population_mean': region_data['Population'].mean(),
                    'gdp_mean': region_data['GDP PPP 2017'].mean(),
                    'country_count': len(region_data['Country Name'].unique())
                }
        
        # 4. 按区域+收入组计算统计量
        region_income_stats = {}
        if 'Region' in df.columns and 'Income Group' in df.columns:
            for region in df['Region'].unique():
                region_income_stats[region] = {}
                
                for income_group in df['Income Group'].unique():
                    group_data = df[(df['Region'] == region) & (df['Income Group'] == income_group)]
                    
                    if len(group_data) > 0:
                        region_income_stats[region][income_group] = {
                            'gdp_pc_mean': group_data['GDP PPP/capita 2017'].mean(),
                            'gdp_pc_median': group_data['GDP PPP/capita 2017'].median(),
                            'population_mean': group_data['Population'].mean(),
                            'gdp_mean': group_data['GDP PPP 2017'].mean(),
                            'country_count': len(group_data['Country Name'].unique())
                        }
        
        # 5. 按人口规模分组计算统计量
        # 将国家按人口规模分为大、中、小三类
        population_thresholds = df['Population'].quantile([0.33, 0.67]).tolist()
        
        population_size_stats = {
            '0': df[df['Population'] <= population_thresholds[0]]['GDP PPP/capita 2017'].describe().to_dict(),
            '1': df[(df['Population'] > population_thresholds[0]) & 
                         (df['Population'] <= population_thresholds[1])]['GDP PPP/capita 2017'].describe().to_dict(),
            '2': df[df['Population'] > population_thresholds[1]]['GDP PPP/capita 2017'].describe().to_dict()
        }
        
        # 6. 计算年度趋势
        yearly_stats = df.groupby('Year').agg({
            'GDP PPP/capita 2017': ['mean', 'median', 'std'],
            'Population': ['mean', 'sum'],
            'GDP PPP 2017': ['mean', 'sum']
        }).to_dict()
        
        # 7. 计算环境库兹涅茨曲线参数
        # 使用High income组的数据确定EKC峰值点
        ekc_params = {}
        if 'Income Group' in df.columns:
            high_income_data = df[df['Income Group'] == 'High income']
            if len(high_income_data) > 0:
                # 假设High income组的75%分位点是EKC峰值
                ekc_params['peak_point'] = high_income_data['GDP PPP/capita 2017'].quantile(0.75)
                ekc_params['early_threshold'] = high_income_data['GDP PPP/capita 2017'].quantile(0.25)
                ekc_params['late_threshold'] = high_income_data['GDP PPP/capita 2017'].quantile(0.75)
            else:
                # 如果没有High income数据，使用全局分位数
                ekc_params['peak_point'] = df['GDP PPP/capita 2017'].quantile(0.75)
                ekc_params['early_threshold'] = df['GDP PPP/capita 2017'].quantile(0.25)
                ekc_params['late_threshold'] = df['GDP PPP/capita 2017'].quantile(0.75)
        else:
            # 如果没有Income Group列，使用全局分位数
            ekc_params['peak_point'] = df['GDP PPP/capita 2017'].quantile(0.75)
            ekc_params['early_threshold'] = df['GDP PPP/capita 2017'].quantile(0.25)
            ekc_params['late_threshold'] = df['GDP PPP/capita 2017'].quantile(0.75)
        
        # 存储所有统计量
        self.global_stats = {
            'global_stats': global_stats,
            'income_group_stats': income_group_stats,
            'region_stats': region_stats,
            'region_income_stats': region_income_stats,
            'population_size_stats': population_size_stats,
            'yearly_stats': yearly_stats,
            'ekc_params': ekc_params,
            'population_thresholds': population_thresholds
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
        for metric in ['GDP PPP 2017', 'GDP PPP/capita 2017', 'Population','Urban population %']:
            metric_name = metric.lower().replace(' ', '_').replace('/', '_per_').replace('%', 'pct')
            
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
        
        # 2. 收入组相关特征
        if 'Income Group' in df.columns:
            # 使用预先计算的统计量（如果有）
            income_group_stats = self.global_stats.get('income_group_stats', {})
            
            # 相对于收入组的发展水平
            df['income_group_relative_gdp_pc'] = df.apply(
                lambda row: row['GDP PPP/capita 2017'] / income_group_stats.get(
                    row['Income Group'], {}).get('gdp_pc_mean', 1) 
                if row['Income Group'] in income_group_stats else 1,
                axis=1
            )
            
            # 相对于High income组的发展水平
            high_income_mean = income_group_stats.get('High income', {}).get('gdp_pc_mean', 1)
            df['high_income_relative_gdp_pc'] = df['GDP PPP/capita 2017'] / high_income_mean
            
            # 收入组内人口规模相对位置
            df['income_group_population_rank'] = df.groupby('Income Group')['Population'].rank(pct=True)
            
            # 收入组内GDP规模相对位置
            df['income_group_gdp_rank'] = df.groupby('Income Group')['GDP PPP 2017'].rank(pct=True)
        
        # 3. 区域相关特征
        if 'Region' in df.columns:
            # 使用预先计算的统计量（如果有）
            region_stats = self.global_stats.get('region_stats', {})
            
            # 相对于区域的发展水平
            df['region_relative_gdp_pc'] = df.apply(
                lambda row: row['GDP PPP/capita 2017'] / region_stats.get(
                    row['Region'], {}).get('gdp_pc_mean', 1) 
                if row['Region'] in region_stats else 1,
                axis=1
            )
            
            # 区域内人口规模相对位置
            df['region_population_rank'] = df.groupby('Region')['Population'].rank(pct=True)
            
            # 区域内GDP规模相对位置
            df['region_gdp_rank'] = df.groupby('Region')['GDP PPP 2017'].rank(pct=True)
        
        # 4. 区域+收入组组合特征
        if 'Region' in df.columns and 'Income Group' in df.columns:
            # 使用预先计算的统计量（如果有）
            region_income_stats = self.global_stats.get('region_income_stats', {})
            
            # 相对于区域+收入组的发展水平
            df['region_income_relative_gdp_pc'] = df.apply(
                lambda row: row['GDP PPP/capita 2017'] / region_income_stats.get(
                    row['Region'], {}).get(row['Income Group'], {}).get('gdp_pc_mean', 1)
                if row['Region'] in region_income_stats and 
                   row['Income Group'] in region_income_stats.get(row['Region'], {}) else 1,
                axis=1
            )
            
            # 区域+收入组内的相对位置
            df['region_income_gdp_pc_rank'] = df.groupby(['Region', 'Income Group'])['GDP PPP/capita 2017'].rank(pct=True)
            df['region_income_population_rank'] = df.groupby(['Region', 'Income Group'])['Population'].rank(pct=True)
        
        # 5. 人口规模分组特征
        population_thresholds = self.global_stats.get('population_thresholds', [df['Population'].quantile(0.33), df['Population'].quantile(0.67)])
        
        # 创建人口规模分类
        df['population_size_category'] = pd.cut(
            df['Population'], 
            bins=[0, population_thresholds[0], population_thresholds[1], float('inf')],
            labels=[0, 1, 2]
        ).astype(int)
        
        # 按人口规模分组的相对发展水平
        population_size_stats = self.global_stats.get('population_size_stats', {})
        for size in [0,1,2]:
            size_mean = population_size_stats.get(size, {}).get('mean', 1)
            mask = df['population_size_category'] == size
            df.loc[mask, f'{size}_population_relative_gdp_pc'] = df.loc[mask, 'GDP PPP/capita 2017'] / size_mean
        
        # 6. 环境库兹涅茨曲线特征
        ekc_params = self.global_stats.get('ekc_params', {
            'peak_point': df['GDP PPP/capita 2017'].quantile(0.75),
            'early_threshold': df['GDP PPP/capita 2017'].quantile(0.25),
            'late_threshold': df['GDP PPP/capita 2017'].quantile(0.75)
        })
        
        # 基于High income组峰值的EKC曲线
        peak_point = ekc_params.get('peak_point', df['GDP PPP/capita 2017'].quantile(0.75))
        early_threshold = ekc_params.get('early_threshold', df['GDP PPP/capita 2017'].quantile(0.25))
        late_threshold = ekc_params.get('late_threshold', df['GDP PPP/capita 2017'].quantile(0.75))
        
        # 创建EKC特征
        df['ekc_distance'] = -(df['GDP PPP/capita 2017'] - peak_point) ** 2  # 倒U型关系
        
        # 发展阶段特征
        df['early_development_stage'] = np.where(df['GDP PPP/capita 2017'] < early_threshold, 
                                               df['GDP PPP/capita 2017'], early_threshold)
        
        df['mid_development_stage'] = np.where(
            (df['GDP PPP/capita 2017'] >= early_threshold) & (df['GDP PPP/capita 2017'] < late_threshold),
            df['GDP PPP/capita 2017'] - early_threshold, 0
        )
        
        df['late_development_stage'] = np.where(df['GDP PPP/capita 2017'] >= late_threshold,
                                              df['GDP PPP/capita 2017'] - late_threshold, 0)
        
        # 7. 综合发展指标
        # 经济发展水平 - 基于人均GDP的归一化值
        gdp_pc = df['GDP PPP/capita 2017']
        df['economic_development_level'] = (gdp_pc - gdp_pc.min()) / (gdp_pc.max() - gdp_pc.min()) if gdp_pc.max() > gdp_pc.min() else 0.5
        
        # 工业化水平指标 - 使用人均GDP相对于High income组的比例
        high_income_mean = self.global_stats.get('income_group_stats', {}).get('High income', {}).get('gdp_pc_mean', gdp_pc.max())
        df['industrialization_level'] = df['GDP PPP/capita 2017'] / high_income_mean
        
        # 8. 交互特征
        # GDP与人口的交互
        df['gdp_population_interaction'] = df['gdp_ppp_2017_log'] * df['population_log']
        
        # 人均GDP与人口规模的交互
        df['gdp_pc_population_interaction'] = df['gdp_ppp_per_capita_2017_log'] * df['population_log']
        
        # 发展水平与人口规模的交互
        df['development_population_interaction'] = df['economic_development_level'] * df['population_log']
        
        # 9. 时间相关特征
        # 年代特征
        df['decade'] = (df['Year'] // 10) * 10
        
        # 时间趋势
        min_year = df['Year'].min()
        df['year_since_min'] = df['Year'] - min_year
        
        # 时间与发展水平的交互
        df['time_development_interaction'] = df['year_since_min'] * df['economic_development_level']
        
        # 10. 工业废物特有特征
        # 工业化强度 - 工业废物与GDP的比例关系
        df['industrial_intensity'] = df['industrialization_level'] ** 2 * df['gdp_ppp_2017_log']
        
        # 工业废物与人口的关系 - 人均工业化水平
        df['per_capita_industrialization'] = df['industrialization_level'] * df['population_log']
    
        # 城市化与经济发展的交互
        df['urban_gdp_interaction'] = df['urban_population_pct_log'] * df['gdp_ppp_per_capita_2017_log']
        
        # 城市化与人口规模的交互
        df['urban_population_interaction'] = df['urban_population_pct_log'] * df['population_log']
        
        # 城市化与工业化水平的交互
        df['urban_industrialization_interaction'] = df['urban_population_pct_log'] * df['industrialization_level']
        
        # CW特有特征
        # 城市化变化率 - 捕捉城市扩张速度
        df['urban_growth_rate'] = df.groupby('Country Name')['Urban population %'].pct_change().fillna(0)
        
        # 建设强度指标 - GDP增长与城市化的复合效应
        df['construction_intensity'] = df['gdp_ppp_2017_growth'] * df['urban_population_pct_log']
        
        # 发展阶段建设需求 - 不同发展阶段的建设需求不同
        df['development_construction_need'] = df['economic_development_level'] * (1 - df['economic_development_level']) * df['urban_population_pct_log']        
        
        
        
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
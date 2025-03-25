import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Dict, Optional, Tuple, Set
from ..config.config import Config

class FeatureEngineering:
    def __init__(self):
        """初始化特征工程类"""
        self.required_columns = Config.FEATURE_CONFIG['usecols']
        self.base_year = Config.FEATURE_CONFIG['base_year']  # 添加缺失的base_year
        
    def save_params(self, params_path: str) -> None:
        """保存特征工程参数到文件"""
        params = {}
        pd.to_pickle(params, params_path)
        
    def load_params(self, params_path: str) -> None:
        """从文件加载特征工程参数"""
        params = pd.read_pickle(params_path)

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
        """拟合训练数据
        """
        pass
            
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """使用已拟合的参数转换数据，生成特征
        
        Args:
            df: 输入数据

        Returns:
            处理后的DataFrame副本，包含所有生成的特征
        """
        self.validate_columns(df)
        df = df.copy()
        
        # 1. 基础指标的非线性特征和时间序列特征
        for metric in ['Population', 'GDP PPP 2017','GDP PPP/capita 2017', 'Urban population %']:
            metric_name = metric.lower().replace(' ', '_').replace('/', '_per_').replace('%', 'pct')
            
            # 对数变换
            if metric != 'Urban population %':
                df[f'{metric_name}_log'] = np.log1p(df[metric])
            else:
                df[f'{metric_name}_log'] = np.log1p(df[metric] / 100) * 100
            
            # 二次项特征（仅对GDP和人口使用）
            if metric != 'Urban population %':
                df[f'{metric_name}_squared'] = np.square(df[f'{metric_name}_log'])
            
            # 时间序列特征
            # 增长率（基于历史数据）
            df[f'{metric_name}_growth'] = df.groupby('Country Name')[metric].pct_change().fillna(0)
            
            # 滑动窗口统计
            windows = [3, 5]
            for w in windows:
                # 滑动平均
                df[f'{metric_name}_ma_{w}'] = df.groupby('Country Name')[metric].transform(
                    lambda x: x.rolling(w, min_periods=1).mean()
                ).fillna(0)
                # 相对变化
                df[f'{metric_name}_rel_change_{w}'] = df[f'{metric_name}_ma_{w}'] / df[metric] - 1
                # 趋势强度
                df[f'{metric_name}_trend_strength_{w}'] = df[f'{metric_name}_ma_{w}'].pct_change().fillna(0)
        
        # 2. 人口与经济的交互特征
        df['gdp_population_interaction'] = df['gdp_ppp_per_capita_2017_log'] * df['population_log']
        
        # 经济增长与人口增长的协同性
        df['growth_synergy'] = df['gdp_ppp_2017_growth'] * df['population_growth']
        
        # 经济规模与人口密度
        df['economic_density'] = df['gdp_ppp_2017_log'] / df.groupby('Country Name')['population_log'].transform('mean')
        
        # 经济发展阶段（基于人均GDP）与人口结构
        df['development_population_impact'] = (
            df['gdp_ppp_per_capita_2017_log'] * 
            df.groupby('Country Name')['population_growth'].transform(
                lambda x: x.rolling(5, min_periods=1).mean()
            )
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
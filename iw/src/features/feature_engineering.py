import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Dict, Optional, Tuple, Set
# 假设你的配置文件路径或导入方式
# from ..config.config import Config 

class FeatureEngineering:
    def __init__(self):
        """初始化特征工程类，定义所需列和处理方法"""
        # 定义模型训练和预测时必需的基础列
        self.required_columns = [
            'Year', 'Country Name', 'Region', 'Income Group',
            'GDP PPP 2017', 'GDP PPP/capita 2017', 'Population',
            'Urban population %'  # 加入城市化率作为必需列
        ]
        # self.base_year = Config.FEATURE_CONFIG['base_year'] # 如果需要基准年份可以取消注释

        # 定义需要进行对数(log1p)变换的数值特征
        self.columns_to_log = ['GDP PPP 2017', 'GDP PPP/capita 2017', 'Population']

        # 用于存储拟合(fit)阶段计算的统计量，例如全局均值、中位数等
        self.global_stats = {}

    def save_params(self, params_path: str) -> None:
        """保存特征工程参数到文件"""
        params = {
            'global_stats': self.global_stats,
            'required_columns': self.required_columns,
            'columns_to_log': self.columns_to_log
            # 'base_year': self.base_year # 如果使用基准年份
        }
        pd.to_pickle(params, params_path)
        print(f"特征工程参数已保存至: {params_path}")

    def load_params(self, params_path: str) -> None:
        """从文件加载特征工程参数"""
        params = pd.read_pickle(params_path)
        self.global_stats = params.get('global_stats', {})
        self.required_columns = params.get('required_columns', self.required_columns)
        self.columns_to_log = params.get('columns_to_log', self.columns_to_log)
        # self.base_year = params.get('base_year', None) # 如果使用基准年份
        print(f"特征工程参数已从 {params_path} 加载")


    def validate_columns(self, df: pd.DataFrame) -> None:
        """验证输入数据是否包含所有必需列"""
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"数据缺失必需列: {missing_cols}")

    def transform_target(self, df: pd.DataFrame, target_column: str, method: str = 'log') -> pd.DataFrame:
        """
        转换目标变量 (例如，对数变换、Box-Cox变换)
        Args:
            df: 输入数据
            target_column: 目标变量列名
            method: 转换方法 ('log', 'boxcox', 'none')
        Returns:
            包含转换后目标变量的DataFrame
        """
        df = df.copy()
        transformed_column = f'{target_column}_{method}'

        if method == 'log':
            # 使用 log1p 转换，即 log(1 + x)，处理 0 值
            df[target_column] = df[target_column].clip(lower=0) # 确保非负
            df[transformed_column] = np.log1p(df[target_column])
            print(f"目标变量 '{target_column}' 已进行 log1p 转换，生成列 '{transformed_column}'")
        elif method == 'boxcox':
            # Box-Cox 要求数据为正数
            target_positive = df[target_column]
            min_val = target_positive.min()
            if min_val <= 0:
                print(f"警告: 目标列 '{target_column}' 包含非正数值。将加上 {1 - min_val} 进行 Box-Cox 变换。")
                target_positive = target_positive + (1 - min_val)

            if target_positive.min() > 0:
                 # 加一个小常数以防万一出现极小值
                df[transformed_column], _ = stats.boxcox(target_positive + 1e-6)
                print(f"目标变量 '{target_column}' 已进行 Box-Cox 转换，生成列 '{transformed_column}'")
            else:
                print(f"错误: Box-Cox 变换无法应用于非正数据，即使调整后仍然存在问题。跳过变换。")
                df[transformed_column] = df[target_column] # 或者抛出错误

        elif method == 'none':
             df[transformed_column] = df[target_column]
             print(f"目标变量 '{target_column}' 未进行转换。")
        else:
            raise ValueError(f"不支持的目标变量转换方法: {method}")

        return df

    def fit(self, df: pd.DataFrame) -> None:
        """
        拟合训练数据 - 简化版.
        主要验证列的存在性，可选计算全局统计量用于后续参考或标准化.
        """
        self.validate_columns(df) # 确保所有必需列都在训练数据中

        # 1. (可选) 计算全局统计量，可用于数据理解或未来的标准化步骤
        global_stats = {}
        # 包含城市化率，不包含工业增加值占比
        numeric_cols_for_stats = self.columns_to_log + ['Urban population %']
        print("正在计算全局统计量...")
        for col in numeric_cols_for_stats:
             if col in df.columns:
                 global_stats[f'{col}_mean'] = df[col].mean()
                 global_stats[f'{col}_median'] = df[col].median()
                 global_stats[f'{col}_std'] = df[col].std()
                 global_stats[f'{col}_min'] = df[col].min()
                 global_stats[f'{col}_max'] = df[col].max()
                 print(f"...计算完成: {col}")
             else:
                 print(f"警告: 列 '{col}' 未在DataFrame中找到，无法计算其全局统计量。")

        self.global_stats = global_stats
        print("Fit 过程完成。全局统计量已计算（如果找到列）。")


    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        转换数据，生成特征用于模型训练或预测.
        应用对数变换、创建交互项、处理分类特征和缺失值等.
        """
        print("开始 Transform 过程...")
        self.validate_columns(df) # 确保输入数据包含必需列
        df = df.copy()

        # 1. 核心数值指标的对数变换
        print("应用对数变换...")
        for col in self.columns_to_log:
            if col in df.columns:
                col_log = f'{col}_log'
                df[col_log] = np.log1p(df[col].clip(lower=1e-6)) # clip 避免 log(非正数)
                print(f"...已生成: {col_log}")

        # 2. 捕捉非线性关系 (简化EKC形式)
        print("创建非线性特征...")
        gdp_pc_log_col = 'GDP PPP/capita 2017_log'
        if gdp_pc_log_col in df.columns:
            df['gdp_pc_log_squared'] = df[gdp_pc_log_col] ** 2
            print(f"...已生成: gdp_pc_log_squared")

        # 3. 时间特征
        print("创建时间特征...")
        if 'Year' in df.columns:
            # 使用一个固定的基准年或者数据中的最小年份
            base_year = df['Year'].min() # 或者 self.base_year 如果设置了
            df['year_relative'] = df['Year'] - base_year
            print(f"...已生成: year_relative (基准年: {base_year})")

        # 4. 增长率 (使用对数差分)
        print("计算增长率...")
        # 确保按国家和年份排序以正确计算差分
        df = df.sort_values(['Country Name', 'Year'])
        for col in ['GDP PPP 2017', 'Population']: # 可根据需要添加其他列
             col_log = f'{col}_log'
             if col_log in df.columns:
                 growth_col = f'{col}_log_growth_1y'
                 # 计算同组内上一行到当前行的差值
                 df[growth_col] = df.groupby('Country Name')[col_log].diff(1)
                 # 填充每个国家第一年的NaN值（通常用0）
                 df[growth_col] = df[growth_col].fillna(0)
                 print(f"...已生成: {growth_col}")

        # 5. 处理城市化率指标
        print("处理城市化率...")
        urban_col = 'Urban population %'
        urban_perc_col = 'urban_pop_perc'
        if urban_col in df.columns:
             # 转换为 0-1 范围内的小数
             df[urban_perc_col] = df[urban_col].clip(0, 100) / 100.0
             print(f"...已生成: {urban_perc_col}")

        # 6. 基本交互特征
        print("创建交互特征...")
        gdp_log_col = 'GDP PPP 2017_log'
        pop_log_col = 'Population_log'
        if gdp_log_col in df.columns and pop_log_col in df.columns:
            df['gdp_log_x_pop_log'] = df[gdp_log_col] * df[pop_log_col]
            print(f"...已生成: gdp_log_x_pop_log")

        gdp_pc_log_col = 'GDP PPP/capita 2017_log'
        gdp_pc_log2_col = 'gdp_pc_log_squared'
        if urban_perc_col in df.columns:
            if gdp_pc_log_col in df.columns:
                 df['gdp_pc_log_x_urban'] = df[gdp_pc_log_col] * df[urban_perc_col]
                 print(f"...已生成: gdp_pc_log_x_urban")
            if gdp_pc_log2_col in df.columns:
                 df['gdp_pc_log2_x_urban'] = df[gdp_pc_log2_col] * df[urban_perc_col]
                 print(f"...已生成: gdp_pc_log2_x_urban")

        # --- 新增的时间相关交互特征 ---
        print("创建时间相关的交互特征...")
        year_rel_col = 'year_relative' # 确保这个变量名在你代码中一致

        if year_rel_col in df.columns:
            # 时间与人均GDP(log)的交互
            if gdp_pc_log_col in df.columns:
                interaction_col = f'{year_rel_col}_x_gdp_pc_log'
                df[interaction_col] = df[year_rel_col] * df[gdp_pc_log_col]
                print(f"...已生成: {interaction_col}")

            # 时间与人口(log)的交互
            if pop_log_col in df.columns:
                interaction_col = f'{year_rel_col}_x_{pop_log_col}'
                df[interaction_col] = df[year_rel_col] * df[pop_log_col]
                print(f"...已生成: {interaction_col}")

            # 时间与城市化率(百分比小数)的交互
            if urban_perc_col in df.columns:
                interaction_col = f'{year_rel_col}_x_{urban_perc_col}'
                df[interaction_col] = df[year_rel_col] * df[urban_perc_col]
                print(f"...已生成: {interaction_col}")
        else:
            print(f"...警告: 时间特征 '{year_rel_col}' 未找到，无法创建时间相关的交互特征。")
        # -------------------------------
        # 8. 清理 & 后处理 (缺失值填充和极端值处理)
        print("处理缺失值和无穷值...")
        # 选择所有数值类型的列进行处理
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        # 排除不应填充的标识符列（如果它们是数值类型）
        cols_to_exclude_from_fill = ['Year'] # 可能还有其他ID列
        numeric_cols_to_fill = [col for col in numeric_cols if col not in cols_to_exclude_from_fill]

        # 强制转为数值，并将无法转换的值变为NaN
        for col in numeric_cols_to_fill:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 将 Inf 和 -Inf 替换为 NaN，以便统一处理
        df[numeric_cols_to_fill] = df[numeric_cols_to_fill].replace([np.inf, -np.inf], np.nan)

        # 使用中位数填充 NaN 值
        na_counts_before = df[numeric_cols_to_fill].isnull().sum()
        print("...正在使用中位数填充NaN值...")
        for col in numeric_cols_to_fill:
            if df[col].isnull().any():
                median_val = df[col].median()
                # 如果整列都是 NaN，中位数也是 NaN，此时用 0 填充
                if pd.isna(median_val):
                     median_val = 0
                     print(f"警告: 列 '{col}' 中位数无法计算 (可能全为NaN)，将使用 0 填充。")
                df[col] = df[col].fillna(median_val)
        na_counts_after = df[numeric_cols_to_fill].isnull().sum()
        na_filled_count = (na_counts_before - na_counts_after).sum()
        print(f"...填充完成，共处理了 {na_filled_count} 个 NaN 值。")

        # (可选) 裁剪极端值，防止数值溢出或模型对极端值过于敏感
        print("裁剪极端数值...")
        finfo = np.finfo(np.float64)
        for col in numeric_cols_to_fill:
             df[col] = np.clip(df[col], finfo.min, finfo.max)

        # # 删除原始的、未处理的列（如果需要）
        # if urban_col in df.columns and urban_perc_col in df.columns:
        #      df = df.drop(columns=[urban_col])
        #      print(f"...已删除原始列: {urban_col}")
        # # 可以考虑删除其他原始数值列，如果它们已被 log 转换等完全替代

        print("Transform 过程完成。")
        return df
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from src.config.config import Config
from src.features.feature_engineering import FeatureEngineering

class DataPreprocessor:
    def __init__(self):
        self.config = Config.FEATURE_CONFIG
        self.path_config = Config.PATH_CONFIG
        self.fe = FeatureEngineering()
        self.params_path = Path(self.path_config['features_dir']) / 'feature_params.pkl'
        
    def process_historical_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理历史数据
        
        Args:
            df: 输入的历史数据DataFrame
            
        Returns:
            处理后的DataFrame
        """
        # 拟合特征工程参数并转换数据
        self.fe.fit(df)
        df = self.fe.transform(df)
        
        # 保存特征工程参数和处理后的特征
        self.fe.save_params(self.params_path)
        return df
        
    def process_future_data(self, historical_df: pd.DataFrame, future_df: pd.DataFrame) -> Dict[str, Path]:
        """处理未来预测数据，合并历史数据以计算时序特征，然后按Scenario分别保存未来特征文件
        
        Args:
            historical_df: 输入的历史数据DataFrame (用于提供计算时序特征的上下文)
            future_df: 输入的未来预测数据DataFrame，必须包含 'Scenario' 列
            
        Returns:
            一个字典，键是场景名称(str)，值是该场景处理后特征文件的路径(Path)
            
        Raises:
            ValueError: 如果 future_df 缺少 'Scenario' 列，或列不匹配等
            FileNotFoundError: 如果特征工程参数文件未找到
        """
        # 验证 'Scenario' 列是否存在于 future_df
        if 'Scenario' not in future_df.columns:
            raise ValueError("未来数据DataFrame中缺少 'Scenario' 列。")

        # 验证输入数据的列是否大致匹配 (允许 future_df 多一个 Scenario 列)
        historical_cols = set(historical_df.columns)
        future_cols_base = set(future_df.columns) - {'Scenario'} # 比较基础列
        if historical_cols != future_cols_base:
             # 找出差异列以便调试
             missing_in_future = historical_cols - future_cols_base
             extra_in_future = future_cols_base - historical_cols
             raise ValueError(
                 f"历史数据和未来数据的基础列不完全一致。\n"
                 f"未来数据缺少列: {missing_in_future if missing_in_future else '无'}\n"
                 f"未来数据多出列(除Scenario): {extra_in_future if extra_in_future else '无'}"
             )

        # 验证特征工程参数文件是否存在
        if not self.params_path.exists():
             raise FileNotFoundError(f"特征工程参数文件未找到: {self.params_path}")
             
        # 加载特征工程参数
        print(f"加载特征工程参数从: {self.params_path}")
        self.fe.load_params(self.params_path)

        # 合并历史和未来数据 (保留 Scenario 列)
        # 为了合并，临时给 historical_df 添加一个 Scenario 列 (可以用 NaN 或特定值)
        historical_df_copy = historical_df.copy()
        historical_df_copy['Scenario'] = np.nan # 或者 'Historical'

        # 确保 future_df 的列顺序和类型与 historical_df_copy 大致匹配以便 concat
        # （concat 对列顺序不敏感，但类型最好一致）
        # future_df = future_df[historical_df_copy.columns] # 如果需要强制列顺序一致

        print("合并历史数据和未来数据...")
        # 使用 outer join 确保所有列都被包含，即使 future_df 多了 Scenario
        df_merged = pd.concat([historical_df_copy, future_df], ignore_index=True, sort=False) 
        print(f"合并后数据形状: {df_merged.shape}")
        
        # 按国家和年份排序，确保 pct_change 等计算正确
        print("按国家和年份排序...")
        df_merged = df_merged.sort_values(['Country Name', 'Year'])
        
        # 应用特征工程转换
        print("应用特征工程转换...")
        # !! 注意: 确保 FeatureEngineering.transform 不会因为合并后的数据而错误地重新计算全局统计量 !!
        # !! 它应该主要使用 self.global_stats 中加载的参数 !!
        try:
            df_transformed = self.fe.transform(df_merged)
            print(f"特征转换后数据形状: {df_transformed.shape}")
        except Exception as e:
            print(f"特征工程转换时出错: {e}")
            # 可以选择记录错误涉及的列或行
            raise e # 重新抛出异常

        # 筛选出属于未来的、有有效Scenario的数据行
        print("筛选未来数据...")
        future_years = future_df['Year'].unique()
        # 同时确保 Scenario 不是我们为历史数据添加的 NaN 值
        future_data_processed = df_transformed[
            df_transformed['Year'].isin(future_years) & 
            df_transformed['Scenario'].notna() &
            df_transformed['Scenario'].isin(future_df['Scenario'].unique()) # 确保是原始的未来场景
        ].copy() # 使用 .copy() 避免 SettingWithCopyWarning
        print(f"筛选出的未来数据形状: {future_data_processed.shape}")
        
        if future_data_processed.empty:
             print("警告：特征转换后未能筛选出任何有效的未来数据行。请检查合并和转换过程。")
             return {}

        # 确保特征目录存在
        features_dir = Path(self.path_config['features_dir'])
        features_dir.mkdir(parents=True, exist_ok=True)

        processed_scenario_paths = {}
        print("按场景拆分并保存未来特征文件...")
        
        # 按 Scenario 分组处理并保存
        for scenario in future_data_processed['Scenario'].unique():
            print(f"  处理场景: {scenario}")
            # 从已经处理好的 future_data_processed 中筛选
            scenario_df_processed = future_data_processed[future_data_processed['Scenario'] == scenario]
            
            if scenario_df_processed.empty:
                print(f"    警告：场景 '{scenario}' 没有数据，跳过保存。")
                continue

            # 定义保存路径
            output_filename = f'future_features_{scenario}.csv'
            output_path = features_dir / output_filename
            
            # 保存处理后的数据
            scenario_df_processed.to_csv(output_path, index=False, encoding='utf-8-sig')
            processed_scenario_paths[scenario] = output_path
            print(f"    已保存场景 '{scenario}' 的处理后特征 ({scenario_df_processed.shape[0]} 行) 至: {output_path}")

        print("\n未来数据特征生成完成。")
        return processed_scenario_paths

    def merge_features(self, msw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """合并特征并分割数据集"""
        # 进行目标变量转换
        target_column = Config.DATA_CONFIG['target_column']
        msw_df = self.fe.transform_target(msw_df, target_column)
        
        # 加载全局特征
        features_path = Path(Config.PATH_CONFIG['features_dir']) / 'global_features.csv'
        feature_df = pd.read_csv(features_path)
        print(len(feature_df))

        target_column = Config.DATA_CONFIG['target_column']
        method = Config.FEATURE_CONFIG['target_transform_method']
        transformed_column = f'{target_column}_{method}'
        # 只保留必要的列，避免重复
        msw_columns = ['Year', 'Country Name', target_column]
        if transformed_column in msw_df.columns:
            msw_columns.append(transformed_column)
            
        msw_df = msw_df[msw_columns]

        # 合并特征，以feature_df为主表
        merged_df = feature_df.merge(
            msw_df,
            on=['Year', 'Country Name'],
            how='left'
        )
        
        # 分割有/无MSW的数据
        train_df = merged_df[merged_df[target_column].notnull()]
        predict_df = merged_df[merged_df[target_column].isnull()]
        
        return train_df, predict_df
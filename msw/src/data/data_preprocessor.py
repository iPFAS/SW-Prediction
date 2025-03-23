import pandas as pd
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
        
    def process_future_data(self, historical_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
        """处理未来预测数据
        
        Args:
            historical_df: 输入的历史数据DataFrame
            future_df: 输入的未来预测数据DataFrame
            
        Returns:
            处理后的DataFrame，仅包含未来预测数据部分
            
        Raises:
            ValueError: 当historical_df和future_df的列不一致时抛出异常
        """
        # 验证输入数据的列是否一致
        historical_cols = set(historical_df.columns)
        future_cols = set(future_df.columns)
        if historical_cols != future_cols:
            raise ValueError(f"历史数据和未来数据的列不一致。\n历史数据列: {historical_cols}\n未来数据列: {future_cols}")
            
        # 合并历史和未来数据
        df = pd.concat([historical_df, future_df], ignore_index=True)
        df = df.sort_values(['Country Name', 'Year'])
        
        # 加载特征工程参数并转换数据
        self.fe.load_params(self.params_path)
        df = self.fe.transform(df)
        
        # 只返回未来数据部分
        future_data = df[df['Year'].isin(future_df['Year'].unique())]
        
        return future_data

    def merge_features(self, msw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """合并特征并分割数据集"""
        # 进行目标变量转换
        target_column = Config.DATA_CONFIG['target_column']
        msw_df = self.fe.transform_target(msw_df, target_column)
        
        # 加载全局特征
        features_path = Path(Config.PATH_CONFIG['features_dir']) / 'global_features.csv'
        feature_df = pd.read_csv(features_path)
        
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
from typing import Dict, List, Optional
import os
from pathlib import Path
import pandas as pd
from typing import Tuple

class Config:
    # 项目根目录
    PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # 数据加载配置
    DATA_CONFIG = {
        'data_path': r"E:\博士\1-课题\0-固废产生的研究\2-数据整理结果\1-MSW_CW_IW_HIW_training_data.xlsx", 
        'target_column': 'CW',
        'common_columns': ['Region', 'Income Group'],
        'country_train_size': 0.8,
        'country_val_size': 0.1,
        'time_test_size': 0.2,
        'random_state': 123
    }

    # 特征工程配置
    FEATURE_CONFIG = {
        'categorical_columns': [],
        'target_transform_method': 'log',
        'base_year': 1990,
        'historical_data_path': r'E:\博士\1-课题\0-固废产生的研究\2-数据整理结果\0-indicator_list-v2.xlsx',
        'historical_sheet': 'CW全部指标',
        'historical_msw_data_path': r'E:\博士\1-课题\0-固废产生的研究\2-数据整理结果\1-MSW_CW_IW_HIW_training_data.xlsx',
        'historical_msw_sheet': 'cw_result',
        'future_data_path': r'E:\博士\1-课题\0-固废产生的研究\2-数据整理结果\0-indicator_list-v2.xlsx',
        'future_sheet': 'CW全部指标',
        'usecols': ['Year', 'Country Name', 'Population', 
                    'GDP PPP 2017', 'GDP PPP/capita 2017','Income Group', 'Region','Urban population %']
    }

    # 模型训练配置
    MODEL_CONFIG = {
        'train_size': 0.8,
        'n_top_models': 10,
        'fold_strategy': 'timeseries',
        'fold_shuffle': False,
        'fold': 5,
        'normalize_method': 'minmax',
        'normalize': True,
        'data_split_shuffle': False,
        'verbose' : True,
        'session_id': 456,
    }

    # 模型调优配置
    TUNING_CONFIG = {
        'n_iter': 100,
        'optimize': 'RMSE',
        'search_library': 'optuna',
        'early_stopping': 20,
        'return_train_score': False
    }

    # 文件路径配置
    PATH_CONFIG = {
        'models_dir': str(PROJECT_ROOT / 'models/modelfile'),
        'data_dir': str(PROJECT_ROOT / 'data/datafile'),
        'features_dir': str(PROJECT_ROOT / 'features/featurefile'),
        'log_dir': str(PROJECT_ROOT / 'logs'),
        'visualization_dir': str(PROJECT_ROOT / 'results' / 'visualization'),
        'prediction_dir': str(PROJECT_ROOT /'results' / 'prediction')
    }

    # 日志配置
    LOG_CONFIG = {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    }
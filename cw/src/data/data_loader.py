import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from src.config.config import Config

class DataLoader:
    def __init__(self, file_path: Optional[str] = None):
        """初始化数据加载器

        Args:
            file_path: Excel文件路径，可选参数
        """
        self.file_path = file_path
        self.common_columns = Config.DATA_CONFIG['common_columns']

    def load_data(self, sheet_name: str, target_column: str,
                  feature_columns: Optional[List[str]] = None,
                  file_path: Optional[str] = None) -> pd.DataFrame:
        """加载并处理数据

        Args:
            sheet_name: Excel表格名称
            target_column: 目标列名（如'MSW', 'CW'等）
            feature_columns: 特征列名列表，如果为None则使用所有可用列
            file_path: 可选的文件路径，如果提供则覆盖实例的file_path

        Returns:
            处理后的DataFrame
        """

        # 确定使用哪个文件路径
        path_to_use = file_path if file_path is not None else self.file_path
        
        # 检查文件路径是否存在
        if path_to_use is None:
            raise ValueError("未设置文件路径，请通过参数提供或使用set_file_path方法设置")
        # 读取数据
        df = pd.read_excel(path_to_use, sheet_name=sheet_name)

        # 确保必要的列存在
        for col in self.common_columns:
            if col not in df.columns:
                raise ValueError(f"缺少必要的列: {col}")

        # 如果未指定特征列，使用除目标列和通用列之外的所有列
        if feature_columns is None:
            feature_columns = [col for col in df.columns
                             if col not in self.common_columns + [target_column]]

        # 合并所有需要的列
        used_columns = self.common_columns + feature_columns + [target_column]
        df = df[used_columns].copy()

        return df

    def split_data_by_countries(self, df: pd.DataFrame,
                              train_size: float = 0.8,
                              val_size: float = 0.1,
                              random_state: int = 888) -> tuple:
        """按国家划分训练集、验证集和测试集

        Args:
            df: 输入数据
            train_size: 训练集比例，默认0.8
            val_size: 验证集比例，默认0.1
            random_state: 随机种子

        Returns:
            训练集、验证集和测试集的元组
        """

        np.random.seed(random_state)

        # 按地区和收入组分组国家
        country_groups = df.groupby(['Country Name']).agg({
            'Region': 'first',
            'Income Group': 'first'
        }).reset_index()

        # 创建分层组合
        country_groups['Strata'] = country_groups['Region'] + '_' + country_groups['Income Group']
        strata_counts = country_groups['Strata'].value_counts()
        total_countries = len(country_groups)

        # 计算验证集和测试集需要的国家数量
        val_size_n = int(total_countries * val_size)
        test_size_n = int(total_countries * (1 - train_size - val_size))

        val_countries = []
        test_countries = []

        # 按比例从每个分层中抽样
        for strata, count in strata_counts.items():
            strata_countries = country_groups[country_groups['Strata'] == strata]['Country Name'].tolist()
            
            # 计算验证集样本数
            n_val = max(1, int(np.ceil(count / total_countries * val_size_n)))
            n_test = max(1, int(np.ceil(count / total_countries * test_size_n)))
            
            if len(strata_countries) > (n_val + n_test):
                val_sampled = np.random.choice(strata_countries, size=n_val, replace=False)
                remaining = [c for c in strata_countries if c not in val_sampled]
                test_sampled = np.random.choice(remaining, size=n_test, replace=False)
            else:
                val_sampled = strata_countries[:n_val]
                test_sampled = strata_countries[n_val:n_val+n_test]
            
            val_countries.extend(val_sampled)
            test_countries.extend(test_sampled)

        # 最终调整确保比例准确
        val_countries = list(np.random.choice(val_countries, size=val_size_n, replace=False))
        test_countries = list(np.random.choice(test_countries, size=test_size_n, replace=False))

        # 划分数据集
        val_data = df[df['Country Name'].isin(val_countries)]
        test_data = df[df['Country Name'].isin(test_countries)]
        train_data = df[~df['Country Name'].isin(val_countries + test_countries)]

        return train_data, val_data, test_data

    def split_data_by_time(self, df: pd.DataFrame, test_size: float = 0.15) -> tuple:
        # 参数校验
        if not 0 < test_size < 1:
            raise ValueError("test_size必须在0到1之间")
        if 'Country Name' not in df.columns or 'Year' not in df.columns:
            raise ValueError("数据框必须包含Country Name和Year列")

        try:
            # 初始化时间外样本测试集
            time_test_data = pd.DataFrame()

            # 对每个国家，提取最后15%的年份数据
            for country in df['Country Name'].unique():
                country_data = df[df['Country Name'] == country].copy()
                
                # 按年份排序
                country_data = country_data.sort_values('Year')
                
                # 计算要提取的数据量
                n_years = len(country_data)
                n_test = max(1, int(n_years * test_size))  # 至少提取1个样本
                
                # 提取最后15%的数据
                country_test = country_data.iloc[-n_test:].copy()
                time_test_data = pd.concat([time_test_data, country_test])

            # 生成训练集（排除测试数据）
            train_data = df[~df.index.isin(time_test_data.index)].copy()

        except Exception as e:
            raise RuntimeError(f"数据划分失败: {str(e)}") from e

        return train_data, time_test_data

    def analyze_datasets(self, df: pd.DataFrame) -> dict:
        """分析数据集的基本统计信息

        Args:
            df: 输入数据集

        Returns:
            包含数据总条数、国家列表和国家总数的字典
        """
        # 获取数据总条数
        total_records = len(df)

        # 获取国家列表并排序
        countries = sorted(df['Country Name'].unique())
        
        # 获取国家总数
        country_count = len(countries)
        
        # 获取特征列（排除Config中定义的列）
        feature_columns = sorted([col for col in df.columns if col not in Config.FEATURE_CONFIG['usecols']])
        feature_count = len(feature_columns)
        
        # 打印基本信息
        print(f"总数据条数: {total_records}")
        print(f"国家总数: {country_count}")
        print(f"包含的国家: {', '.join(countries)}")
        print(f"\n特征数量: {feature_count}")
        print(f"特征列表: {', '.join(feature_columns)}")

        time_test_years = df['Year'].value_counts().sort_index()
        year_str = ' | '.join([f"{year}:{count}" for year, count in time_test_years.items()])
        print(f"\n年份分布: {year_str}")

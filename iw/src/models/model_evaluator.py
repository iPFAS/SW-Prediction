import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from pycaret.regression import *
import os
import datetime
from ..config.config import Config
import warnings


class ModelEvaluator:
    def __init__(self):
        """初始化模型评估器"""
        self.models = {}
        self.model_scores = {}
        self.ensemble_scores = {}
        self.tuning_results = []
        self.timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")

    def setup_experiment(self,train_size: float,
                        train_data: pd.DataFrame,
                        val_data: pd.DataFrame,
                        target_column: str,
                        categorical_features: Optional[List[str]] = None,
                        numeric_features: Optional[List[str]] = None):
        """设置实验环境

        Args:
            train_data: 训练数据
            target_column: 目标列名
            categorical_features: 类别特征列表
            numeric_features: 数值特征列表
        """
        # 设置pycaret实验
        self.exp = setup(
            session_id = Config.MODEL_CONFIG['session_id'],
            train_size = train_size,
            data= train_data,
            test_data = val_data,
            target= target_column,
            categorical_features= categorical_features,
            fold_strategy= Config.MODEL_CONFIG['fold_strategy'],
            data_split_shuffle = Config.MODEL_CONFIG['data_split_shuffle'],
            fold_shuffle = Config.MODEL_CONFIG['fold_shuffle'],
            fold = Config.MODEL_CONFIG['fold'],
            normalize_method = Config.MODEL_CONFIG['normalize_method'],
            normalize = Config.MODEL_CONFIG['normalize'],
            log_experiment=True,
        )

    def train_top_models(self, n_models: int = 4) -> Dict:
        """训练并获取前N个最佳模型

        Args:
            n_models: 要选择的最佳模型数量

        Returns:
            训练好的模型字典
        """
        # 比较并获取最佳模型
        warnings.filterwarnings('ignore', category=UserWarning, message=r'.*LightGBM.*')
        warnings.filterwarnings('ignore', category=FutureWarning, message=r'.*LightGBM.*')
        best_models = compare_models(n_select=n_models,sort = Config.TUNING_CONFIG['optimize'])
        result = pull()
        
        # 添加模型对比结果
        self.tuning_results.append({
            'stage': 'top_models_comparison',
            'model': 'top_models',
            'metrics': result
        })
        
        # 创建模型字典
        model_dict = {}
        for i, model in enumerate(best_models):
            model_name = result.index[i]
            model_dict[model_name] = model
        return model_dict

    def tune_models(self, models: Dict) -> Tuple[Dict, str]:
        """模型调优方法
        Args:
            models: 需要调优的模型字典 {模型名: 模型对象}
        Returns:
            Tuple[Dict, str]: 返回一个元组，包含:
                - 调优后的模型字典 {模型名: 模型对象}
                - 性能最好的模型名称
        """
        tuned_models = {}
        
        for name, model in models.items():
            print(f'当前调优模型: {name} , 结果如下:')
             # 超参数调优
            tuned_model = tune_model(
                model,
                n_iter=Config.TUNING_CONFIG['n_iter'],
                optimize=Config.TUNING_CONFIG['optimize'],
                search_library=Config.TUNING_CONFIG['search_library'],
                early_stopping=Config.TUNING_CONFIG['early_stopping'],
                return_train_score=Config.TUNING_CONFIG['return_train_score'],
                choose_better=True  # 自动选择更好的参数
            )
            tune_result = pull()
            predict_model(tuned_model)
            # 在结果中添加模型名称列
            tune_result['Model'] = name
            
            # 如果是第一个模型，直接使用结果
            if not self.tuning_results or 'tuned_model' not in [r['stage'] for r in self.tuning_results]:
                combined_metrics = tune_result
                self.tuning_results.append({
                    'stage': 'tuned_model',
                    'model': 'combined',
                    'metrics': combined_metrics
                })
            else:
                # 找到tuned_model阶段的结果并更新
                for result in self.tuning_results:
                    if result['stage'] == 'tuned_model':
                        result['metrics'] = pd.concat([result['metrics'], tune_result])
                        break
            
            tuned_models[name] = tuned_model
        
        return tuned_models

    def saved_models(self, models: dict):
        """保存模型到指定路径并生成清单文件
        Args:
            models: 要保存的模型字典 {模型名: 模型对象}
        """
        # 创建模型目录
        model_dir = Config.PATH_CONFIG['models_dir']
        os.makedirs(model_dir, exist_ok=True)
        
        # 保存每个模型
        model_list = []
        
        for name, model in models.items():
            # 校验模型名称
            if not name.isidentifier():
                raise ValueError(f"非法模型名称: {name}")
            
            # 使用类的时间戳
            new_name = f"{name}_{self.timestamp}"
                
            # 构建保存路径
            save_path = os.path.join(model_dir, new_name)
            
            # 保存模型
            model = finalize_model(model)
            save_model(model, save_path)
            model_list.append({'model_name': new_name, 'path': save_path})
        
        # 生成模型清单
        with pd.ExcelWriter(
            os.path.join(model_dir, f'model_manifest_{self.timestamp}.xlsx'),
            engine='openpyxl'
        ) as writer:
            pd.DataFrame(model_list).to_excel(writer, index=False)

    def _save_training_results(self, results: List[Dict]):
        """保存训练结果到Excel文件
        Args:
            results: 训练结果列表
        """
        output_path = os.path.join(Config.PATH_CONFIG['models_dir'], f'training_results_{self.timestamp}.xlsx')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            # 处理已存在文件
            if os.path.exists(output_path):
                os.remove(output_path)
            
            # 确保至少有一个工作表被创建
            default_sheet_created = False
            
            with pd.ExcelWriter(
                output_path,
                engine='openpyxl'
            ) as writer:
                stages = {
                    'top_models_comparison': '模型初始对比',
                    'tuned_model': '调优模型结果',
                    'blend_model': '混合模型结果',
                    'final_comparison': '最终性能对比'
                }
                
                for result in results:
                    stage = result['stage']
                    sheet_name = stages.get(stage, '未知阶段')
                    result['metrics'].to_excel(writer, sheet_name=sheet_name)
                    default_sheet_created = True
                
                if not default_sheet_created:
                    pd.DataFrame().to_excel(writer, sheet_name='空白工作表')
        
        except Exception as e:
            print(f'保存训练结果时发生错误: {str(e)}')
            raise

    def ensemble_models(self, models: dict):
        # 集成模型
        estimator_list = list(models.values())
        blend = blend_models(estimator_list=estimator_list)

        blend_result = pull()
        # predict_model(blend)
        self.tuning_results.append({
            'stage': 'blend_model',
            'model': 'blend',
            'metrics': blend_result
        })

        models['blend'] = blend
        # 性能对比
        all_models = list(models.values())
        compare_models(include=all_models)
        final_result = pull()
        self.tuning_results.append({
            'stage': 'final_comparison',
            'model': 'all',
            'metrics': final_result
        })
        # 保存所有训练结果
        self._save_training_results(self.tuning_results)   

        return models
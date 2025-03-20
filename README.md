# 固体废物产生量预测系统

本项目是一个基于机器学习的固体废物产生量预测系统，支持多种固体废物类型（MSW、CW、IW、HIW等）的预测分析。系统采用多种机器学习模型和集成方法，通过对比不同模型组合的性能来选择最优预测策略。

## 功能特点

- 支持多种固体废物类型的预测
- 灵活的特征和标签配置
- 多模型集成评估框架
- 交叉验证性能评估
- 可视化分析工具

## 项目结构

```
├── data/               # 数据目录
├── models/             # 保存训练好的模型
├── notebooks/          # Jupyter notebooks
├── reports/            # 生成的图表和报告
├── src/                # 源代码
│   ├── data/           # 数据处理相关代码
│   ├── features/       # 特征工程相关代码
│   ├── models/         # 模型训练和评估代码
│   └── visualization/  # 数据可视化代码
├── tests/              # 测试代码
├── requirements.txt    # 项目依赖
└── README.md          # 项目说明文档
```

## 安装说明

1. 克隆项目到本地：
```bash
git clone [repository-url]
cd SW-Prediction
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 数据准备：
   - 将数据文件放入 `data` 目录
   - 支持Excel格式的输入数据
   - 数据必须包含 'Country Name' 和 'Year' 列

2. 配置预测参数：
   - 在配置文件中指定特征列和标签列
   - 选择需要预测的固废类型

3. 运行预测：
   - 使用提供的API或命令行工具进行预测
   - 支持单模型预测和模型集成预测

## 模型评估

系统提供以下评估指标：
- R²分数
- 均方根误差（RMSE）
- 平均绝对误差（MAE）
- 交叉验证分数

## 贡献指南

欢迎提交问题和改进建议！请遵循以下步骤：
1. Fork 本仓库
2. 创建您的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开一个 Pull Request

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件
# Solid Waste Generation Prediction

## Overview

This project aims to predict the generation of different types of solid waste (Municipal Solid Waste - MSW, Construction Waste - CW, Industrial Waste - IW) for various countries based on historical data and socioeconomic indicators. It utilizes a machine learning approach, primarily leveraging ensemble models trained using the PyCaret library, to capture complex relationships between waste generation and factors like GDP, population, urbanization, region, and income group.

The framework is designed to be adaptable for different waste streams, potentially with variations in feature engineering and data splitting strategies depending on data availability and characteristics.

## Workflow Summary

The project follows a standard machine learning pipeline:

1.  **Data Preprocessing**: Load raw data, handle missing values, detect/treat outliers, and filter data based on quality criteria.
2.  **Feature Engineering**: Generate a rich set of features from base indicators (GDP, population, etc.) including non-linear transformations (log, polynomial), growth rates, relative indicators (compared to regional/income group averages), interaction terms, and time-based features. Target variable transformation (e.g., log) is also applied.
3.  **Data Splitting**:
    *   For data-rich streams (MSW, CW): Split data into training, time-series test (recent years), and country-out-of-sample test sets.
    *   For data-scarce streams (IW): Split data into training and country-out-of-sample test sets.
4.  **Model Training & Selection**: Use PyCaret to set up the experiment, compare various regression models (Random Forest, Gradient Boosting, etc.) using cross-validation, and select top performers.
5.  **Model Ensembling**: Combine the best-performing models using blending or stacking techniques to create a robust ensemble model.
6.  **Model Tuning (Optional)**: Fine-tune hyperparameters of selected models or the ensemble.
7.  **Model Saving**: Save the trained PyCaret pipeline (including preprocessing and the final model) and feature engineering parameters.
8.  **Prediction & Evaluation**: Load the saved model and parameters, apply feature engineering to test data, generate predictions, inverse-transform predictions if necessary, and evaluate performance using metrics like R², MAE, RMSE, and MAPE on the test sets.
9.  **Visualization**: Generate plots comparing actual vs. predicted values, time-series trends, and performance across different countries.

## Directory Structure

```plaintext
e:\code\jupyter\固废产生\SW-Prediction\
├── config                  # Configuration files
│   └── config.py
├── data                    # Raw, intermediate, and processed data
│   ├── raw                 # Original data files
│   └── processed           # Processed data (train/test splits)
├── models                  # Saved model files and feature engineering parameters
├── notebooks               # Jupyter notebooks for exploration, testing, and analysis
├── reports                 # Generated reports and figures
│   └── figures             # Saved plots
├── src                     # Source code
│   ├── data                # Data loading and splitting scripts (e.g., data_loader.py)
│   ├── features            # Feature engineering scripts (e.g., feature_engineering.py)
│   ├── models              # Model training, evaluation scripts (e.g., model_evaluator.py)
│   ├── visualization       # Visualization scripts (e.g., visualizer.py)
│   └── __init__.py
├── tests                   # Unit tests (if any)
├── main.py                 # Main script to run the pipeline (example)
├── requirements.txt        # Project dependencies
└── README.md               # This file

# Example structure within config.py
class Config:
    # --- Path Configuration ---
    PATH_CONFIG = {
        'data_dir': 'e:\\code\\jupyter\\data\\processed', # Adjusted path
        'model_dir': 'e:\\code\\jupyter\\models', # Adjusted path
        'output_dir': 'e:\\code\\jupyter\\reports\\figures', # Adjusted path
        # ... other paths
    }

    # --- Data Configuration ---
    DATA_CONFIG = {
        'data_path': 'e:\\code\\jupyter\\data\\raw\\your_data.xlsx', # Adjusted path
        'sheet_name': 'Sheet1', # Sheet containing the data
        'target_column': 'MSW_Generation', # Name of the target variable column
        'feature_columns': ['GDP PPP/capita 2017', 'Population', ...], # List of base feature columns to use
        'test_size': 0.2, # Proportion of countries for the country-out-of-sample test set
        'random_state': 42, # Random seed for reproducibility
        # ... other data loading/splitting params
    }

    # --- Feature Engineering Configuration ---
    FEATURE_CONFIG = {
        'usecols': ['Year', 'Country Name', 'Region', 'Income Group', 'GDP PPP/capita 2017', ...], # All columns required by feature engineering
        'target_transform_method': 'log', # Method to transform target ('log', 'boxcox', 'none')
        'base_year': 1990, # Base year for time-related features
        'categorical_columns': ['Region', 'Income Group'], # Columns to be treated as categorical
        # ... other feature engineering params
    }

    # --- Model Configuration ---
    MODEL_CONFIG = {
        'train_size': 0.8, # Proportion of data used for training within PyCaret setup (after initial splits)
        'models_to_compare': ['rf', 'et', 'gbc', 'lightgbm'], # List of model IDs to compare in PyCaret
        'models_to_exclude': [], # List of model IDs to exclude
        'sort_metric': 'R2', # Metric to sort models by in compare_models
        'ensemble_method': 'blend', # Ensemble method ('blend' or 'stack')
        'n_select': 3, # Number of top models to use for ensembling
        # ... other PyCaret setup or modeling params
    }

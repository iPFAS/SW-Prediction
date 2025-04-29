# Solid Waste Generation Prediction

## Overview

This project aims to predict the generation of different types of solid waste (Municipal Solid Waste - MSW, Construction and Demolition Waste - CW, Industrial Waste - IW) for various countries based on historical data and socioeconomic indicators. It utilizes a machine learning approach, primarily leveraging ensemble models trained using the PyCaret library, to capture complex relationships between waste generation and factors like GDP, population, urbanization, region, and income group.

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
10.  **Feature Selection, Comparison & Voting**: Multiple feature selection methods (e.g., model-based importance, recursive feature elimination, Lasso, mutual information) are used. The results are compared and a voting mechanism is applied to determine the final feature set for model training, improving robustness and reducing overfitting.
11.  **Outlier Detection & Filtering (Optional)**: Outlier detection is performed on the target variable and/or key features using statistical methods (such as z-score, IQR, or time-series ratio checks). Detected outliers can be optionally removed or flagged for further inspection, improving data quality and model reliability.
12.  **Model Interpretation**: Model interpretation is conducted using tools such as SHAP or feature importance plots. This allows users to understand the contribution of each feature to the model’s predictions, providing transparency and supporting decision-making.
13.  **External Prediction Module**: The framework supports external scenario-based prediction. After training, the model can be applied to future scenario datasets (e.g., 2022-2050) that may include different socioeconomic assumptions. The pipeline ensures that only non-overlapping years are used and that all feature engineering steps are consistently applied.
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
```
## Feature Engineering Considerations and Potential Data Leakage

This project employs a comprehensive feature engineering strategy to capture complex relationships relevant to waste generation prediction. However, users should be aware of certain aspects related to how some features are calculated, particularly concerning potential data leakage in a traditional time-series evaluation context.

**Features Calculated Dynamically in `transform`:**

Several features are calculated dynamically within the `transform` step of the feature engineering pipeline. These include:

1.  **Ranking Features:** Features like `income_group_population_rank`, `region_gdp_rank`, `region_income_gdp_pc_rank`, etc., are calculated using `pandas.groupby().rank()`. When applied to a dataset spanning multiple time steps (e.g., a validation set), the rank for a given time step `t` is influenced by data from subsequent time steps (`t+1`, `t+2`, ...) within that dataset.
2.  **Normalization/Scaling Features (Potentially, if not using `fit` parameters):** Features like `economic_development_level` or `year_since_min` *could* potentially be calculated using `min`/`max` values derived from the entire input `DataFrame` during `transform`. (Note: The current implementation aims to mitigate this for normalization by calculating parameters in `fit` and reusing them in `transform`, which is the standard best practice).
3.  **Growth Rate Features:** Features like `consumption_growth` calculated using `pct_change()` inherently use information from the previous time step.

**Traditional Data Leakage Perspective:**

In standard machine learning practice, especially for time-series forecasting evaluation, using information from future time steps (relative to the point being predicted) to generate features for the current time step is considered data leakage. This can lead to overly optimistic performance metrics during validation because the model effectively gets a "peek" at future information it wouldn't have in a real-world deployment scenario when predicting step-by-step.

**Justification for this Project's Approach:**

This project operates under a specific assumption relevant to long-term forecasting: **future values of *input features* (like GDP, population, etc.) are considered knowable**, typically obtained from forecasts provided by authoritative institutions (e.g., World Bank, IMF).

Therefore, when the model is deployed to predict waste generation for a future year (e.g., 2030), it *will* have access to the predicted GDP and population figures for 2030 for all relevant countries. Calculating features like the global or regional rank for a country in 2030 based on these *predicted* 2030 inputs is a necessary and valid part of the prediction process in this specific context.

**Impact on Validation Metrics:**

While this approach aligns with the intended application scenario, it's important to acknowledge its impact during the **evaluation phase** using historical data:

*   When the `transform` method (calculating ranks dynamically) is applied to a historical validation set spanning multiple years (e.g., 2018-2020), the calculation *does* introduce information from later years into earlier year feature calculations.
*   Consequently, the validation metrics (e.g., R², MAE) reported might be **optimistically biased** compared to a strict evaluation protocol where features for year `t` are calculated *only* using information available up to year `t`.

**Conclusion:**

The dynamic calculation of certain features (especially rankings) within the `transform` step reflects the operational reality of using predicted future inputs for future predictions. While this introduces a form of data leakage during historical validation, potentially inflating performance metrics, this trade-off is considered acceptable given the specific nature and requirements of this forecasting task. Users interpreting the validation results should be mindful of this potential bias. The most critical leakage (using future data for normalization parameters like min/max) is addressed by calculating these parameters during the `fit` phase on the training data.

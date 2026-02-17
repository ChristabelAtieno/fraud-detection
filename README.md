## Fraud Detection

An end-to-end machine learning pipeline designed to identify fraudulent transactions by merging identity and transaction data, performing advanced feature engineering, and optimizing tree-based models. 

### Project Overview
This project addresses the critical challenge of financial fraud by analyzing large-scale transactional datasets. It implements a robust pipeline that handles extreme class imbalance and ensures experiment reproducibility using MLflow

### Technical Pipeline

1. Data Integration
Table Merging: Joins Transaction and Identity datasets using specific keys to create a unified view.
 
2. Preprocessing & Feature Engineering
Automated Pipeline: Imputes missing values, encodes categorical variables, and normalizes numeric fields.
Correlated Features: Identified and kept the features with the high correlation to fraud.
Aggregation: Creates "group-by" features (e.g., mean transaction amount per card) to capture behavioral anomalies.

3. Model Development

Trains and benchmarks three state-of-the-art gradient boosting frameworks:
- XGBoost: High performance with customizable objective functions.
- LightGBM: Optimized for speed and large datasets.
- CatBoost: Natively handles categorical features without manual encoding.
- Imbalance Handling: Uses scale_pos_weight to focus learning on the rare fraud cases.

4. Experiment Tracking (MLflow)

Every run is logged to MLflow for full transparency: 
- Parameters: Hyperparameters like learning rate, max depth, and weight scales.
- Metrics: Logs Precision, Recall, F1-Score, and Average Precision (PR AUC).
- Artifacts: Saves confusion matrices, feature importance plots, and the serialized model. 
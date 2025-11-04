from src.data_merging import load_data
from src.data_preprocessing import preprocess_data, split_data
from src.xgboost import xgboost_model
from src.catboost import catboost_model
from src.lighgbm import lgbm_model

import mlflow

if __name__== "__main__":

    mlflow.set_experiment("Fraud Detection model")

    # load and merge the train/test
    train_df, test_df = load_data(
            train_transaction_path='data/train_transaction.csv',
            train_identity_path='data/train_identity.csv',
            test_transaction_path='data/test_transaction.csv',
            test_identity_path='data/test_identity.csv'
    )

    # Preprocess
    train_df_processed, test_df_processed = preprocess_data(train_df, test_df)

    # split data 
    X_train, X_test, y_train, y_test = split_data(train_df_processed)

    #xgboost model
    best_xgb_model, y_pred_xgb, y_proba_xgb = xgboost_model(X_train, X_test, y_train, y_test)

    #lightgbm
    best_lgbm_model, y_pred_lgb, y_proba_lgb = lgbm_model(X_train, X_test, y_train, y_test)

    #catboost
    #best_cat_model, y_pred_cat, y_proba_cat = catboost_model(X_train, X_test, y_train, y_test)

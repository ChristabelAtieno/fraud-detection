import mlflow
import mlflow.lightgbm
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, average_precision_score, roc_auc_score,accuracy_score, precision_score, recall_score

def lgbm_model(X_train, X_test, y_train, y_test):

    lgb_tuned = LGBMClassifier(
        objective='binary',
        boosting_type='gbdt',
        n_jobs=-1,
        random_state=42
    )

    param_grid ={
        'num_leaves': [31,63,127],
        'max_depth': [6, 8, 10, -1],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [500, 800, 1000],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'reg_alpha': [0, 1, 3, 5],
        'reg_lambda': [0, 1, 3, 5],
        'scale_pos_weight': [15, 20, 25] #handle class imbalanceness

    }

    random_search = RandomizedSearchCV(
        estimator=lgb_tuned,
        param_distributions=param_grid,
        n_iter=15,
        scoring='average_precision',
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    #===start mlflow
    with mlflow.start_run(run_name="lightgbm_best_run"):

        #fit
        random_search.fit(X_train, y_train)


        print("Best parameters found:", random_search.best_params_)
        print("Best score found:", random_search.best_score_)

        # Evaluation
        best_lgbm = random_search.best_estimator_

        y_pred_lgbm = best_lgbm.predict(X_test)
        y_proba_lgbm = best_lgbm.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred_lgbm)
        rec = recall_score(y_test, y_pred_lgbm)
        prec = precision_score(y_test, y_pred_lgbm)
        avg_pr = average_precision_score(y_test, y_proba_lgbm)
        roc = roc_auc_score(y_test, y_pred_lgbm)

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_lgbm))
        print("AUC-PR:",avg_pr)
        print("ROC-AUC:", roc)

        #===mlflow parameters amd metrics==
        mlflow.log_params(random_search.best_params_)
        mlflow.log_metrics({
            "accuracy":acc,
            "recall":rec,
            "precision":prec,
            "average_precision_score":avg_pr,
            "roc-auc":roc
        })

        
        #---feature importance
        feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': best_lgbm.feature_importances_
        }).sort_values(by='Importance', ascending=False).head(10)

        print("\nTop 10 important features:")
        print(feature_importance)

         #===log and register the model
        mlflow.xgboost.log_model(
                                lightgbm_model= best_lgbm,
                                artifact_path="lightgbm-model",
                                registered_model_name="fraud-detection-lightgbm")
        
        print("Model logged and registered successfully in MLflow!")

        
        return best_lgbm, y_pred_lgbm, y_proba_lgbm
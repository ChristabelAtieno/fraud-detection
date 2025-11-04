import mlflow
import mlflow.xgboost
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, average_precision_score, accuracy_score, precision_score, recall_score

# xgboost model
def xgboost_model(X_train, X_test, y_train, y_test):

    xgb_tuned = XGBClassifier(objective='binary:logistic', 
                            eval_metric='aucpr',
                            tree_method='hist',
                            random_state=42)

    #parameters
    param_grid = {
        'n_estimators':[300,500,700],
        'max_depth':[3,5,7,9],
        'learning_rate':[0.01, 0.05, 0.1, 0.2],
        'subsample':[0.6,0.8,1.0],
        'colsample_bytree':[0.6,0.8,1.0],
        'min_child_weight':[1,3,5,7],
        'gamma':[0,0.1,0.3,0.5],
        'reg_alpha':[0,0.01,0.1,1],
        'reg_lambda':[0.1,1,5,10],
        'scale_pos_weight':[1,5,10,20]
        }
  
    #instantiate random search
    rs = RandomizedSearchCV(
        estimator=xgb_tuned,
        param_distributions=param_grid,
        n_iter=20,
        cv=3,
        scoring='average_precision',
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    #===start mlflow
    with mlflow.start_run(run_name='xgboost_best_run'):

        #fit the model
        rs.fit(X_train, y_train)

        print("Best parameters found: ", rs.best_params_)
        print("Best score found: ", rs.best_score_)

        #--evaluate the best model
        best_xgb_model = rs.best_estimator_

        y_pred = best_xgb_model.predict(X_test)
        y_pred_proba = best_xgb_model.predict_proba(X_test)[:, 1]

        #--metrics
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        avg_p = average_precision_score(y_test, y_pred_proba)

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("AUC-PR:", avg_p)

        #===mlflow parameters amd metrics==
        mlflow.log_params(rs.best_params_)
        mlflow.log_metrics({
            "accuracy":acc,
            "recall":rec,
            "precision":prec,
            "average_precision_score":avg_p
        })

        #---feature importance
        feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': best_xgb_model.feature_importances_
        }).sort_values(by='Importance', ascending=False).head(10)

        print("\nTop 10 important features:")
        print(feature_importance)
        
        #===log and register the model
        mlflow.xgboost.log_model(
                                best_xgb_model, 
                                  artifact_path="xgboost-model",
                                  registered_model_name="fraud-detection-xgboost")
        
        print("Model logged and registered successfully in MLflow!")

        return best_xgb_model, y_pred, y_pred_proba


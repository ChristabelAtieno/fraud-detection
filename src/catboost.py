from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, average_precision_score, roc_auc_score

def catboost_model(X_train,X_test, y_train, y_test):

    cat_model = CatBoostClassifier(iterations=1000,
                                   learning_rate=0.1,
                                    depth=8,
                                    eval_metric='AUC',
                                    random_state=42,
                                    verbose=200,
                                    class_weights=[1,20]

    )

    cat_model.fit(X_train, y_train)


    y_pred_cat = cat_model.predict(X_test)
    y_proba_cat = cat_model.predict_proba(X_test)[:, 1]

    # Evaluate
    print(classification_report(y_test, y_pred_cat))
    print("AUC-PR:", average_precision_score(y_test, y_proba_cat))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba_cat))

    return y_pred_cat, y_proba_cat
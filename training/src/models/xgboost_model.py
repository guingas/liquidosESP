"""XGBoost Classifier."""

import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import ML_PARAMS


def treinar_xgboost(X_train, y_train, X_val, y_val):
    """Treina XGBoost e retorna resultado."""
    params = ML_PARAMS["xgboost"].copy()
    print("\n  [XGB] Treinando XGBoost...")

    n_classes = len(np.unique(y_train))
    if n_classes == 2:
        params["objective"] = "binary:logistic"
        params["eval_metric"] = "logloss"
    else:
        params["objective"] = "multi:softmax"
        params["eval_metric"] = "mlogloss"
        params["num_class"] = n_classes

    modelo = XGBClassifier(**params, use_label_encoder=False, verbosity=0)
    modelo.fit(X_train, y_train,
               eval_set=[(X_val, y_val)],
               verbose=False)

    acc_train = accuracy_score(y_train, modelo.predict(X_train))
    acc_val = accuracy_score(y_val, modelo.predict(X_val))

    print(f"        Acc treino: {acc_train:.4f} | Acc val: {acc_val:.4f}")

    return {
        "nome": "XGBoost",
        "modelo": modelo,
        "acc_train": acc_train,
        "acc_val": acc_val,
        "tipo": "xgboost",
    }

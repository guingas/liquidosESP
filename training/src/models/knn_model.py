"""k-NN Classifier."""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import ML_PARAMS


def treinar_knn(X_train, y_train, X_val, y_val):
    """Treina k-NN e retorna resultado."""
    params = ML_PARAMS["knn"]
    print("\n  [KNN] Treinando k-NN...")

    modelo = KNeighborsClassifier(**params)
    modelo.fit(X_train, y_train)

    acc_train = accuracy_score(y_train, modelo.predict(X_train))
    acc_val = accuracy_score(y_val, modelo.predict(X_val))

    print(f"        Acc treino: {acc_train:.4f} | Acc val: {acc_val:.4f}")

    return {
        "nome": "k-NN",
        "modelo": modelo,
        "acc_train": acc_train,
        "acc_val": acc_val,
        "tipo": "sklearn",
    }

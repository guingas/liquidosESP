"""SVM Classifier com subsampling inteligente para datasets grandes."""

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import ML_PARAMS, RANDOM_SEED

SVM_MAX_TRAIN_SAMPLES = 15000  # SVM é O(n²-n³); subsample preserva qualidade


def treinar_svm(X_train, y_train, X_val, y_val):
    """Treina SVM com kernel RBF e retorna resultado."""
    params = ML_PARAMS["svm"]
    print("\n  [SVM] Treinando SVM (RBF)...")

    # Subsample estratificado se dataset for muito grande
    if len(X_train) > SVM_MAX_TRAIN_SAMPLES:
        rng = np.random.RandomState(RANDOM_SEED)
        classes = np.unique(y_train)
        indices = []
        per_class = SVM_MAX_TRAIN_SAMPLES // len(classes)
        for c in classes:
            c_idx = np.where(y_train == c)[0]
            chosen = rng.choice(c_idx, min(per_class, len(c_idx)), replace=False)
            indices.extend(chosen)
        indices = np.array(indices)
        rng.shuffle(indices)
        X_fit = X_train[indices]
        y_fit = y_train[indices]
        print(f"        Subsampling: {len(X_train)} -> {len(X_fit)} amostras (estratificado)")
    else:
        X_fit = X_train
        y_fit = y_train

    modelo = SVC(**params, probability=True)
    modelo.fit(X_fit, y_fit)

    acc_train = accuracy_score(y_train, modelo.predict(X_train))
    acc_val = accuracy_score(y_val, modelo.predict(X_val))

    print(f"        Acc treino: {acc_train:.4f} | Acc val: {acc_val:.4f}")

    return {
        "nome": "SVM (RBF)",
        "modelo": modelo,
        "acc_train": acc_train,
        "acc_val": acc_val,
        "tipo": "sklearn",
    }

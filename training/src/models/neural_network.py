"""Rede Neural MLP (Keras/TensorFlow) — otimizada para TinyML."""

import numpy as np
import os
import sys
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import tensorflow as tf
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import ML_PARAMS, RANDOM_SEED


def criar_modelo_mlp(n_features: int, n_classes: int, params: dict):
    """Cria modelo MLP leve compatível com TFLite."""
    tf.random.set_seed(RANDOM_SEED)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(n_features,)))

    for units in params["hidden_layers"]:
        model.add(tf.keras.layers.Dense(units, activation="relu"))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Dense(n_classes, activation="softmax"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def treinar_rede_neural(X_train, y_train, X_val, y_val, n_classes):
    """Treina MLP e retorna resultado."""
    params = ML_PARAMS["neural_network"]
    print("\n  [MLP] Treinando Rede Neural...")

    n_features = X_train.shape[1]
    modelo = criar_modelo_mlp(n_features, n_classes, params)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True
    )

    history = modelo.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        callbacks=[early_stop],
        verbose=0,
    )

    acc_train = accuracy_score(y_train, np.argmax(modelo.predict(X_train, verbose=0), axis=1))
    acc_val = accuracy_score(y_val, np.argmax(modelo.predict(X_val, verbose=0), axis=1))

    print(f"        Acc treino: {acc_train:.4f} | Acc val: {acc_val:.4f}")
    print(f"        Epocas treinadas: {len(history.history['loss'])}")

    return {
        "nome": "Rede Neural (MLP)",
        "modelo": modelo,
        "acc_train": acc_train,
        "acc_val": acc_val,
        "tipo": "keras",
        "history": history.history,
    }

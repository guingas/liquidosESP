"""
============================================================
 PIPELINE DE PRÉ-PROCESSAMENTO
 Limpeza, normalização e divisão dos dados
============================================================
 Etapas:
   1. Compensação de temperatura na condutividade
   2. Engenharia de features (razões espectrais)
   3. Tratamento de valores ausentes
   4. Normalização (StandardScaler)
   5. Divisão treino / validação / teste (estratificada)
============================================================
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import sys
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import (
    SENSOR_NAMES,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
    DATA_PROCESSED,
)


def compensar_temperatura_condutividade(df: pd.DataFrame,
                                        alpha: float = 0.02) -> pd.DataFrame:
    """
    Normaliza a condutividade elétrica para 25°C.
    Fórmula: sigma_25 = sigma_T / (1 + alpha * (T - 25))
    """
    df = df.copy()
    temp = df["temperatura_C"]
    cond = df["condutividade_uS"]
    df["condutividade_25C"] = cond / (1 + alpha * (temp - 25))
    return df


def criar_features_espectrais(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features derivadas a partir dos canais espectrais do AS7341.
    """
    df = df.copy()
    eps = 1e-6

    df["ratio_azul_vermelho"] = df["spec_F3_480nm"] / (df["spec_F7_630nm"] + eps)
    df["ratio_verde_vermelho"] = df["spec_F4_515nm"] / (df["spec_F7_630nm"] + eps)
    df["ratio_nir_clear"] = df["spec_NIR"] / (df["spec_Clear"] + eps)
    df["ratio_violeta_laranja"] = df["spec_F1_415nm"] / (df["spec_F6_590nm"] + eps)

    spec_cols = [c for c in df.columns if c.startswith("spec_")]
    df["spectral_mean"] = df[spec_cols].mean(axis=1)
    df["spectral_std"] = df[spec_cols].std(axis=1)
    df["spectral_range"] = df[spec_cols].max(axis=1) - df[spec_cols].min(axis=1)

    return df


def tratar_valores_ausentes(df: pd.DataFrame) -> pd.DataFrame:
    """Trata missing values com mediana por classe."""
    df = df.copy()
    feature_cols = [c for c in df.columns if c in SENSOR_NAMES or
                    c.startswith("ratio_") or c.startswith("spectral_") or
                    c == "condutividade_25C"]

    for col in feature_cols:
        if df[col].isna().any():
            if "subtipo" in df.columns:
                df[col] = df.groupby("subtipo")[col].transform(
                    lambda x: x.fillna(x.median())
                )
            df[col] = df[col].fillna(df[col].median())

    return df


def preparar_dados(df: pd.DataFrame,
                   target_col: str = "tipo",
                   seed: int = RANDOM_SEED):
    """
    Pipeline completo de pré-processamento.

    Returns:
        Dicionário com X_train/val/test, y_train/val/test,
        feature_names, scaler, label_encoder
    """
    print("\n" + "=" * 60)
    print(" PRE-PROCESSAMENTO DOS DADOS")
    print("=" * 60)

    # Etapa 1: Compensação de temperatura
    print("  [1/5] Compensando condutividade para 25C...")
    df = compensar_temperatura_condutividade(df)

    # Etapa 2: Engenharia de features
    print("  [2/5] Criando features espectrais derivadas...")
    df = criar_features_espectrais(df)

    # Etapa 3: Tratar missing values
    print("  [3/5] Tratando valores ausentes...")
    df = tratar_valores_ausentes(df)

    # Etapa 4: Selecionar features e target
    print("  [4/5] Selecionando features...")
    colunas_excluir = ["tipo", "subtipo", "adulteracao_pct", "potabilidade",
                       "genuina", "marca_cerveja", "genuina_label", "marca",
                       "timestamp", "sensor_hw", "augmentation"]
    feature_names = [c for c in df.columns if c not in colunas_excluir
                     and c != target_col]
    X = df[feature_names].values
    y_raw = df[target_col].values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    print(f"        Classes: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")

    # Etapa 5: Dividir dados
    print(f"  [5/5] Dividindo dados ({TRAIN_RATIO:.0%}/{VAL_RATIO:.0%}/{TEST_RATIO:.0%})...")

    val_test_ratio = VAL_RATIO + TEST_RATIO
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=val_test_ratio, random_state=seed, stratify=y
    )

    test_fraction = TEST_RATIO / val_test_ratio
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_fraction,
        random_state=seed, stratify=y_temp
    )

    print(f"        Treino:     {X_train.shape[0]} amostras")
    print(f"        Validacao:  {X_val.shape[0]} amostras")
    print(f"        Teste:      {X_test.shape[0]} amostras")

    # Normalização
    print("        Normalizando com StandardScaler...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Salvar artefatos
    scaler_path = os.path.join(DATA_PROCESSED, "scaler.joblib")
    joblib.dump(scaler, scaler_path)

    encoder_path = os.path.join(DATA_PROCESSED, "label_encoder.joblib")
    joblib.dump(label_encoder, encoder_path)

    # Salvar parâmetros de normalização para o ESP32
    norm_params = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "feature_names": feature_names,
    }
    norm_path = os.path.join(DATA_PROCESSED, "normalization_params.joblib")
    joblib.dump(norm_params, norm_path)

    print(f"        Scaler salvo em: {scaler_path}")
    print("=" * 60)

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "feature_names": feature_names,
        "scaler": scaler,
        "label_encoder": label_encoder,
    }

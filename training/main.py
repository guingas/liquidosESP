"""
============================================================
 MAIN — PIPELINE PRINCIPAL Lab7 (EXPANDIDO)
 Identificação de Líquidos com LiquidsML no ESP32
============================================================
 Pipeline Hierárquico:
   ETAPA 1: Gerar dados sintéticos expandidos (22 classes)
   ETAPA 2: Data Augmentation
   ETAPA 3: Classificar TIPO (agua/gasolina/suco/alcoolica)
   ETAPA 4a: Água → variante + potabilidade
   ETAPA 4b: Gasolina → pura vs adulterada (grau)
   ETAPA 4c: Suco → tipo (laranja, uva, limão, maçã, manga)
   ETAPA 4d: Alcoólica → subtipo
   ETAPA 4e: Cerveja → marca (Heineken vs Stella vs Budweiser vs IPA)
   ETAPA 4f: Alcoólica → genuína vs adulterada
   ETAPA 5: Converter melhor modelo para TFLite
   ETAPA 6: Simulação de cenários reais
============================================================
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config.settings import RESULTS_DIR, MODELS_DIR, FIGURES_DIR
from src.data_generation.synthetic_sensors import (
    gerar_dataset_completo, PERFIS_SENSORES, MAPA_TIPO, SENSOR_NAMES,
    MAPA_POTABILIDADE, MAPA_GENUINIDADE_ALCOOL, MAPA_MARCA_CERVEJA,
)
from src.data_generation.augmentation import executar_augmentation
from src.preprocessing.pipeline import preparar_dados
from src.models.random_forest import treinar_random_forest
from src.models.xgboost_model import treinar_xgboost
from src.models.neural_network import treinar_rede_neural
from src.models.cnn_model import treinar_cnn
from src.models.svm_model import treinar_svm
from src.models.knn_model import treinar_knn
from src.models.tflite_converter import converter_para_tflite, exportar_parametros_normalizacao
from src.evaluation.compare_models import executar_avaliacao_completa
from src.feature_analysis.sensor_importance import executar_analise_sensores

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def treinar_todos_modelos(dados: dict, n_classes: int) -> list:
    """Treina todos os 6 modelos de ML."""
    X_train = dados["X_train"]
    y_train = dados["y_train"]
    X_val = dados["X_val"]
    y_val = dados["y_val"]

    print("\n" + "=" * 60)
    print(" TREINAMENTO DOS MODELOS DE MACHINE LEARNING")
    print(f" (n_classes={n_classes}, n_features={X_train.shape[1]}, "
          f"n_treino={X_train.shape[0]})")
    print("=" * 60)

    modelos = []
    modelos.append(treinar_random_forest(X_train, y_train, X_val, y_val))
    modelos.append(treinar_xgboost(X_train, y_train, X_val, y_val))
    modelos.append(treinar_knn(X_train, y_train, X_val, y_val))
    modelos.append(treinar_svm(X_train, y_train, X_val, y_val))
    modelos.append(treinar_rede_neural(X_train, y_train, X_val, y_val, n_classes))
    modelos.append(treinar_cnn(X_train, y_train, X_val, y_val, n_classes))

    print("\n  +----------------------+----------------------+")
    print("  | Modelo               | Acuracia Validacao   |")
    print("  +----------------------+----------------------+")
    for m in modelos:
        nome = m["nome"][:20].ljust(20)
        acc = f"{m['acc_val'] * 100:.2f}%".rjust(18)
        print(f"  | {nome} | {acc}   |")
    print("  +----------------------+----------------------+")

    return modelos


def executar_pipeline(df, target_col: str, nome_problema: str, skip_importance=False):
    """Executa pipeline completo para um problema de classificação."""
    sufixo = f"_{nome_problema}"

    print("\n")
    print("+" + "=" * 58 + "+")
    print(f"| PROBLEMA: {nome_problema.upper():<47s}|")
    print(f"|  Target:  {target_col:<47s}|")
    print("+" + "=" * 58 + "+")

    dados = preparar_dados(df, target_col=target_col)
    n_classes = len(dados["label_encoder"].classes_)
    modelos = treinar_todos_modelos(dados, n_classes)

    df_comparacao = executar_avaliacao_completa(
        modelos_treinados=modelos,
        X_test=dados["X_test"],
        y_test=dados["y_test"],
        label_encoder=dados["label_encoder"],
        sufixo=sufixo,
    )

    ranking = None
    if not skip_importance:
        ranking = executar_analise_sensores(
            modelos_treinados=modelos,
            X_test=dados["X_test"],
            y_test=dados["y_test"],
            feature_names=dados["feature_names"],
            sufixo=sufixo,
        )

    return df_comparacao, ranking, modelos, dados


def selecionar_melhor_modelo_keras(modelos: list) -> dict:
    keras_models = [m for m in modelos if m["tipo"] == "keras"]
    if not keras_models:
        return None
    return max(keras_models, key=lambda m: m["acc_val"])


# ============================================================
# SIMULAÇÃO DE CENÁRIOS REAIS
# ============================================================

def simular_amostra(perfil_nome: str, rng: np.random.Generator) -> np.ndarray:
    """Gera uma amostra simulada de um perfil de sensor."""
    perfil = PERFIS_SENSORES[perfil_nome]
    amostra = []
    for sensor in SENSOR_NAMES:
        media, desvio = perfil[sensor]
        val = rng.normal(media, desvio)
        if sensor in ["condutividade_uS", "acustico_freq_Hz"]:
            val = max(val, 0)
        elif sensor == "pH":
            val = np.clip(val, 0, 14)
        elif sensor.startswith("spec_"):
            val = np.clip(val, 0, 1)
        amostra.append(val)
    return np.array(amostra)


def preprocessar_amostra_para_modelo(amostra_raw: np.ndarray, scaler,
                                     feature_names: list) -> np.ndarray:
    """Aplica feature engineering e normalização a uma amostra."""
    df_tmp = pd.DataFrame([amostra_raw], columns=SENSOR_NAMES)

    # Compensação de temperatura
    eps = 1e-6
    alpha = 0.02
    temp = df_tmp["temperatura_C"].values[0]
    cond = df_tmp["condutividade_uS"].values[0]
    df_tmp["condutividade_25C"] = cond / (1 + alpha * (temp - 25))

    # Features espectrais
    df_tmp["ratio_azul_vermelho"] = df_tmp["spec_F3_480nm"] / (df_tmp["spec_F7_630nm"] + eps)
    df_tmp["ratio_verde_vermelho"] = df_tmp["spec_F4_515nm"] / (df_tmp["spec_F7_630nm"] + eps)
    df_tmp["ratio_nir_clear"] = df_tmp["spec_NIR"] / (df_tmp["spec_Clear"] + eps)
    df_tmp["ratio_violeta_laranja"] = df_tmp["spec_F1_415nm"] / (df_tmp["spec_F6_590nm"] + eps)

    spec_cols = [c for c in df_tmp.columns if c.startswith("spec_")]
    df_tmp["spectral_mean"] = df_tmp[spec_cols].mean(axis=1)
    df_tmp["spectral_std"] = df_tmp[spec_cols].std(axis=1)
    df_tmp["spectral_range"] = df_tmp[spec_cols].max(axis=1) - df_tmp[spec_cols].min(axis=1)

    # Extrair apenas as features que o modelo espera, na ordem correta
    X = df_tmp[feature_names].values
    return scaler.transform(X)


def classificar_amostra(amostra_raw, modelo, scaler, label_encoder, feature_names,
                        modelo_tipo="sklearn"):
    """Classifica uma amostra usando um modelo treinado."""
    X = preprocessar_amostra_para_modelo(amostra_raw, scaler, feature_names)
    if modelo_tipo == "keras":
        pred_idx = int(np.argmax(modelo.predict(X, verbose=0), axis=1)[0])
    else:
        pred_idx = int(modelo.predict(X)[0])
    return label_encoder.inverse_transform([pred_idx])[0]


def _classificar(amostra_raw, m_info, dados_entry):
    """Helper: classificar amostra usando modelos_dict entry."""
    return classificar_amostra(
        amostra_raw,
        m_info["melhor"],
        dados_entry["scaler"],
        dados_entry["label_encoder"],
        dados_entry["feature_names"],
        modelo_tipo=m_info.get("tipo", "sklearn"),
    )


def executar_simulacao_cenarios(modelos_dict: dict, dados_dict: dict):
    """Simula cenários reais e testa a classificação hierárquica."""
    print("\n")
    print("=" * 70)
    print("  SIMULACAO DE CENARIOS REAIS — VALIDACAO PRE-FLASH")
    print("=" * 70)

    rng = np.random.default_rng(999)

    cenarios = [
        ("gasolina_pura",                 "Gasolina PURA do posto"),
        ("gasolina_adulterada_10",        "Gasolina adulterada 10%"),
        ("gasolina_adulterada_30",        "Gasolina adulterada 30% (GRAVE)"),
        ("agua_torneira",                 "Agua da torneira (potavel)"),
        ("agua_mineral",                  "Agua mineral (potavel)"),
        ("agua_contaminada",              "Agua CONTAMINADA (nao potavel!)"),
        ("suco_laranja",                  "Suco de LARANJA natural"),
        ("suco_manga",                    "Suco de MANGA natural"),
        ("suco_uva",                      "Suco de UVA integral"),
        ("alcoolica_cerveja_heineken",    "Cerveja HEINEKEN"),
        ("alcoolica_cerveja_stella",      "Cerveja STELLA ARTOIS"),
        ("alcoolica_cerveja_budweiser",   "Cerveja BUDWEISER"),
        ("alcoolica_cerveja_ipa",         "Cerveja IPA"),
        ("alcoolica_vodka",               "Vodka GENUINA"),
        ("alcoolica_vodka_adulterada",    "Vodka ADULTERADA (metanol!)"),
        ("alcoolica_cachaca",             "Cachaca GENUINA"),
        ("alcoolica_cachaca_adulterada",  "Cachaca ADULTERADA"),
        ("alcoolica_whisky",              "Whisky GENUINO"),
    ]

    total_ok = 0
    total = 0

    for perfil_nome, descricao in cenarios:
        print(f"\n  {'='*60}")
        print(f"  Cenario: {descricao}")
        print(f"  {'='*60}")
        amostra = simular_amostra(perfil_nome, rng)

        # 1) TIPO
        if "tipo" in modelos_dict:
            d = dados_dict["tipo"]
            m_info = modelos_dict["tipo"]
            tipo_pred = _classificar(amostra, m_info, d)
            tipo_real = MAPA_TIPO[perfil_nome]
            ok = tipo_pred == tipo_real
            total += 1
            total_ok += int(ok)
            print(f"    Tipo:        {tipo_pred:<15s} (real: {tipo_real}) [{'OK' if ok else 'ERRO'}]")

        tipo_key = MAPA_TIPO[perfil_nome]

        # 2) Subclassificação por tipo
        if tipo_key == "gasolina" and "gasolina" in modelos_dict:
            d = dados_dict["gasolina"]
            m_info = modelos_dict["gasolina"]
            sub_pred = _classificar(amostra, m_info, d)
            ok = sub_pred == perfil_nome
            total += 1
            total_ok += int(ok)
            adulterada = "pura" not in sub_pred
            print(f"    Subtipo:     {sub_pred:<35s} [{'OK' if ok else 'ERRO'}]")
            print(f"    Adulterada:  {'SIM — NAO USE NO CARRO!' if adulterada else 'NAO — OK para uso'}")

        elif tipo_key == "agua" and "agua" in modelos_dict:
            d = dados_dict["agua"]
            m_info = modelos_dict["agua"]
            sub_pred = _classificar(amostra, m_info, d)
            ok = sub_pred == perfil_nome
            total += 1
            total_ok += int(ok)
            potavel = MAPA_POTABILIDADE.get(perfil_nome, -1) == 1
            cond_val = amostra[1]
            ph_val = amostra[2]
            print(f"    Variante:    {sub_pred:<35s} [{'OK' if ok else 'ERRO'}]")
            print(f"    Potavel:     {'SIM — propria para consumo' if potavel else 'NAO — IMPROPRIA PARA CONSUMO!'}")
            print(f"    pH:          {ph_val:.1f} {'(normal 6.5-8.5)' if 6.5 <= ph_val <= 8.5 else '(FORA DO PADRAO!)'}")
            print(f"    Condut.:     {cond_val:.0f} uS/cm {'(normal <1000)' if cond_val < 1000 else '(ELEVADA — contaminantes!)'}")
            turb = 1.0 - amostra[11]  # spec_Clear invertido = turbidez
            print(f"    Turbidez:    {turb:.2f} {'(limpa)' if turb < 0.3 else '(TURVA!)'}")

        elif tipo_key == "suco" and "suco" in modelos_dict:
            d = dados_dict["suco"]
            m_info = modelos_dict["suco"]
            sub_pred = _classificar(amostra, m_info, d)
            ok = sub_pred == perfil_nome
            total += 1
            total_ok += int(ok)
            print(f"    Sabor:       {sub_pred:<35s} [{'OK' if ok else 'ERRO'}]")

        elif tipo_key == "alcoolica":
            # Subtipo geral
            if "alcoolica" in modelos_dict:
                d = dados_dict["alcoolica"]
                m_info = modelos_dict["alcoolica"]
                sub_pred = _classificar(amostra, m_info, d)
                ok = sub_pred == perfil_nome
                total += 1
                total_ok += int(ok)
                print(f"    Subtipo:     {sub_pred:<35s} [{'OK' if ok else 'ERRO'}]")

            # Genuína vs adulterada
            if "alcoolica_genuina" in modelos_dict:
                d = dados_dict["alcoolica_genuina"]
                m_info = modelos_dict["alcoolica_genuina"]
                gen_pred = _classificar(amostra, m_info, d)
                gen_real = "genuina" if MAPA_GENUINIDADE_ALCOOL.get(perfil_nome, 1) == 1 else "adulterada"
                ok = gen_pred == gen_real
                total += 1
                total_ok += int(ok)
                if gen_pred == "adulterada":
                    print(f"    Genuina:     ADULTERADA — PERIGO! NAO CONSUMA! [{'OK' if ok else 'ERRO'}]")
                else:
                    print(f"    Genuina:     GENUINA — segura para consumo [{'OK' if ok else 'ERRO'}]")

            # Marca de cerveja
            if "cerveja_marca" in modelos_dict and "cerveja" in perfil_nome:
                d = dados_dict["cerveja_marca"]
                m_info = modelos_dict["cerveja_marca"]
                marca_pred = _classificar(amostra, m_info, d)
                marca_real = MAPA_MARCA_CERVEJA.get(perfil_nome, "?")
                ok = marca_pred == marca_real
                total += 1
                total_ok += int(ok)
                print(f"    Marca:       {marca_pred:<15s} (real: {marca_real}) [{'OK' if ok else 'ERRO'}]")

    # Resumo
    print("\n" + "=" * 70)
    print("  RESUMO DA SIMULACAO DE CENARIOS")
    print("=" * 70)
    pct = total_ok / total * 100 if total > 0 else 0
    print(f"  Total de verificacoes: {total}")
    print(f"  Acertos:               {total_ok} ({pct:.1f}%)")
    print(f"  Erros:                 {total - total_ok}")
    if pct >= 90:
        print(f"  Status:                APROVADO — pronto para flash!")
    elif pct >= 70:
        print(f"  Status:                ACEITAVEL — revisar erros antes do flash")
    else:
        print(f"  Status:                REPROVADO — modelos precisam de ajuste")
    print("=" * 70)

    return total_ok, total


def main():
    """Ponto de entrada principal."""
    print("\n")
    print("+" + "=" * 62 + "+")
    print("|                                                                |")
    print("|   Lab7 — IDENTIFICACAO DE LIQUIDOS COM ESP32 (EXPANDIDO)       |")
    print("|                                                                |")
    print("|   22 classes: agua (4) + gasolina (4) + suco (5) + alcool (11) |")
    print("|   Sensores: AS7341 + DS18B20 + TDS + pH + Piezo               |")
    print("|   Modelos: RF | XGB | k-NN | SVM | MLP | CNN                  |")
    print("|                                                                |")
    print("|   Capacidades:                                                 |")
    print("|     - Gasolina pura vs adulterada (10/20/30%)                  |")
    print("|     - Agua potavel vs contaminada + caracteristicas            |")
    print("|     - Suco: laranja vs manga vs uva vs limao vs maca          |")
    print("|     - Cerveja: Heineken vs Stella vs Budweiser vs IPA          |")
    print("|     - Bebida alcoolica genuina vs adulterada                   |")
    print("|                                                                |")
    print("+" + "=" * 62 + "+")

    modelos_dict = {}
    dados_dict = {}

    # =========================================================
    # ETAPA 1: GERAR DADOS
    # =========================================================
    df = gerar_dataset_completo()

    # =========================================================
    # ETAPA 2: AUGMENTATION
    # =========================================================
    df_aug = executar_augmentation(df, n_rounds=1)

    # =========================================================
    # ETAPA 3: TIPO (agua/gasolina/suco/alcoolica)
    # =========================================================
    print("\n" + "#" * 60)
    print("  ETAPA 3 — TIPO DO LIQUIDO")
    print("#" * 60)
    df_comp, ranking, modelos, dados = executar_pipeline(
        df_aug, target_col="tipo", nome_problema="tipo_liquido"
    )
    melhor = max(modelos, key=lambda m: m["acc_val"])
    modelos_dict["tipo"] = {"modelos": modelos, "melhor": melhor["modelo"], "tipo": melhor["tipo"]}
    dados_dict["tipo"] = dados

    # =========================================================
    # ETAPA 4a: ÁGUA → variante + potabilidade
    # =========================================================
    print("\n" + "#" * 60)
    print("  ETAPA 4a — AGUA: VARIANTE + POTABILIDADE")
    print("#" * 60)
    df_agua = df_aug[df_aug["tipo"] == "agua"].copy()
    if len(df_agua) > 0:
        df_c, rk, mods, dad = executar_pipeline(
            df_agua, target_col="subtipo", nome_problema="agua_variante",
            skip_importance=True
        )
        melhor = max(mods, key=lambda m: m["acc_val"])
        modelos_dict["agua"] = {"modelos": mods, "melhor": melhor["modelo"], "tipo": melhor["tipo"]}
        dados_dict["agua"] = dad

    # =========================================================
    # ETAPA 4b: GASOLINA → grau de adulteração
    # =========================================================
    print("\n" + "#" * 60)
    print("  ETAPA 4b — GASOLINA: PURA vs ADULTERADA")
    print("#" * 60)
    df_gas = df_aug[df_aug["tipo"] == "gasolina"].copy()
    if len(df_gas) > 0:
        df_c, rk, mods, dad = executar_pipeline(
            df_gas, target_col="subtipo", nome_problema="gasolina_adulteracao",
            skip_importance=True
        )
        melhor = max(mods, key=lambda m: m["acc_val"])
        modelos_dict["gasolina"] = {"modelos": mods, "melhor": melhor["modelo"], "tipo": melhor["tipo"]}
        dados_dict["gasolina"] = dad

    # =========================================================
    # ETAPA 4c: SUCO → sabor
    # =========================================================
    print("\n" + "#" * 60)
    print("  ETAPA 4c — SUCO: SABOR (laranja/manga/uva/limao/maca)")
    print("#" * 60)
    df_suco = df_aug[df_aug["tipo"] == "suco"].copy()
    if len(df_suco) > 0:
        df_c, rk, mods, dad = executar_pipeline(
            df_suco, target_col="subtipo", nome_problema="suco_sabor",
            skip_importance=True
        )
        melhor = max(mods, key=lambda m: m["acc_val"])
        modelos_dict["suco"] = {"modelos": mods, "melhor": melhor["modelo"], "tipo": melhor["tipo"]}
        dados_dict["suco"] = dad

    # =========================================================
    # ETAPA 4d: ALCOÓLICA → subtipo completo (11 classes)
    # =========================================================
    print("\n" + "#" * 60)
    print("  ETAPA 4d — ALCOOLICA: SUBTIPO (11 classes)")
    print("#" * 60)
    df_alc = df_aug[df_aug["tipo"] == "alcoolica"].copy()
    if len(df_alc) > 0:
        df_c, rk, mods, dad = executar_pipeline(
            df_alc, target_col="subtipo", nome_problema="alcoolica_subtipo",
            skip_importance=True
        )
        melhor = max(mods, key=lambda m: m["acc_val"])
        modelos_dict["alcoolica"] = {"modelos": mods, "melhor": melhor["modelo"], "tipo": melhor["tipo"]}
        dados_dict["alcoolica"] = dad

    # =========================================================
    # ETAPA 4e: CERVEJA → marca específica
    # =========================================================
    print("\n" + "#" * 60)
    print("  ETAPA 4e — CERVEJA: MARCA (Heineken/Stella/Budweiser/IPA)")
    print("#" * 60)
    df_cerv = df_aug[df_aug["subtipo"].str.contains("cerveja", na=False)].copy()
    if len(df_cerv) > 0:
        df_cerv["marca"] = df_cerv["subtipo"].map(MAPA_MARCA_CERVEJA)
        df_cerv = df_cerv.dropna(subset=["marca"])
        if len(df_cerv) > 0:
            df_c, rk, mods, dad = executar_pipeline(
                df_cerv, target_col="marca", nome_problema="cerveja_marca",
                skip_importance=True
            )
            melhor = max(mods, key=lambda m: m["acc_val"])
            modelos_dict["cerveja_marca"] = {"modelos": mods, "melhor": melhor["modelo"], "tipo": melhor["tipo"]}
            dados_dict["cerveja_marca"] = dad

    # =========================================================
    # ETAPA 4f: ALCOÓLICA → genuína vs adulterada
    # =========================================================
    print("\n" + "#" * 60)
    print("  ETAPA 4f — ALCOOLICA: GENUINA vs ADULTERADA")
    print("#" * 60)
    df_alc2 = df_aug[df_aug["tipo"] == "alcoolica"].copy()
    if len(df_alc2) > 0:
        df_alc2["genuina_label"] = df_alc2["subtipo"].map(
            lambda s: "genuina" if MAPA_GENUINIDADE_ALCOOL.get(s, 1) == 1 else "adulterada"
        )
        df_c, rk, mods, dad = executar_pipeline(
            df_alc2, target_col="genuina_label", nome_problema="alcoolica_genuina",
            skip_importance=True
        )
        melhor = max(mods, key=lambda m: m["acc_val"])
        modelos_dict["alcoolica_genuina"] = {"modelos": mods, "melhor": melhor["modelo"], "tipo": melhor["tipo"]}
        dados_dict["alcoolica_genuina"] = dad

    # =========================================================
    # ETAPA 5: CONVERTER PARA TFLITE
    # =========================================================
    print("\n" + "#" * 60)
    print("  ETAPA 5 — CONVERSAO PARA TFLITE (ESP32)")
    print("#" * 60)

    melhor_keras = selecionar_melhor_modelo_keras(modelos_dict["tipo"]["modelos"])
    if melhor_keras:
        print(f"\n  Melhor modelo Keras (tipo): {melhor_keras['nome']} "
              f"(acc_val={melhor_keras['acc_val']:.4f})")
        tflite_path = converter_para_tflite(
            modelo_keras=melhor_keras["modelo"],
            X_calibration=dados_dict["tipo"]["X_train"],
            nome_modelo="liquid_classifier",
            quantizar=True,
        )
        exportar_parametros_normalizacao()

    # =========================================================
    # ETAPA 6: SIMULAÇÃO DE CENÁRIOS REAIS
    # =========================================================
    total_ok, total = executar_simulacao_cenarios(modelos_dict, dados_dict)

    # =========================================================
    # RESUMO FINAL
    # =========================================================
    print("\n")
    print("+" + "=" * 62 + "+")
    print("|              PIPELINE CONCLUIDO COM SUCESSO                    |")
    print("+" + "=" * 62 + "+")
    print("|  Etapa 3:  TIPO (agua/gasolina/suco/alcoolica)                |")
    print("|  Etapa 4a: Agua -> variante + potabilidade + pH + condut.     |")
    print("|  Etapa 4b: Gasolina -> pura vs adulterada (grau 10/20/30%)    |")
    print("|  Etapa 4c: Suco -> sabor (laranja/manga/uva/limao/maca)       |")
    print("|  Etapa 4d: Alcoolica -> subtipo completo (11 classes)          |")
    print("|  Etapa 4e: Cerveja -> marca (Heineken/Stella/Bud/IPA)         |")
    print("|  Etapa 4f: Alcoolica -> genuina vs adulterada                 |")
    print("|  Etapa 5:  Conversao TFLite (int8) para ESP32                 |")
    print("|  Etapa 6:  Simulacao 18 cenarios reais                        |")
    print("+" + "=" * 62 + "+")
    print(f"\n  Simulacao: {total_ok}/{total} acertos ({total_ok/total*100:.1f}%)")
    print(f"  Resultados: {RESULTS_DIR}")
    print(f"  Modelos:    {MODELS_DIR}")

    # Lista arquivos
    print("\n  Arquivos gerados:")
    for base_dir in [RESULTS_DIR, MODELS_DIR]:
        for root, dirs, files in os.walk(base_dir):
            for f in sorted(files):
                caminho = os.path.join(root, f)
                tamanho = os.path.getsize(caminho)
                rel = os.path.relpath(caminho, PROJECT_ROOT)
                print(f"    -> {rel:<55s} ({tamanho:>8,d} bytes)")

    print()


if __name__ == "__main__":
    main()

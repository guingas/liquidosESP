"""
============================================================
 GERADOR DE DADOS SINTÉTICOS DE SENSORES — EXPANDIDO
 Classes: água, gasolina (pura/adulterada), sucos, bebidas alcoólicas
============================================================
 Perfis baseados em propriedades físico-químicas de literatura.
 Referências:
   - WHO Guidelines for Drinking-water Quality
   - BJCP Style Guidelines 2021
   - ANP Resolução 40/2013 (teor de etanol anidro na gasolina)
   - CRC Handbook of Chemistry & Physics
   - USDA FoodData Central (sucos)
============================================================
"""

import numpy as np
import pandas as pd
import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import (
    N_AMOSTRAS_POR_CLASSE,
    RANDOM_SEED,
    SENSOR_NAMES,
    DATA_SYNTHETIC,
    DATA_METADATA,
)

# ============================================================
# PERFIS FÍSICO-QUÍMICOS DOS LÍQUIDOS
# ============================================================
# Cada perfil: {sensor_name: [média, desvio_padrão]}
#
# Ordem: [temp, condutividade, pH,
#          F1(415), F2(445), F3(480), F4(515), F5(555), F6(590),
#          F7(630), F8(680), Clear, NIR,
#          acustico]

PERFIS_SENSORES = {
    # =========================================================
    # ÁGUA
    # =========================================================
    # ---------------------------------------------------------
    # ÁGUA DA TORNEIRA (potável, tratada)
    # Condutividade: 200-800 µS/cm, pH 6.5-8.0
    # ---------------------------------------------------------
    "agua_torneira": {
        "temperatura_C":     [22.0,  2.0],
        "condutividade_uS":  [450.0, 150.0],
        "pH":                [7.2,   0.4],
        "spec_F1_415nm":     [0.90,  0.05],
        "spec_F2_445nm":     [0.92,  0.04],
        "spec_F3_480nm":     [0.93,  0.04],
        "spec_F4_515nm":     [0.91,  0.04],
        "spec_F5_555nm":     [0.90,  0.05],
        "spec_F6_590nm":     [0.89,  0.05],
        "spec_F7_630nm":     [0.88,  0.05],
        "spec_F8_680nm":     [0.85,  0.06],
        "spec_Clear":        [0.91,  0.04],
        "spec_NIR":          [0.70,  0.08],
        "acustico_freq_Hz":  [1482,  8.0],
    },
    # ---------------------------------------------------------
    # ÁGUA MINERAL (SEM GÁS) — potável, baixa mineralização
    # ---------------------------------------------------------
    "agua_mineral": {
        "temperatura_C":     [20.0,  2.0],
        "condutividade_uS":  [250.0, 80.0],
        "pH":                [7.5,   0.3],
        "spec_F1_415nm":     [0.93,  0.03],
        "spec_F2_445nm":     [0.94,  0.03],
        "spec_F3_480nm":     [0.95,  0.03],
        "spec_F4_515nm":     [0.94,  0.03],
        "spec_F5_555nm":     [0.93,  0.03],
        "spec_F6_590nm":     [0.92,  0.03],
        "spec_F7_630nm":     [0.91,  0.04],
        "spec_F8_680nm":     [0.88,  0.05],
        "spec_Clear":        [0.93,  0.03],
        "spec_NIR":          [0.72,  0.07],
        "acustico_freq_Hz":  [1480,  6.0],
    },
    # ---------------------------------------------------------
    # ÁGUA MINERAL COM GÁS — CO2 reduz pH, bolhas alteram óptica
    # ---------------------------------------------------------
    "agua_mineral_gas": {
        "temperatura_C":     [8.0,   3.0],
        "condutividade_uS":  [350.0, 100.0],
        "pH":                [5.5,   0.5],
        "spec_F1_415nm":     [0.80,  0.08],
        "spec_F2_445nm":     [0.82,  0.07],
        "spec_F3_480nm":     [0.83,  0.07],
        "spec_F4_515nm":     [0.82,  0.07],
        "spec_F5_555nm":     [0.81,  0.08],
        "spec_F6_590nm":     [0.80,  0.08],
        "spec_F7_630nm":     [0.78,  0.08],
        "spec_F8_680nm":     [0.75,  0.09],
        "spec_Clear":        [0.80,  0.07],
        "spec_NIR":          [0.65,  0.10],
        "acustico_freq_Hz":  [1450,  20.0],
    },
    # ---------------------------------------------------------
    # ÁGUA CONTAMINADA — NÃO POTÁVEL
    # Alta condutividade (metais pesados/sais), pH fora do padrão,
    # turbidez alta (partículas em suspensão)
    # Ref: WHO Guidelines — limites de potabilidade
    # ---------------------------------------------------------
    "agua_contaminada": {
        "temperatura_C":     [24.0,  4.0],
        "condutividade_uS":  [1200.0, 400.0],
        "pH":                [5.5,   1.2],
        "spec_F1_415nm":     [0.55,  0.15],
        "spec_F2_445nm":     [0.58,  0.14],
        "spec_F3_480nm":     [0.60,  0.14],
        "spec_F4_515nm":     [0.62,  0.14],
        "spec_F5_555nm":     [0.60,  0.14],
        "spec_F6_590nm":     [0.57,  0.14],
        "spec_F7_630nm":     [0.53,  0.14],
        "spec_F8_680nm":     [0.48,  0.15],
        "spec_Clear":        [0.55,  0.13],
        "spec_NIR":          [0.50,  0.12],
        "acustico_freq_Hz":  [1475,  15.0],
    },
    # =========================================================
    # GASOLINA
    # =========================================================
    # ---------------------------------------------------------
    # GASOLINA PURA (tipo C padrão ANP, ~27% AEAC)
    # Muito baixa condutividade, alta transmissão NIR
    # ---------------------------------------------------------
    "gasolina_pura": {
        "temperatura_C":     [25.0,  3.0],
        "condutividade_uS":  [0.5,   0.3],
        "pH":                [6.0,   0.5],
        "spec_F1_415nm":     [0.45,  0.08],
        "spec_F2_445nm":     [0.50,  0.07],
        "spec_F3_480nm":     [0.55,  0.07],
        "spec_F4_515nm":     [0.60,  0.06],
        "spec_F5_555nm":     [0.70,  0.06],
        "spec_F6_590nm":     [0.75,  0.05],
        "spec_F7_630nm":     [0.80,  0.05],
        "spec_F8_680nm":     [0.85,  0.04],
        "spec_Clear":        [0.65,  0.06],
        "spec_NIR":          [0.88,  0.04],
        "acustico_freq_Hz":  [1250,  15.0],
    },
    # ---------------------------------------------------------
    # GASOLINA ADULTERADA 10% (solvente / etanol excedente)
    # ---------------------------------------------------------
    "gasolina_adulterada_10": {
        "temperatura_C":     [25.0,  3.0],
        "condutividade_uS":  [5.0,   2.0],
        "pH":                [5.8,   0.6],
        "spec_F1_415nm":     [0.47,  0.08],
        "spec_F2_445nm":     [0.52,  0.07],
        "spec_F3_480nm":     [0.57,  0.07],
        "spec_F4_515nm":     [0.62,  0.07],
        "spec_F5_555nm":     [0.71,  0.06],
        "spec_F6_590nm":     [0.74,  0.06],
        "spec_F7_630nm":     [0.78,  0.06],
        "spec_F8_680nm":     [0.82,  0.05],
        "spec_Clear":        [0.64,  0.06],
        "spec_NIR":          [0.85,  0.05],
        "acustico_freq_Hz":  [1270,  18.0],
    },
    "gasolina_adulterada_20": {
        "temperatura_C":     [25.0,  3.0],
        "condutividade_uS":  [12.0,  5.0],
        "pH":                [5.5,   0.7],
        "spec_F1_415nm":     [0.50,  0.09],
        "spec_F2_445nm":     [0.55,  0.08],
        "spec_F3_480nm":     [0.60,  0.08],
        "spec_F4_515nm":     [0.65,  0.07],
        "spec_F5_555nm":     [0.72,  0.07],
        "spec_F6_590nm":     [0.73,  0.07],
        "spec_F7_630nm":     [0.76,  0.07],
        "spec_F8_680nm":     [0.79,  0.06],
        "spec_Clear":        [0.63,  0.07],
        "spec_NIR":          [0.82,  0.06],
        "acustico_freq_Hz":  [1290,  20.0],
    },
    "gasolina_adulterada_30": {
        "temperatura_C":     [25.0,  3.0],
        "condutividade_uS":  [25.0,  10.0],
        "pH":                [5.2,   0.8],
        "spec_F1_415nm":     [0.53,  0.10],
        "spec_F2_445nm":     [0.58,  0.09],
        "spec_F3_480nm":     [0.63,  0.08],
        "spec_F4_515nm":     [0.67,  0.08],
        "spec_F5_555nm":     [0.73,  0.07],
        "spec_F6_590nm":     [0.72,  0.07],
        "spec_F7_630nm":     [0.74,  0.07],
        "spec_F8_680nm":     [0.76,  0.07],
        "spec_Clear":        [0.62,  0.08],
        "spec_NIR":          [0.78,  0.07],
        "acustico_freq_Hz":  [1320,  22.0],
    },
    # =========================================================
    # SUCOS NATURAIS
    # =========================================================
    "suco_laranja": {
        "temperatura_C":     [10.0,  4.0],
        "condutividade_uS":  [2800,  400.0],
        "pH":                [3.5,   0.3],
        "spec_F1_415nm":     [0.10,  0.05],
        "spec_F2_445nm":     [0.15,  0.06],
        "spec_F3_480nm":     [0.25,  0.07],
        "spec_F4_515nm":     [0.45,  0.08],
        "spec_F5_555nm":     [0.65,  0.07],
        "spec_F6_590nm":     [0.80,  0.06],
        "spec_F7_630nm":     [0.70,  0.07],
        "spec_F8_680nm":     [0.50,  0.08],
        "spec_Clear":        [0.42,  0.07],
        "spec_NIR":          [0.55,  0.08],
        "acustico_freq_Hz":  [1510,  12.0],
    },
    "suco_uva": {
        "temperatura_C":     [10.0,  4.0],
        "condutividade_uS":  [3200,  500.0],
        "pH":                [3.3,   0.3],
        "spec_F1_415nm":     [0.08,  0.04],
        "spec_F2_445nm":     [0.10,  0.05],
        "spec_F3_480nm":     [0.12,  0.05],
        "spec_F4_515nm":     [0.15,  0.06],
        "spec_F5_555nm":     [0.18,  0.06],
        "spec_F6_590nm":     [0.22,  0.07],
        "spec_F7_630nm":     [0.30,  0.08],
        "spec_F8_680nm":     [0.25,  0.08],
        "spec_Clear":        [0.15,  0.05],
        "spec_NIR":          [0.40,  0.09],
        "acustico_freq_Hz":  [1530,  14.0],
    },
    "suco_limao": {
        "temperatura_C":     [10.0,  4.0],
        "condutividade_uS":  [3500,  600.0],
        "pH":                [2.3,   0.3],
        "spec_F1_415nm":     [0.60,  0.08],
        "spec_F2_445nm":     [0.65,  0.07],
        "spec_F3_480nm":     [0.70,  0.07],
        "spec_F4_515nm":     [0.75,  0.06],
        "spec_F5_555nm":     [0.80,  0.06],
        "spec_F6_590nm":     [0.78,  0.06],
        "spec_F7_630nm":     [0.72,  0.07],
        "spec_F8_680nm":     [0.65,  0.08],
        "spec_Clear":        [0.72,  0.06],
        "spec_NIR":          [0.58,  0.08],
        "acustico_freq_Hz":  [1500,  10.0],
    },
    "suco_maca": {
        "temperatura_C":     [10.0,  4.0],
        "condutividade_uS":  [2500,  350.0],
        "pH":                [3.6,   0.3],
        "spec_F1_415nm":     [0.40,  0.08],
        "spec_F2_445nm":     [0.48,  0.07],
        "spec_F3_480nm":     [0.55,  0.07],
        "spec_F4_515nm":     [0.65,  0.07],
        "spec_F5_555nm":     [0.72,  0.06],
        "spec_F6_590nm":     [0.70,  0.07],
        "spec_F7_630nm":     [0.60,  0.07],
        "spec_F8_680nm":     [0.50,  0.08],
        "spec_Clear":        [0.58,  0.07],
        "spec_NIR":          [0.52,  0.08],
        "acustico_freq_Hz":  [1505,  11.0],
    },
    # ---------------------------------------------------------
    # SUCO DE MANGA — cor laranja-amarela intensa, alta turbidez
    # Mais viscoso que laranja, beta-caroteno dominante
    # pH ~3.4, condutividade moderada
    # ---------------------------------------------------------
    "suco_manga": {
        "temperatura_C":     [10.0,  4.0],
        "condutividade_uS":  [2600,  380.0],
        "pH":                [3.4,   0.3],
        "spec_F1_415nm":     [0.08,  0.04],
        "spec_F2_445nm":     [0.12,  0.05],
        "spec_F3_480nm":     [0.20,  0.07],
        "spec_F4_515nm":     [0.40,  0.08],
        "spec_F5_555nm":     [0.60,  0.07],
        "spec_F6_590nm":     [0.75,  0.06],
        "spec_F7_630nm":     [0.65,  0.07],
        "spec_F8_680nm":     [0.45,  0.08],
        "spec_Clear":        [0.38,  0.07],
        "spec_NIR":          [0.50,  0.09],
        "acustico_freq_Hz":  [1520,  13.0],
    },
    # =========================================================
    # BEBIDAS ALCOÓLICAS — CERVEJAS (marcas específicas)
    # =========================================================
    # ---------------------------------------------------------
    # HEINEKEN (Premium Lager) — SRM 2-4, ABV 5.0%
    # Dourada clara, carbonatação alta, lúpulo Saaz
    # ---------------------------------------------------------
    "alcoolica_cerveja_heineken": {
        "temperatura_C":     [4.0,   1.5],
        "condutividade_uS":  [1750,  180.0],
        "pH":                [4.25,  0.15],
        "spec_F1_415nm":     [0.32,  0.06],
        "spec_F2_445nm":     [0.42,  0.06],
        "spec_F3_480nm":     [0.56,  0.06],
        "spec_F4_515nm":     [0.72,  0.05],
        "spec_F5_555nm":     [0.80,  0.05],
        "spec_F6_590nm":     [0.74,  0.06],
        "spec_F7_630nm":     [0.52,  0.06],
        "spec_F8_680nm":     [0.36,  0.07],
        "spec_Clear":        [0.57,  0.06],
        "spec_NIR":          [0.46,  0.07],
        "acustico_freq_Hz":  [1458,  9.0],
    },
    # ---------------------------------------------------------
    # STELLA ARTOIS (Belgian Lager) — SRM 3-4, ABV 5.2%
    # Dourada mais intensa que Heineken, mais maltada
    # Maior corpo → condutividade levemente maior
    # ---------------------------------------------------------
    "alcoolica_cerveja_stella": {
        "temperatura_C":     [4.0,   1.5],
        "condutividade_uS":  [1850,  190.0],
        "pH":                [4.18,  0.15],
        "spec_F1_415nm":     [0.28,  0.06],
        "spec_F2_445nm":     [0.38,  0.06],
        "spec_F3_480nm":     [0.52,  0.07],
        "spec_F4_515nm":     [0.68,  0.06],
        "spec_F5_555nm":     [0.76,  0.05],
        "spec_F6_590nm":     [0.72,  0.06],
        "spec_F7_630nm":     [0.54,  0.07],
        "spec_F8_680nm":     [0.38,  0.07],
        "spec_Clear":        [0.53,  0.06],
        "spec_NIR":          [0.44,  0.07],
        "acustico_freq_Hz":  [1456,  9.0],
    },
    # ---------------------------------------------------------
    # BUDWEISER (American Lager) — SRM 1.5-3, ABV 5.0%
    # Mais pálida e leve que Heineken, arroz na receita
    # ---------------------------------------------------------
    "alcoolica_cerveja_budweiser": {
        "temperatura_C":     [4.0,   1.5],
        "condutividade_uS":  [1650,  170.0],
        "pH":                [4.10,  0.15],
        "spec_F1_415nm":     [0.35,  0.06],
        "spec_F2_445nm":     [0.45,  0.06],
        "spec_F3_480nm":     [0.60,  0.06],
        "spec_F4_515nm":     [0.74,  0.05],
        "spec_F5_555nm":     [0.83,  0.04],
        "spec_F6_590nm":     [0.76,  0.05],
        "spec_F7_630nm":     [0.56,  0.06],
        "spec_F8_680nm":     [0.40,  0.07],
        "spec_Clear":        [0.61,  0.05],
        "spec_NIR":          [0.49,  0.07],
        "acustico_freq_Hz":  [1462,  9.0],
    },
    # ---------------------------------------------------------
    # IPA (India Pale Ale) — SRM 6-14, ABV 6.5%
    # Âmbar/cobre, mais amarga, mais álcool, mais malte
    # ---------------------------------------------------------
    "alcoolica_cerveja_ipa": {
        "temperatura_C":     [7.0,   2.0],
        "condutividade_uS":  [2100,  250.0],
        "pH":                [4.0,   0.2],
        "spec_F1_415nm":     [0.15,  0.06],
        "spec_F2_445nm":     [0.22,  0.07],
        "spec_F3_480nm":     [0.35,  0.08],
        "spec_F4_515nm":     [0.50,  0.08],
        "spec_F5_555nm":     [0.60,  0.07],
        "spec_F6_590nm":     [0.68,  0.07],
        "spec_F7_630nm":     [0.60,  0.08],
        "spec_F8_680nm":     [0.45,  0.09],
        "spec_Clear":        [0.42,  0.08],
        "spec_NIR":          [0.40,  0.09],
        "acustico_freq_Hz":  [1448,  12.0],
    },
    # =========================================================
    # BEBIDAS ALCOÓLICAS — VINHOS
    # =========================================================
    "alcoolica_vinho_tinto": {
        "temperatura_C":     [16.0,  2.0],
        "condutividade_uS":  [2200,  300.0],
        "pH":                [3.5,   0.3],
        "spec_F1_415nm":     [0.05,  0.03],
        "spec_F2_445nm":     [0.08,  0.04],
        "spec_F3_480nm":     [0.10,  0.05],
        "spec_F4_515nm":     [0.12,  0.05],
        "spec_F5_555nm":     [0.15,  0.06],
        "spec_F6_590nm":     [0.20,  0.07],
        "spec_F7_630nm":     [0.35,  0.08],
        "spec_F8_680nm":     [0.28,  0.08],
        "spec_Clear":        [0.14,  0.05],
        "spec_NIR":          [0.35,  0.08],
        "acustico_freq_Hz":  [1430,  12.0],
    },
    # ---------------------------------------------------------
    # VINHO BRANCO — amarelo-palha, ABV ~12%, menos taninos
    # Muito mais transparente que tinto, perfil espectral diferente
    # ---------------------------------------------------------
    "alcoolica_vinho_branco": {
        "temperatura_C":     [10.0,  2.0],
        "condutividade_uS":  [1900,  250.0],
        "pH":                [3.2,   0.2],
        "spec_F1_415nm":     [0.50,  0.08],
        "spec_F2_445nm":     [0.58,  0.07],
        "spec_F3_480nm":     [0.65,  0.07],
        "spec_F4_515nm":     [0.72,  0.06],
        "spec_F5_555nm":     [0.76,  0.06],
        "spec_F6_590nm":     [0.73,  0.06],
        "spec_F7_630nm":     [0.65,  0.07],
        "spec_F8_680nm":     [0.55,  0.08],
        "spec_Clear":        [0.65,  0.07],
        "spec_NIR":          [0.42,  0.08],
        "acustico_freq_Hz":  [1435,  11.0],
    },
    # =========================================================
    # BEBIDAS ALCOÓLICAS — DESTILADOS
    # =========================================================
    "alcoolica_vodka": {
        "temperatura_C":     [3.0,   2.0],
        "condutividade_uS":  [50.0,  20.0],
        "pH":                [6.5,   0.5],
        "spec_F1_415nm":     [0.88,  0.04],
        "spec_F2_445nm":     [0.90,  0.04],
        "spec_F3_480nm":     [0.91,  0.04],
        "spec_F4_515nm":     [0.90,  0.04],
        "spec_F5_555nm":     [0.89,  0.04],
        "spec_F6_590nm":     [0.88,  0.04],
        "spec_F7_630nm":     [0.87,  0.05],
        "spec_F8_680nm":     [0.85,  0.05],
        "spec_Clear":        [0.89,  0.04],
        "spec_NIR":          [0.60,  0.08],
        "acustico_freq_Hz":  [1350,  15.0],
    },
    # ---------------------------------------------------------
    # VODKA ADULTERADA — metanol/álcool industrial + água
    # Condutividade sobe (impurezas), espectro altera no NIR,
    # pH mais ácido, acústico diferente (menos etanol puro)
    # ---------------------------------------------------------
    "alcoolica_vodka_adulterada": {
        "temperatura_C":     [3.0,   2.0],
        "condutividade_uS":  [180.0, 60.0],
        "pH":                [5.0,   0.8],
        "spec_F1_415nm":     [0.82,  0.06],
        "spec_F2_445nm":     [0.84,  0.06],
        "spec_F3_480nm":     [0.85,  0.06],
        "spec_F4_515nm":     [0.84,  0.06],
        "spec_F5_555nm":     [0.83,  0.06],
        "spec_F6_590nm":     [0.82,  0.06],
        "spec_F7_630nm":     [0.80,  0.07],
        "spec_F8_680nm":     [0.78,  0.07],
        "spec_Clear":        [0.83,  0.06],
        "spec_NIR":          [0.68,  0.08],
        "acustico_freq_Hz":  [1400,  20.0],
    },
    "alcoolica_cachaca": {
        "temperatura_C":     [22.0,  3.0],
        "condutividade_uS":  [80.0,  30.0],
        "pH":                [4.5,   0.5],
        "spec_F1_415nm":     [0.70,  0.08],
        "spec_F2_445nm":     [0.73,  0.07],
        "spec_F3_480nm":     [0.76,  0.07],
        "spec_F4_515nm":     [0.78,  0.07],
        "spec_F5_555nm":     [0.80,  0.06],
        "spec_F6_590nm":     [0.82,  0.06],
        "spec_F7_630nm":     [0.80,  0.06],
        "spec_F8_680nm":     [0.75,  0.07],
        "spec_Clear":        [0.77,  0.06],
        "spec_NIR":          [0.55,  0.09],
        "acustico_freq_Hz":  [1370,  14.0],
    },
    # ---------------------------------------------------------
    # CACHAÇA ADULTERADA — diluída com água e/ou metanol
    # Condutividade sobe, espectro altera, acústico sobe (mais água)
    # ---------------------------------------------------------
    "alcoolica_cachaca_adulterada": {
        "temperatura_C":     [22.0,  3.0],
        "condutividade_uS":  [220.0, 80.0],
        "pH":                [4.0,   0.7],
        "spec_F1_415nm":     [0.74,  0.08],
        "spec_F2_445nm":     [0.76,  0.08],
        "spec_F3_480nm":     [0.78,  0.08],
        "spec_F4_515nm":     [0.80,  0.07],
        "spec_F5_555nm":     [0.81,  0.07],
        "spec_F6_590nm":     [0.83,  0.06],
        "spec_F7_630nm":     [0.82,  0.07],
        "spec_F8_680nm":     [0.79,  0.07],
        "spec_Clear":        [0.80,  0.07],
        "spec_NIR":          [0.62,  0.09],
        "acustico_freq_Hz":  [1410,  18.0],
    },
    # ---------------------------------------------------------
    # WHISKY — âmbar dourado (envelhecido em barril), ABV ~40%
    # Cor do barril de carvalho, condutividade baixa
    # ---------------------------------------------------------
    "alcoolica_whisky": {
        "temperatura_C":     [20.0,  3.0],
        "condutividade_uS":  [65.0,  25.0],
        "pH":                [4.0,   0.4],
        "spec_F1_415nm":     [0.35,  0.08],
        "spec_F2_445nm":     [0.42,  0.08],
        "spec_F3_480nm":     [0.50,  0.08],
        "spec_F4_515nm":     [0.60,  0.07],
        "spec_F5_555nm":     [0.68,  0.07],
        "spec_F6_590nm":     [0.72,  0.06],
        "spec_F7_630nm":     [0.70,  0.07],
        "spec_F8_680nm":     [0.62,  0.07],
        "spec_Clear":        [0.55,  0.07],
        "spec_NIR":          [0.50,  0.08],
        "acustico_freq_Hz":  [1360,  13.0],
    },
}

# Mapeamento: subtipo → tipo principal
MAPA_TIPO = {
    "agua_torneira":                  "agua",
    "agua_mineral":                   "agua",
    "agua_mineral_gas":               "agua",
    "agua_contaminada":               "agua",
    "gasolina_pura":                  "gasolina",
    "gasolina_adulterada_10":         "gasolina",
    "gasolina_adulterada_20":         "gasolina",
    "gasolina_adulterada_30":         "gasolina",
    "suco_laranja":                   "suco",
    "suco_uva":                       "suco",
    "suco_limao":                     "suco",
    "suco_maca":                      "suco",
    "suco_manga":                     "suco",
    "alcoolica_cerveja_heineken":     "alcoolica",
    "alcoolica_cerveja_stella":       "alcoolica",
    "alcoolica_cerveja_budweiser":    "alcoolica",
    "alcoolica_cerveja_ipa":          "alcoolica",
    "alcoolica_vinho_tinto":          "alcoolica",
    "alcoolica_vinho_branco":         "alcoolica",
    "alcoolica_vodka":                "alcoolica",
    "alcoolica_vodka_adulterada":     "alcoolica",
    "alcoolica_cachaca":              "alcoolica",
    "alcoolica_cachaca_adulterada":   "alcoolica",
    "alcoolica_whisky":               "alcoolica",
}

# Mapeamento: gasolina → grau de adulteração
MAPA_ADULTERACAO = {
    "gasolina_pura":             0,
    "gasolina_adulterada_10":    10,
    "gasolina_adulterada_20":    20,
    "gasolina_adulterada_30":    30,
}

# Mapeamento: potabilidade da água (1=potável, 0=não)
MAPA_POTABILIDADE = {
    "agua_torneira":     1,
    "agua_mineral":      1,
    "agua_mineral_gas":  1,
    "agua_contaminada":  0,
}

# Mapeamento: adulteração de bebida alcoólica (1=genuína, 0=adulterada)
MAPA_GENUINIDADE_ALCOOL = {
    "alcoolica_cerveja_heineken":   1,
    "alcoolica_cerveja_stella":     1,
    "alcoolica_cerveja_budweiser":  1,
    "alcoolica_cerveja_ipa":        1,
    "alcoolica_vinho_tinto":        1,
    "alcoolica_vinho_branco":       1,
    "alcoolica_vodka":              1,
    "alcoolica_vodka_adulterada":   0,
    "alcoolica_cachaca":            1,
    "alcoolica_cachaca_adulterada": 0,
    "alcoolica_whisky":             1,
}

# Mapeamento: marca de cerveja
MAPA_MARCA_CERVEJA = {
    "alcoolica_cerveja_heineken":   "heineken",
    "alcoolica_cerveja_stella":     "stella_artois",
    "alcoolica_cerveja_budweiser":  "budweiser",
    "alcoolica_cerveja_ipa":        "ipa",
}


def gerar_amostras_liquido(subtipo: str, n_amostras: int,
                           rng: np.random.Generator) -> pd.DataFrame:
    """Gera N amostras sintéticas para um subtipo de líquido."""
    perfil = PERFIS_SENSORES[subtipo]
    dados = {}

    for sensor_name in SENSOR_NAMES:
        media, desvio = perfil[sensor_name]
        valores = rng.normal(loc=media, scale=desvio, size=n_amostras)

        if sensor_name in ["condutividade_uS", "acustico_freq_Hz"]:
            valores = np.clip(valores, 0, None)
        elif sensor_name == "pH":
            valores = np.clip(valores, 0, 14)
        elif sensor_name.startswith("spec_"):
            valores = np.clip(valores, 0, 1)

        dados[sensor_name] = valores

    dados["subtipo"] = subtipo
    dados["tipo"] = MAPA_TIPO[subtipo]
    dados["adulteracao_pct"] = MAPA_ADULTERACAO.get(subtipo, -1)
    dados["potabilidade"] = MAPA_POTABILIDADE.get(subtipo, -1)
    dados["genuina"] = MAPA_GENUINIDADE_ALCOOL.get(subtipo, -1)
    dados["marca_cerveja"] = MAPA_MARCA_CERVEJA.get(subtipo, "")
    dados["timestamp"] = [datetime.now().isoformat()] * n_amostras
    dados["sensor_hw"] = ["AS7341+DS18B20+PCB+Piezo"] * n_amostras

    return pd.DataFrame(dados)


def gerar_dataset_completo(n_amostras_por_classe: int = N_AMOSTRAS_POR_CLASSE,
                           seed: int = RANDOM_SEED,
                           salvar: bool = True) -> pd.DataFrame:
    """Gera o dataset completo com todos os tipos de líquidos expandidos."""
    rng = np.random.default_rng(seed)
    frames = []

    print("=" * 60)
    print(" GERANDO DADOS SINTÉTICOS DE SENSORES (EXPANDIDO)")
    print("=" * 60)

    for subtipo in PERFIS_SENSORES:
        df_sub = gerar_amostras_liquido(subtipo, n_amostras_por_classe, rng)
        frames.append(df_sub)
        print(f"  + {subtipo:<30s} -> {n_amostras_por_classe} amostras")

    df_completo = pd.concat(frames, ignore_index=True)
    df_completo = df_completo.sample(frac=1, random_state=seed).reset_index(drop=True)

    total = len(df_completo)
    n_features = len(SENSOR_NAMES)
    print(f"\n  Total: {total} amostras | {n_features} features")
    print(f"  Classes (tipo):    {df_completo['tipo'].value_counts().to_dict()}")
    print(f"  Classes (subtipo): {df_completo['subtipo'].value_counts().to_dict()}")

    if salvar:
        caminho_csv = os.path.join(DATA_SYNTHETIC, "dataset_sensores_expandido.csv")
        df_completo.to_csv(caminho_csv, index=False)
        print(f"\n  Salvo em: {caminho_csv}")

        # Salvar metadados
        metadata = {
            "n_amostras_por_classe": n_amostras_por_classe,
            "seed": seed,
            "n_total": total,
            "n_features": n_features,
            "tipos": list(df_completo["tipo"].unique()),
            "subtipos": list(df_completo["subtipo"].unique()),
            "sensor_names": SENSOR_NAMES,
            "gerado_em": datetime.now().isoformat(),
        }
        meta_path = os.path.join(DATA_METADATA, "dataset_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"  Metadados: {meta_path}")

    print("=" * 60)
    return df_completo


if __name__ == "__main__":
    df = gerar_dataset_completo()
    print("\nPrimeiras 5 linhas:")
    print(df.head())
    print(f"\nEstatísticas descritivas:")
    print(df.describe().round(2))

"""
============================================================
 CONFIGURAÇÕES GLOBAIS DO PROJETO
 Lab7 - Identificação de Líquidos com LiquidsML no ESP32
============================================================
 Classes expandidas: água, gasolina, sucos, bebidas alcoólicas
============================================================
"""

import os

# ------------------------------------------------------------
# Caminhos do Projeto
# ------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATA_RAW = os.path.join(DATA_DIR, "raw")
DATA_PROCESSED = os.path.join(DATA_DIR, "processed")
DATA_SYNTHETIC = os.path.join(DATA_DIR, "synthetic")
DATA_METADATA = os.path.join(DATA_DIR, "metadata")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
TFLITE_DIR = os.path.join(MODELS_DIR, "tflite")
CHECKPOINTS_DIR = os.path.join(MODELS_DIR, "checkpoints")

# Criar diretórios se não existirem
for d in [DATA_RAW, DATA_PROCESSED, DATA_SYNTHETIC, DATA_METADATA,
          RESULTS_DIR, FIGURES_DIR, MODELS_DIR, TFLITE_DIR, CHECKPOINTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ------------------------------------------------------------
# Parâmetros de Geração de Dados Sintéticos
# ------------------------------------------------------------
N_AMOSTRAS_POR_CLASSE = 500
RANDOM_SEED = 42
NOISE_LEVEL = 0.05

# ------------------------------------------------------------
# Definição dos Líquidos e Suas Classes
# ------------------------------------------------------------
# Nível 1: Tipo principal do líquido
TIPOS_LIQUIDO = ["agua", "gasolina", "suco", "alcoolica"]

# Nível 2: Subtipos específicos por tipo
SUBTIPOS = {
    "agua": ["torneira", "mineral", "mineral_gas", "contaminada"],
    "gasolina": ["pura", "adulterada_10", "adulterada_20", "adulterada_30"],
    "suco": ["laranja", "uva", "limao", "maca", "manga"],
    "alcoolica": [
        "cerveja_heineken", "cerveja_stella", "cerveja_budweiser", "cerveja_ipa",
        "vinho_tinto", "vinho_branco",
        "vodka", "vodka_adulterada",
        "cachaca", "cachaca_adulterada",
        "whisky",
    ],
}

# Nível 3: Qualidade / segurança por tipo
QUALIDADE = {
    "agua": {
        "potavel": ["torneira", "mineral", "mineral_gas"],
        "nao_potavel": ["contaminada"],
    },
    "gasolina": {
        "conforme": ["pura"],
        "adulterada": ["adulterada_10", "adulterada_20", "adulterada_30"],
    },
    "alcoolica": {
        "genuina": [
            "cerveja_heineken", "cerveja_stella", "cerveja_budweiser", "cerveja_ipa",
            "vinho_tinto", "vinho_branco", "vodka", "cachaca", "whisky",
        ],
        "adulterada": ["vodka_adulterada", "cachaca_adulterada"],
    },
}

# ------------------------------------------------------------
# Nomes dos Sensores (Features do vetor x~)
# ------------------------------------------------------------
SENSOR_NAMES = [
    "temperatura_C",
    "condutividade_uS",
    "pH",
    "spec_F1_415nm",    # Violeta
    "spec_F2_445nm",    # Azul escuro
    "spec_F3_480nm",    # Azul
    "spec_F4_515nm",    # Verde
    "spec_F5_555nm",    # Verde-amarelo
    "spec_F6_590nm",    # Laranja
    "spec_F7_630nm",    # Vermelho
    "spec_F8_680nm",    # Vermelho escuro
    "spec_Clear",       # Luz visível total
    "spec_NIR",         # Infravermelho próximo
    "acustico_freq_Hz",
]

N_FEATURES = len(SENSOR_NAMES)

# ------------------------------------------------------------
# Parâmetros de Divisão dos Dados
# ------------------------------------------------------------
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ------------------------------------------------------------
# Parâmetros dos Modelos de Machine Learning
# ------------------------------------------------------------
ML_PARAMS = {
    "random_forest": {
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_split": 5,
        "random_state": RANDOM_SEED,
    },
    "xgboost": {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": RANDOM_SEED,
        "eval_metric": "mlogloss",
    },
    "neural_network": {
        "hidden_layers": [128, 64, 32],
        "epochs": 50,
        "batch_size": 64,
        "learning_rate": 0.001,
    },
    "cnn": {
        "filters": [32, 64],
        "kernel_size": 3,
        "epochs": 50,
        "batch_size": 64,
        "learning_rate": 0.001,
    },
    "svm": {
        "kernel": "rbf",
        "C": 10.0,
        "gamma": "scale",
        "random_state": RANDOM_SEED,
    },
    "knn": {
        "n_neighbors": 7,
        "weights": "distance",
        "metric": "minkowski",
    },
}

# ------------------------------------------------------------
# Parâmetros de Data Augmentation
# ------------------------------------------------------------
AUGMENTATION = {
    "temperature_shift_range": (-5.0, 5.0),     # °C
    "noise_sigma_factor": (0.5, 2.0),           # Multiplicador do ruído
    "contamination_ratio": (0.0, 0.05),         # Até 5% de contaminação cruzada
    "n_augmented_per_sample": 3,                # Amostras augmentadas por original
}

# ------------------------------------------------------------
# Parâmetros de Quantização TFLite
# ------------------------------------------------------------
TFLITE_PARAMS = {
    "quantization": "int8",
    "representative_dataset_size": 200,
}

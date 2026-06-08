"""
============================================================
 CONVERSÃO PARA TFLITE COM QUANTIZAÇÃO INT8
 Exporta modelo Keras para TFLite + cabeçalho C (.h)
============================================================
"""

import numpy as np
import os
import sys
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import tensorflow as tf
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import (
    TFLITE_DIR,
    CHECKPOINTS_DIR,
    DATA_PROCESSED,
    TFLITE_PARAMS,
    RANDOM_SEED,
)


def converter_para_tflite(modelo_keras, X_calibration,
                          nome_modelo: str = "liquid_classifier",
                          quantizar: bool = True):
    """
    Converte modelo Keras para TFLite com quantização int8.

    Args:
        modelo_keras: Modelo Keras treinado
        X_calibration: Dados de calibração para quantização representativa
        nome_modelo: Nome base para os arquivos de saída
        quantizar: Se True, aplica quantização int8

    Returns:
        Caminho do arquivo .tflite gerado
    """
    print("\n" + "=" * 60)
    print(" CONVERSAO PARA TENSORFLOW LITE")
    print("=" * 60)

    # Salvar modelo Keras completo
    keras_path = os.path.join(CHECKPOINTS_DIR, f"{nome_modelo}.keras")
    modelo_keras.save(keras_path)
    print(f"  Modelo Keras salvo: {keras_path}")

    # Converter para TFLite via tf.Module (compatível com Keras 3)
    n_features = X_calibration.shape[1]

    class _TFLiteModule(tf.Module):
        def __init__(self, keras_model, n_feat):
            super().__init__()
            self.model = keras_model
            self._serve = tf.function(
                lambda x: self.model(x, training=False),
                input_signature=[tf.TensorSpec(shape=[1, n_feat], dtype=tf.float32)],
            )

    module = _TFLiteModule(modelo_keras, n_features)
    concrete_fn = module._serve.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn], module)

    if quantizar:
        print("  Aplicando quantizacao int8 (post-training)...")

        def representative_dataset():
            n_samples = min(len(X_calibration),
                            TFLITE_PARAMS["representative_dataset_size"])
            indices = np.random.RandomState(RANDOM_SEED).choice(
                len(X_calibration), n_samples, replace=False
            )
            for i in indices:
                sample = X_calibration[i:i+1].astype(np.float32)
                yield [sample]

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    # Salvar .tflite
    tflite_path = os.path.join(TFLITE_DIR, f"{nome_modelo}.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"  Modelo TFLite salvo: {tflite_path}")
    print(f"  Tamanho: {size_kb:.1f} KB")

    # Gerar cabeçalho C (.h)
    header_path = gerar_cabecalho_c(tflite_model, nome_modelo)
    print(f"  Cabecalho C gerado: {header_path}")

    # Validar modelo TFLite
    validar_tflite(tflite_path, X_calibration[:5])

    print("=" * 60)
    return tflite_path


def gerar_cabecalho_c(tflite_model_bytes: bytes,
                      nome_modelo: str = "liquid_classifier") -> str:
    """Gera arquivo .h com o modelo como array de bytes para C/C++."""
    var_name = nome_modelo.replace("-", "_").replace(" ", "_")

    lines = []
    lines.append(f"// Auto-generated TFLite model header")
    lines.append(f"// Model: {nome_modelo}")
    lines.append(f"// Size: {len(tflite_model_bytes)} bytes")
    lines.append(f"#ifndef {var_name.upper()}_H")
    lines.append(f"#define {var_name.upper()}_H")
    lines.append("")
    lines.append(f"const unsigned int {var_name}_len = {len(tflite_model_bytes)};")
    lines.append(f"alignas(8) const unsigned char {var_name}[] = {{")

    # Converter para hex
    hex_values = []
    for i, byte in enumerate(tflite_model_bytes):
        hex_values.append(f"0x{byte:02x}")
        if (i + 1) % 12 == 0:
            lines.append("  " + ", ".join(hex_values) + ",")
            hex_values = []

    if hex_values:
        lines.append("  " + ", ".join(hex_values))

    lines.append("};")
    lines.append("")
    lines.append(f"#endif // {var_name.upper()}_H")

    header_path = os.path.join(TFLITE_DIR, f"{nome_modelo}.h")
    with open(header_path, "w") as f:
        f.write("\n".join(lines))

    return header_path


def validar_tflite(tflite_path: str, X_sample: np.ndarray):
    """Valida o modelo TFLite convertido com amostras de teste."""
    print("\n  Validando modelo TFLite...")

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"    Input shape:  {input_details[0]['shape']}")
    print(f"    Input dtype:  {input_details[0]['dtype']}")
    print(f"    Output shape: {output_details[0]['shape']}")
    print(f"    Output dtype: {output_details[0]['dtype']}")

    # Testar inferência
    input_scale = input_details[0].get("quantization_parameters", {}).get("scales", [1.0])
    input_zero = input_details[0].get("quantization_parameters", {}).get("zero_points", [0])

    for i in range(min(3, len(X_sample))):
        sample = X_sample[i:i+1].astype(np.float32)

        if input_details[0]["dtype"] == np.int8:
            if len(input_scale) > 0 and input_scale[0] != 0:
                sample = (sample / input_scale[0] + input_zero[0]).astype(np.int8)
            else:
                sample = sample.astype(np.int8)

        interpreter.set_tensor(input_details[0]["index"], sample)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])
        pred_class = np.argmax(output)
        print(f"    Amostra {i}: classe predita = {pred_class}")

    print("    Validacao TFLite OK!")


def exportar_parametros_normalizacao(scaler_path: str = None):
    """Exporta parâmetros de normalização como cabeçalho C para o ESP32."""
    if scaler_path is None:
        scaler_path = os.path.join(DATA_PROCESSED, "normalization_params.joblib")

    norm_params = joblib.load(scaler_path)
    mean = norm_params["mean"]
    scale = norm_params["scale"]
    feature_names = norm_params["feature_names"]

    lines = []
    lines.append("// Auto-generated normalization parameters")
    lines.append("#ifndef NORMALIZATION_PARAMS_H")
    lines.append("#define NORMALIZATION_PARAMS_H")
    lines.append("")
    lines.append(f"#define N_FEATURES {len(mean)}")
    lines.append("")

    # Feature names as comments
    lines.append("// Feature order:")
    for i, name in enumerate(feature_names):
        lines.append(f"//   [{i}] {name}")
    lines.append("")

    # Mean array
    lines.append("const float feature_mean[] = {")
    for i in range(0, len(mean), 6):
        chunk = mean[i:i+6]
        lines.append("  " + ", ".join(f"{v:.6f}f" for v in chunk) + ",")
    lines.append("};")
    lines.append("")

    # Scale array
    lines.append("const float feature_scale[] = {")
    for i in range(0, len(scale), 6):
        chunk = scale[i:i+6]
        lines.append("  " + ", ".join(f"{v:.6f}f" for v in chunk) + ",")
    lines.append("};")
    lines.append("")
    lines.append("#endif // NORMALIZATION_PARAMS_H")

    header_path = os.path.join(TFLITE_DIR, "normalization_params.h")
    with open(header_path, "w") as f:
        f.write("\n".join(lines))

    print(f"  Parametros de normalizacao exportados: {header_path}")
    return header_path

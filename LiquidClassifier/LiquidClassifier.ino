/*
 * ============================================================
 *  LiquidClassifier — Identificação de Líquidos com ESP32
 *  TensorFlow Lite Micro + Sensores Multimodais + OLED
 * ============================================================
 *
 *  Classificação em tempo real de 4 tipos de líquidos:
 *    - Água, Gasolina, Suco, Bebida Alcoólica
 *
 *  Hardware necessário:
 *    - ESP32 DevKit V1
 *    - AS7341  — sensor espectral 11 canais (I2C)
 *    - DS18B20 — sensor de temperatura (OneWire, pino 4)
 *    - Eletrodos PCB — condutividade elétrica (ADC, pino 34)
 *    - Módulo pH (E-201-C) — (ADC, pino 35)
 *    - Disco piezoelétrico — frequência acústica (ADC, pino 32)
 *    - LED RGB (pinos 25, 26, 27)
 *    - (OPCIONAL) Display OLED SSD1306 128x64 (I2C)
 *
 *  Saída dos resultados:
 *    - Serial Monitor (115200 baud) — saída principal, sempre ativa
 *    - Display OLED — opcional, auto-detectado na inicialização
 *    - LED RGB — feedback visual por cor
 *
 *  Bibliotecas necessárias (Arduino Library Manager):
 *    - Adafruit AS7341
 *    - OneWire
 *    - DallasTemperature
 *    - TensorFlowLite_ESP32 (por Masayuki Tanaka)
 *    - (OPCIONAL) Adafruit SSD1306 + Adafruit GFX Library
 *
 *  Como usar:
 *    1. Instale as bibliotecas acima
 *    2. Selecione Board: "ESP32 Dev Module"
 *    3. Compile e faça upload
 *    4. Abra o Serial Monitor (115200 baud)
 *    5. Mergulhe os sensores no líquido
 *    6. Pressione BOOT ou envie qualquer texto pelo Serial Monitor
 *    7. O resultado aparece no Serial Monitor (e no OLED, se conectado)
 *
 * ============================================================
 */

#include <Wire.h>
#include <Adafruit_AS7341.h>
#include <OneWire.h>
#include <DallasTemperature.h>
// OLED é opcional — comente a linha abaixo se quiser usar display OLED
// Para usar OLED: comente #define NO_OLED e instale Adafruit SSD1306 + GFX
#define NO_OLED
#ifndef NO_OLED
  #include <Adafruit_GFX.h>
  #include <Adafruit_SSD1306.h>
#endif

// TensorFlow Lite Micro
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Modelo treinado e parâmetros de normalização
#include "liquid_classifier.h"
#include "normalization_params.h"

// ============================================================
//  CONFIGURAÇÃO DE PINOS
// ============================================================
#define PIN_DS18B20     4     // Temperatura (OneWire)
#define PIN_COND_ADC    34    // Condutividade (ADC)
#define PIN_PH_ADC      35    // pH (ADC)
#define PIN_PIEZO_ADC   32    // Piezoelétrico (ADC)
#define PIN_LED_R       25    // LED RGB — Vermelho
#define PIN_LED_G       26    // LED RGB — Verde
#define PIN_LED_B       27    // LED RGB — Azul
#define PIN_SDA         21    // I2C SDA
#define PIN_SCL         22    // I2C SCL
#define PIN_BOTAO       0     // Botão BOOT do ESP32

// ============================================================
//  DISPLAY OLED
// ============================================================
#define SCREEN_WIDTH    128
#define SCREEN_HEIGHT   64
#define OLED_RESET      -1
#define OLED_ADDR       0x3C

#ifndef NO_OLED
  Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);
#endif
bool oled_ok = false;

// ============================================================
//  SENSORES
// ============================================================
Adafruit_AS7341 as7341;
bool as7341_ok = false;

OneWire oneWire(PIN_DS18B20);
DallasTemperature ds18b20(&oneWire);
bool ds18b20_ok = false;

// ============================================================
//  CALIBRAÇÃO DOS SENSORES ANALÓGICOS
//  Ajuste estes valores após calibrar com soluções conhecidas
// ============================================================
// pH: mapeia tensão ADC para escala 0-14
// Calibre com soluções tampão pH 4.0 e pH 7.0
const float PH_OFFSET   = 0.0;    // Offset de calibração
const float PH_SLOPE    = 14.0 / 4095.0;  // Inclinação

// Condutividade: mapeia ADC para µS/cm
// Calibre com solução de KCl 1413 µS/cm
const float COND_FACTOR = 5000.0 / 4095.0;

// Piezoelétrico: mapeia ADC para frequência Hz
const float PIEZO_BASE_FREQ = 1200.0;
const float PIEZO_RANGE     = 600.0;

// ============================================================
//  TFLITE MICRO
// ============================================================
constexpr int kTensorArenaSize = 16 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter_tfl = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

// ============================================================
//  CLASSES DE SAÍDA
// ============================================================
const char* CLASS_NAMES[] = {"Agua", "Alcoolica", "Gasolina", "Suco"};
const int N_CLASSES = 4;

// Cores do LED para cada classe (R, G, B)
const uint8_t CLASS_COLORS[][3] = {
    {0,   0,   255},  // Água      → Azul
    {255, 0,   0},    // Alcoólica → Vermelho
    {255, 255, 0},    // Gasolina  → Amarelo
    {0,   255, 0},    // Suco      → Verde
};

// ============================================================
//  BUFFERS
// ============================================================
float raw_features[N_FEATURES];
float norm_features[N_FEATURES];

// Estado
bool modo_continuo = false;
unsigned long ultimo_scan = 0;
const unsigned long INTERVALO_SCAN = 5000;

// ============================================================
//  FUNÇÕES DE LEITURA DOS SENSORES
// ============================================================

float ler_temperatura() {
    if (!ds18b20_ok) {
        // Fallback: leitura analógica do pino (sensor resistivo)
        int raw = analogRead(PIN_DS18B20);
        return (raw / 4095.0) * 100.0 - 10.0;
    }
    ds18b20.requestTemperatures();
    float temp = ds18b20.getTempCByIndex(0);
    if (temp == DEVICE_DISCONNECTED_C) {
        Serial.println("  [WARN] DS18B20 desconectado");
        return 25.0;  // Valor padrão
    }
    return temp;
}

float ler_condutividade() {
    // Média de 10 leituras para reduzir ruído
    long soma = 0;
    for (int i = 0; i < 10; i++) {
        soma += analogRead(PIN_COND_ADC);
        delay(2);
    }
    return (soma / 10.0) * COND_FACTOR;
}

float ler_ph() {
    // Média de 10 leituras
    long soma = 0;
    for (int i = 0; i < 10; i++) {
        soma += analogRead(PIN_PH_ADC);
        delay(2);
    }
    return (soma / 10.0) * PH_SLOPE + PH_OFFSET;
}

float ler_acustico() {
    // Medir frequência do piezoelétrico via contagem de pulsos
    long soma = 0;
    for (int i = 0; i < 10; i++) {
        soma += analogRead(PIN_PIEZO_ADC);
        delay(2);
    }
    return PIEZO_BASE_FREQ + (soma / 10.0 / 4095.0) * PIEZO_RANGE;
}

bool ler_as7341(float* canais) {
    if (!as7341_ok) {
        // Valores padrão se sensor não disponível
        for (int i = 0; i < 10; i++) canais[i] = 0.5;
        Serial.println("  [WARN] AS7341 nao disponivel, usando valores padrao");
        return false;
    }

    if (!as7341.readAllChannels()) {
        Serial.println("  [WARN] Falha na leitura AS7341");
        for (int i = 0; i < 10; i++) canais[i] = 0.5;
        return false;
    }

    // Ler contagens brutas e normalizar para 0-1
    uint16_t counts[12];
    counts[0]  = as7341.getChannel(AS7341_CHANNEL_415nm_F1);
    counts[1]  = as7341.getChannel(AS7341_CHANNEL_445nm_F2);
    counts[2]  = as7341.getChannel(AS7341_CHANNEL_480nm_F3);
    counts[3]  = as7341.getChannel(AS7341_CHANNEL_515nm_F4);
    counts[4]  = as7341.getChannel(AS7341_CHANNEL_555nm_F5);
    counts[5]  = as7341.getChannel(AS7341_CHANNEL_590nm_F6);
    counts[6]  = as7341.getChannel(AS7341_CHANNEL_630nm_F7);
    counts[7]  = as7341.getChannel(AS7341_CHANNEL_680nm_F8);
    counts[8]  = as7341.getChannel(AS7341_CHANNEL_CLEAR);
    counts[9]  = as7341.getChannel(AS7341_CHANNEL_NIR);

    // Encontrar valor máximo para normalização
    uint16_t max_val = 1;
    for (int i = 0; i < 10; i++) {
        if (counts[i] > max_val) max_val = counts[i];
    }

    // Normalizar para intervalo 0.0 - 1.0
    for (int i = 0; i < 10; i++) {
        canais[i] = (float)counts[i] / (float)max_val;
    }

    return true;
}

// ============================================================
//  COLETA E PROCESSAMENTO DE FEATURES
// ============================================================

void coletar_features() {
    Serial.println("  Coletando dados dos sensores...");

    // [0] Temperatura
    raw_features[0] = ler_temperatura();
    Serial.printf("    Temp: %.1f C\n", raw_features[0]);

    // [1] Condutividade
    raw_features[1] = ler_condutividade();
    Serial.printf("    Cond: %.0f uS/cm\n", raw_features[1]);

    // [2] pH
    raw_features[2] = ler_ph();
    Serial.printf("    pH:   %.2f\n", raw_features[2]);

    // [3..12] Espectro AS7341 (F1-F8, Clear, NIR)
    float spec[10];
    ler_as7341(spec);
    for (int i = 0; i < 10; i++) {
        raw_features[3 + i] = spec[i];
    }

    // [13] Frequência acústica
    raw_features[13] = ler_acustico();
    Serial.printf("    Freq: %.0f Hz\n", raw_features[13]);

    // === FEATURES DERIVADAS (calculadas on-device) ===
    const float eps = 1e-6f;
    float temp      = raw_features[0];
    float cond      = raw_features[1];
    float spec_F1   = raw_features[3];   // 415nm
    float spec_F3   = raw_features[5];   // 480nm
    float spec_F4   = raw_features[6];   // 515nm
    float spec_F6   = raw_features[8];   // 590nm
    float spec_F7   = raw_features[9];   // 630nm
    float spec_Clr  = raw_features[11];  // Clear
    float spec_NIR  = raw_features[12];  // NIR

    // [14] Condutividade compensada para 25°C
    raw_features[14] = cond / (1.0f + 0.02f * (temp - 25.0f));

    // [15] Ratio azul/vermelho (F3/F7)
    raw_features[15] = spec_F3 / (spec_F7 + eps);

    // [16] Ratio verde/vermelho (F4/F7)
    raw_features[16] = spec_F4 / (spec_F7 + eps);

    // [17] Ratio NIR/Clear
    raw_features[17] = spec_NIR / (spec_Clr + eps);

    // [18] Ratio violeta/laranja (F1/F6)
    raw_features[18] = spec_F1 / (spec_F6 + eps);

    // [19-21] Estatísticas espectrais
    float soma = 0, minv = 1e9, maxv = -1e9;
    for (int i = 3; i <= 12; i++) {
        soma += raw_features[i];
        if (raw_features[i] < minv) minv = raw_features[i];
        if (raw_features[i] > maxv) maxv = raw_features[i];
    }
    float media = soma / 10.0f;
    raw_features[19] = media;          // spectral_mean

    float var_soma = 0;
    for (int i = 3; i <= 12; i++) {
        float d = raw_features[i] - media;
        var_soma += d * d;
    }
    raw_features[20] = sqrtf(var_soma / 10.0f);  // spectral_std
    raw_features[21] = maxv - minv;                // spectral_range
}

// ============================================================
//  NORMALIZAÇÃO (StandardScaler idêntico ao Python)
// ============================================================

void normalizar_features() {
    for (int i = 0; i < N_FEATURES; i++) {
        norm_features[i] = (raw_features[i] - feature_mean[i]) / feature_scale[i];
    }
}

// ============================================================
//  INFERÊNCIA TFLITE
// ============================================================

int executar_inferencia(float* confianca) {
    // Quantizar float → int8 e copiar para tensor de entrada
    TfLiteQuantizationParams in_params = input_tensor->params;
    for (int i = 0; i < N_FEATURES; i++) {
        float val = norm_features[i];
        int32_t q = (int32_t)(val / in_params.scale + in_params.zero_point);
        if (q < -128) q = -128;
        if (q > 127) q = 127;
        input_tensor->data.int8[i] = (int8_t)q;
    }

    // Executar
    TfLiteStatus status = interpreter_tfl->Invoke();
    if (status != kTfLiteOk) {
        Serial.println("  [ERRO] Falha na inferencia TFLite!");
        return -1;
    }

    // Desquantizar saída e encontrar classe com maior score
    TfLiteQuantizationParams out_params = output_tensor->params;
    int melhor_classe = 0;
    float melhor_score = -1e9;

    for (int i = 0; i < N_CLASSES; i++) {
        float score = (output_tensor->data.int8[i] - out_params.zero_point)
                      * out_params.scale;
        if (score > melhor_score) {
            melhor_score = score;
            melhor_classe = i;
        }
    }

    *confianca = melhor_score;
    return melhor_classe;
}

// ============================================================
//  LED RGB
// ============================================================

void set_led(uint8_t r, uint8_t g, uint8_t b) {
    analogWrite(PIN_LED_R, r);
    analogWrite(PIN_LED_G, g);
    analogWrite(PIN_LED_B, b);
}

void led_resultado(int classe) {
    if (classe >= 0 && classe < N_CLASSES) {
        set_led(CLASS_COLORS[classe][0],
                CLASS_COLORS[classe][1],
                CLASS_COLORS[classe][2]);
    } else {
        set_led(255, 255, 255);  // Branco = erro
    }
}

// ============================================================
//  DISPLAY OLED (opcional — auto-detectado)
// ============================================================

#ifndef NO_OLED
void oled_tela_inicial() {
    if (!oled_ok) return;
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 0);
    display.println("=== LiquidClassifier ===");
    display.println();
    display.println("Mergulhe os sensores");
    display.println("no liquido e pressione");
    display.println("BOOT ou envie texto");
    display.println("pelo Serial Monitor.");
    display.display();
}

void oled_classificando() {
    if (!oled_ok) return;
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 0);
    display.println("  CLASSIFICANDO...");
    display.println();
    display.println("  Lendo sensores...");
    display.display();
}

void oled_resultado(int classe, float confianca) {
    if (!oled_ok) return;
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 0);
    display.println("   RESULTADO");
    display.drawLine(0, 10, 127, 10, SSD1306_WHITE);
    display.setTextSize(2);
    display.setCursor(0, 16);
    if (classe >= 0 && classe < N_CLASSES) {
        display.println(CLASS_NAMES[classe]);
    } else {
        display.println("ERRO");
    }
    display.setTextSize(1);
    display.setCursor(0, 38);
    display.print("Confianca: ");
    display.print(confianca * 100, 1);
    display.println("%");
    display.setCursor(0, 50);
    display.printf("T:%.0fC pH:%.1f C:%.0f",
                   raw_features[0], raw_features[2], raw_features[1]);
    display.display();
}
#else
// Stubs quando OLED desativado
void oled_tela_inicial() {}
void oled_classificando() {}
void oled_resultado(int, float) {}
#endif

// ============================================================
//  CLASSIFICAÇÃO COMPLETA
// ============================================================

void classificar() {
    Serial.println("--------------------------------------------");
    Serial.println("  NOVA CLASSIFICACAO");
    Serial.println("--------------------------------------------");

    oled_classificando();

    // 1. Ler sensores
    coletar_features();

    // 2. Normalizar (StandardScaler)
    normalizar_features();

    // 3. Inferência TFLite
    float confianca = 0;
    int classe = executar_inferencia(&confianca);

    // 4. Mostrar resultado no Serial Monitor (saída principal)
    Serial.println();
    if (classe >= 0 && classe < N_CLASSES) {
        Serial.println("  ╔══════════════════════════════════╗");
        Serial.print(  "  ║  LIQUIDO: ");
        // Pad nome da classe para alinhar
        char buf[32];
        snprintf(buf, sizeof(buf), "%-22s", CLASS_NAMES[classe]);
        Serial.print(buf);
        Serial.println("║");
        Serial.print(  "  ║  CONFIANCA: ");
        snprintf(buf, sizeof(buf), "%-20.1f%%", confianca * 100);
        Serial.print(buf);
        Serial.println("║");
        Serial.println("  ╠══════════════════════════════════╣");
        Serial.print(  "  ║  Temp: ");
        snprintf(buf, sizeof(buf), "%6.1f C", raw_features[0]);
        Serial.print(buf);
        Serial.print("   pH: ");
        snprintf(buf, sizeof(buf), "%5.2f", raw_features[2]);
        Serial.print(buf);
        Serial.println("    ║");
        Serial.print(  "  ║  Cond: ");
        snprintf(buf, sizeof(buf), "%6.0f uS/cm", raw_features[1]);
        Serial.print(buf);
        Serial.print("  Freq: ");
        snprintf(buf, sizeof(buf), "%4.0f Hz", raw_features[13]);
        Serial.print(buf);
        Serial.println(" ║");
        Serial.println("  ╚══════════════════════════════════╝");

        led_resultado(classe);
        oled_resultado(classe, confianca);
    } else {
        Serial.println("  [ERRO] Falha na classificacao!");
        set_led(255, 0, 0);
    }

    Serial.println();
}

// ============================================================
//  SETUP
// ============================================================

void setup() {
    Serial.begin(115200);
    while (!Serial) delay(10);

    Serial.println();
    Serial.println("============================================");
    Serial.println(" LiquidClassifier — ESP32");
    Serial.println(" Identificacao de Liquidos por ML");
    Serial.println("============================================");

    // Pinos
    pinMode(PIN_LED_R, OUTPUT);
    pinMode(PIN_LED_G, OUTPUT);
    pinMode(PIN_LED_B, OUTPUT);
    pinMode(PIN_BOTAO, INPUT_PULLUP);
    set_led(50, 50, 50);  // LED branco = inicializando

    // I2C
    Wire.begin(PIN_SDA, PIN_SCL);
    Wire.setClock(400000);

    // --- Display OLED (opcional) ---
#ifndef NO_OLED
    Serial.print("  Display OLED... ");
    if (display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
        oled_ok = true;
        Serial.println("OK (conectado)");
        display.clearDisplay();
        display.setTextSize(1);
        display.setTextColor(SSD1306_WHITE);
        display.setCursor(10, 28);
        display.println("Inicializando...");
        display.display();
    } else {
        Serial.println("nao conectado (usando apenas Serial Monitor)");
    }
#else
    Serial.println("  Display OLED... desativado (NO_OLED definido)");
#endif

    // --- Sensor espectral AS7341 ---
    Serial.print("  Sensor AS7341... ");
    if (as7341.begin()) {
        as7341_ok = true;
        as7341.setATIME(100);
        as7341.setASTEP(999);
        as7341.setGain(AS7341_GAIN_256X);
        Serial.println("OK");
    } else {
        Serial.println("NAO ENCONTRADO (continuando sem espectro)");
    }

    // --- Sensor de temperatura DS18B20 ---
    Serial.print("  Sensor DS18B20... ");
    ds18b20.begin();
    if (ds18b20.getDeviceCount() > 0) {
        ds18b20_ok = true;
        ds18b20.setResolution(12);
        Serial.println("OK");
    } else {
        Serial.println("NAO ENCONTRADO (usando ADC fallback)");
    }

    // --- Modelo TFLite ---
    Serial.print("  Modelo TFLite... ");
    model = tflite::GetModel(liquid_classifier);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("ERRO: versao do schema incompativel!");
#ifndef NO_OLED
        if (oled_ok) {
            display.clearDisplay();
            display.setCursor(0, 0);
            display.println("ERRO: Modelo TFLite");
            display.println("versao incompativel!");
            display.display();
        }
#endif
        while (1) { set_led(255, 0, 0); delay(500); set_led(0, 0, 0); delay(500); }
    }

    static tflite::AllOpsResolver resolver;
    static tflite::MicroErrorReporter micro_error_reporter;
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, &micro_error_reporter);
    interpreter_tfl = &static_interpreter;

    TfLiteStatus alloc_status = interpreter_tfl->AllocateTensors();
    if (alloc_status != kTfLiteOk) {
        Serial.println("ERRO: falha ao alocar tensores!");
        while (1) { set_led(255, 0, 0); delay(500); set_led(0, 0, 0); delay(500); }
    }

    input_tensor  = interpreter_tfl->input(0);
    output_tensor = interpreter_tfl->output(0);

    Serial.println("OK");
    Serial.print("    Arena: ");
    Serial.print(interpreter_tfl->arena_used_bytes());
    Serial.print("/");
    Serial.print(kTensorArenaSize);
    Serial.println(" bytes");

    // Pronto
    Serial.println("============================================");
    Serial.println("  PRONTO! Comandos via Serial Monitor:");
    Serial.println("    <ENTER>  = classificar uma vez");
    Serial.println("    auto     = modo continuo (5s)");
    Serial.println("    Botao BOOT = classificar uma vez");
    Serial.println("============================================");
    Serial.println();

    set_led(0, 50, 0);  // LED verde = pronto
    oled_tela_inicial();
}

// ============================================================
//  LOOP
// ============================================================

void loop() {
    bool executar = false;

    // Botão BOOT pressionado?
    if (digitalRead(PIN_BOTAO) == LOW) {
        delay(50);  // debounce
        if (digitalRead(PIN_BOTAO) == LOW) {
            executar = true;
            while (digitalRead(PIN_BOTAO) == LOW) delay(10);  // esperar soltar
        }
    }

    // Comando serial?
    if (Serial.available() > 0) {
        String cmd = Serial.readStringUntil('\n');
        cmd.trim();
        if (cmd == "auto") {
            modo_continuo = !modo_continuo;
            Serial.print("  Modo continuo: ");
            Serial.println(modo_continuo ? "ATIVADO (5s)" : "DESATIVADO");
        } else {
            executar = true;
        }
    }

    // Modo contínuo
    if (modo_continuo && (millis() - ultimo_scan >= INTERVALO_SCAN)) {
        executar = true;
        ultimo_scan = millis();
    }

    if (executar) {
        classificar();
    }
}

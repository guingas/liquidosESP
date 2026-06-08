# LiquidClassifier — Identificação de Líquidos com ESP32 e Machine Learning

Sistema de classificação de líquidos em tempo real usando ESP32, sensores multimodais e um modelo de Machine Learning (TFLite int8) embarcado.

**Classifica 4 tipos de líquidos:**
- 💧 Água
- ⛽ Gasolina
- 🧃 Suco
- 🍺 Bebida Alcoólica

## Resultado na Tela (Serial Monitor)

```
  ╔══════════════════════════════════════╗
  ║  LIQUIDO: Agua                       ║
  ║  CONFIANCA: 98.7%                    ║
  ╠══════════════════════════════════════╣
  ║  Temp:   23.5 C   pH:  7.01         ║
  ║  Cond:    520 uS/cm  Freq: 1380 Hz  ║
  ╚══════════════════════════════════════╝
```

---

## Hardware Necessário

| Componente | Função | Pino ESP32 | Custo Aprox. |
|---|---|---|---|
| ESP32 DevKit V1 | Microcontrolador | — | R$ 30 |
| AS7341 | Sensor espectral 11 canais | I2C (SDA=21, SCL=22) | R$ 45 |
| DS18B20 | Temperatura | GPIO 4 (OneWire) | R$ 8 |
| Eletrodos PCB | Condutividade elétrica | ADC GPIO 34 | R$ 5 |
| Módulo pH E-201-C | pH do líquido | ADC GPIO 35 | R$ 25 |
| Disco piezoelétrico | Frequência acústica | ADC GPIO 32 | R$ 3 |
| LED RGB | Feedback visual por cor | GPIO 25, 26, 27 | R$ 2 |
| **OLED SSD1306 128x64** | **Display (OPCIONAL)** | **I2C (0x3C)** | **R$ 15** |

**Custo total mínimo: ~R$ 118** (sem o display OLED)

---

## Diagrama de Ligação

```
ESP32 DevKit V1
┌─────────────────────┐
│                     │
│  GPIO 4  ──────────── DS18B20 (DATA) ── resistor 4.7kΩ ── 3.3V
│                     │
│  GPIO 34 ──────────── Eletrodos Condutividade (sinal)
│  GPIO 35 ──────────── Módulo pH E-201-C (sinal)
│  GPIO 32 ──────────── Disco Piezoelétrico (sinal)
│                     │
│  GPIO 21 (SDA) ────── AS7341 SDA ── OLED SDA (opcional)
│  GPIO 22 (SCL) ────── AS7341 SCL ── OLED SCL (opcional)
│                     │
│  GPIO 25 ──────────── LED Vermelho (+ resistor 220Ω)
│  GPIO 26 ──────────── LED Verde    (+ resistor 220Ω)
│  GPIO 27 ──────────── LED Azul     (+ resistor 220Ω)
│                     │
│  3.3V ─────────────── VCC sensores
│  GND ──────────────── GND sensores
└─────────────────────┘
```

> **Nota:** O AS7341 e o OLED SSD1306 compartilham o mesmo barramento I2C (endereços diferentes: 0x39 e 0x3C).

---

## Instalação no Arduino IDE

### 1. Instalar suporte ao ESP32

1. Abra **Arduino IDE** → `Arquivo` → `Preferências`
2. Em **URLs Adicionais de Gerenciadores de Placas**, adicione:
   ```
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
   ```
3. Vá em `Ferramentas` → `Placa` → `Gerenciador de Placas`
4. Pesquise **esp32** e instale **"ESP32 by Espressif Systems"**
5. Selecione a placa: `Ferramentas` → `Placa` → `ESP32 Dev Module`

### 2. Instalar bibliotecas

Vá em `Ferramentas` → `Gerenciar Bibliotecas` e instale:

| Biblioteca | Autor | Obrigatória? |
|---|---|---|
| **Adafruit AS7341** | Adafruit | ✅ Sim |
| **OneWire** | Paul Stoffregen | ✅ Sim |
| **DallasTemperature** | Miles Burton | ✅ Sim |
| **TensorFlowLite_ESP32** | Masayuki Tanaka | ✅ Sim |
| **Adafruit SSD1306** | Adafruit | ⬜ Só se usar OLED |
| **Adafruit GFX Library** | Adafruit | ⬜ Só se usar OLED |

### 3. Corrigir flag de compilação (se necessário)

Se o TensorFlowLite_ESP32 não compilar, adicione `-fpermissive` nas flags:

1. Encontre a pasta de instalação do ESP32:
   - **Windows:** `%LOCALAPPDATA%\Arduino15\packages\esp32\hardware\esp32\<versao>\`
   - **macOS:** `~/Library/Arduino15/packages/esp32/hardware/esp32/<versao>/`
   - **Linux:** `~/.arduino15/packages/esp32/hardware/esp32/<versao>/`
2. Crie (ou edite) o arquivo `platform.local.txt` nessa pasta
3. Adicione a linha:
   ```
   compiler.cpp.extra_flags=-fpermissive
   ```
4. Reinicie o Arduino IDE

### 4. Compilar e fazer upload

1. Abra `LiquidClassifier/LiquidClassifier.ino`
2. Selecione a porta COM correta em `Ferramentas` → `Porta`
3. Clique em **Upload** (→)
4. Abra o **Serial Monitor** (115200 baud)

### Sem display OLED?

Se você **não tem** o display OLED e **não quer instalar** as bibliotecas Adafruit SSD1306/GFX:

1. Abra `LiquidClassifier.ino`
2. Descomente a linha no início:
   ```cpp
   #define NO_OLED
   ```
3. Compile normalmente — nenhuma biblioteca de display será necessária

---

## Como Usar

1. **Conecte** os sensores conforme o diagrama acima
2. **Ligue** o ESP32 via USB ao computador
3. **Abra** o Serial Monitor (115200 baud)
4. **Mergulhe** os sensores no líquido a ser classificado
5. **Pressione** o botão **BOOT** no ESP32 ou envie qualquer texto pelo Serial Monitor
6. O resultado aparece no **Serial Monitor** (e no OLED, se conectado)

### Modo Contínuo

Digite `auto` no Serial Monitor para ativar/desativar o modo contínuo (leitura a cada 5 segundos).

### LED de Feedback

| Cor | Líquido |
|---|---|
| 🔵 Azul | Água |
| 🔴 Vermelho | Bebida Alcoólica |
| 🟡 Amarelo | Gasolina |
| 🟢 Verde | Suco |
| ⚪ Branco | Inicializando |

---

## Estrutura do Projeto

```
liquidosESP/
├── README.md                           ← Este arquivo
├── LiquidClassifier/                   ← Sketch Arduino IDE
│   ├── LiquidClassifier.ino            ← Código principal
│   ├── liquid_classifier.h             ← Modelo TFLite (int8, 22KB)
│   └── normalization_params.h          ← Parâmetros do StandardScaler
├── models/
│   └── tflite/
│       └── liquid_classifier.tflite    ← Modelo binário TFLite
└── training/                           ← Código Python de treinamento
    ├── main.py                         ← Pipeline principal
    ├── config/
    │   └── settings.py                 ← Configurações (classes, features)
    └── src/
        ├── data_generation/            ← Geração de dados sintéticos
        ├── models/                     ← Treinamento e conversão TFLite
        └── preprocessing/             ← Normalização e feature engineering
```

---

## Sobre o Modelo

- **Tipo:** Rede neural (MLP) convertida para TFLite int8
- **Tamanho:** ~22 KB (cabe em qualquer ESP32)
- **Features:** 14 sensores + 8 features derivadas = **22 features**
- **Acurácia no teste:** **99.82%** (classificação de tipo)
- **Features derivadas calculadas no ESP32:**
  - `condutividade_25C` — compensação térmica (α=0.02)
  - `ratio_azul_vermelho` — F3(480nm)/F7(630nm)
  - `ratio_verde_vermelho` — F4(515nm)/F7(630nm)
  - `ratio_nir_clear` — NIR/Clear
  - `ratio_violeta_laranja` — F1(415nm)/F6(590nm)
  - `spectral_mean`, `spectral_std`, `spectral_range`

---

## Calibração dos Sensores

Antes de usar em produção, calibre os sensores analógicos editando as constantes no início do arquivo `.ino`:

```cpp
// pH: calibre com soluções tampão pH 4.0 e pH 7.0
const float PH_OFFSET = 0.0;
const float PH_SLOPE  = 14.0 / 4095.0;

// Condutividade: calibre com solução de KCl 1413 µS/cm
const float COND_FACTOR = 5000.0 / 4095.0;
```

---

## Re-treinar o Modelo

O código de treinamento está na pasta `training/`. Para re-treinar:

```bash
cd training
python -m venv .venv
.venv\Scripts\activate
pip install scikit-learn xgboost tensorflow
python main.py
```

O pipeline gera automaticamente:
- Dados sintéticos de 22 líquidos
- Treinamento e validação de múltiplos modelos
- Conversão para TFLite int8
- Headers C para embarcar no ESP32

---

## Licença

MIT License

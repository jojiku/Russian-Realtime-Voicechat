# Легковесный голосовой чат на русском

## 🎬 Демо
<table class="center">
  <tr style="font-weight: bolder;text-align:center;">
  </tr>
  <tr>
    <td>
      <video src=https://github.com/user-attachments/assets/765ccd70-aeda-4105-ae19-734be95395b1 controls preload></video>
    </td>
  </tr>
</table>

## ✨ Особенности

| Функция | Описание | Технологии
|---------|-------------|-------------
| ➡️ **Минимальная задержка в ~300мс** | <ul><li>Стриминг LLM, STT</li><li>заранее предсказание ответа пользователя</li><li>модель для определения окончания мысли</li></ul> | Silero, Faster whisper, любая локальная или API LLM. Дообучил RuBERT на синт. датасете ([HuggingFace](https://huggingface.co/Silxxor/Russian-Addressee-detector))
| 🔄 **Прерывание** | Естественное прерывание системы во время речи | Внутренняя логика
| 🎯 **Определение адресата** | Понимает, когда вы обращаетесь к ней, а не к кому-то другому | Дообучил RuBERT на ASR транскриптах русских телефонных разговоров ([HuggingFace](https://huggingface.co/Silxxor/russian-turn-detector))

## 🏗️ Архитектура

```
┌─────────────┐     ┌─────────────────┐     ┌─────────────┐
│  Microphone │────▶│  RealtimeSTT    │────▶│    Turn     │
│ (WebSocket) │     │ (Faster Whisper)│     │  Detection  │
└─────────────┘     └─────────────────┘     └──────┬──────┘
                                                   │
                      ┌──────────────┐             │
                      │  Addressee   │◀────────────┤
                      │  Detector    │             │
                      └──────┬───────┘             │
                             │                     │
                             ▼                     ▼
                      ┌──────────────┐     ┌─────────────┐
                      │     LLM      │◀────│  Pipeline   │
                      │  (Streaming) │     │  Manager    │
                      └──────┬───────┘     └─────────────┘
                             │
                             ▼
                      ┌──────────────┐     ┌─────────────┐
                      │  RealtimeTTS │────▶│   Speaker   │
                      │   (Silero)   │     │  (WebSocket)│
                      └──────────────┘     └─────────────┘
```


## 📋 Требования

- **Python** 3.10 - 3.12
- **CUDA** 12.1 (рекомендуется, ~4GB VRAM)
- **Node.js** 18+ (для фронтенда)
- **Poetry** 1.8+ для управления зависимостями

## 🚀 Установка

### 1. Установите Poetry

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### 2. Клонируйте и установите

```bash
git clone https://github.com/jojiku/Russian-Realtime-Voicechat.git
cd Russian-Realtime-Voicechat
poetry install
```

### 3. Настройте окружение

```bash
cd Russian-Realtime-Voicechat/code
cp .env.template .env
```

### 4. Установите фронтенд

```bash
cd Russian-Realtime-Voicechat/code
npm install
```

## ⚙️ Конфигурация

Отредактируйте `.env`:
```env
# Язык
APP_LANG=ru

# LLM бэкенд (выберите один)
LMSTUDIO_BASE_URL=http://127.0.0.1:1234/v1
# OLLAMA_BASE_URL=http://127.0.0.1:11434
# OPENAI_API_KEY=sk-... 
# GEMINI_API_KEY=...
# GROQ_API_KEY=gsk_...

# Архитектура GPU
TORCH_CUDA_ARCH_LIST=7.5
```

## 🎮 Использование

### Запустите сервер
```bash
# Вариант 1: Используя poetry run
poetry run python server.py
```

### Откройте интерфейс

Откройте `http://localhost:3000` в браузере.

## 📁 Структура проекта

```
Russian-Realtime-Voicechat/
├── server.py                  # FastAPI WebSocket сервер
├── speech_pipeline_manager.py # Оркестрация LLM + TTS
├── audio_module.py            # Обработка TTS (Silero)
├── audio_in.py                # Обработка входного аудио
├── transcribe.py              # Обработка STT (Whisper)
├── llm_module.py              # Мультибэкенд интерфейс LLM
├── addressee_detector.py      # "Это обращаются ко мне или нет?"
├── turndetect.py              # Предсказание окончания мысли
├── silero_engine.py           # Обёртка движка Silero TTS
├── pyproject.toml             # Конфигурация Poetry
├── static/                    # Исходники фронтенда
├── dist/                      # Собранный фронтенд (генерируется при запуске)
└── resources/                 # Тут лежит промпт
```

## 📊 Производительность

Показатели при 6 GB VRAM на 1660 TI от последнего слова пользователя до первого аудио чанка от системы:

| Компонент | Задержка | Память
|-----------|---------|---------
| STT (Whisper base) | ~100мс | ~1000 MB
| LLM (vikhr-llama-2b) | ~150мс TTFT | ~3 GB
| TTS (Silero) | ~40мс | ~200 MB
| Turn detection (Rubert) | ~20мс | ~100 MB
| Addressee detection (Rubert) | ~20мс | ~100 MB
| **Весь пайплайн** | **~300мс** | **~5 GB**


## 🙏 Благодарность
Замечательный код от Kolja Beigel и модели от snakers4
- https://github.com/KoljaB/RealtimeTTS
- https://github.com/KoljaB/RealtimeVoiceChat
- https://github.com/KoljaB/RealtimeSTT
- https://github.com/snakers4/open_stt
- https://github.com/snakers4/silero-models

## License 📜
**MIT License**


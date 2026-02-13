# HistoAnalyzer — оффлайн-модели (Windows)

## Где лежат модели
Модели хранятся в:
`%LOCALAPPDATA%\HistoAnalyzer\models`

Пример: `C:\Users\<user>\AppData\Local\HistoAnalyzer\models`.

## Как скачать вручную
1. Откройте **ИИ → Модели**.
2. Нажмите **Скачать**.
3. Приложение скачает `ModelPack.zip`, проверит SHA256, распакует и обновит статусы.

Также можно в CLI:
```bash
python -m app.main --download-models
```

## Как проверить установку
В окне **ИИ → Модели** нажмите **Проверить**. Для каждой модели должен быть статус `установлено`:
- phikon_v2
- cellpose_weights
- stardist_weights
- hovernet_onnx
- sam_checkpoint

## Как перенести models_dir на другой ПК
1. Скопируйте всю папку `%LOCALAPPDATA%\HistoAnalyzer\models` на новый ПК в тот же путь.
2. Запустите HistoAnalyzer и нажмите **ИИ → Модели → Проверить**.
3. Интернет для работы модулей после этого не нужен.

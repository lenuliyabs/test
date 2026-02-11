# HistoAnalyzer (MVP)

Desktop-приложение для анализа гистологических изображений (PySide6).

## Возможности MVP
- Открытие JPG/PNG/TIFF.
- Viewer: zoom колесом, pan (hand tool / middle mouse), fit-to-window.
- Enhance-пайплайн (как «ползунки»): brightness, contrast, highlights, shadows, saturation, warmth, sharpness, noise reduction.
- Segmentation: threshold (Otsu/manual + morph close/open), опциональный Cellpose.
- Ручная правка маски: brush/eraser, undo/redo (20 шагов).
- Морфометрия: line, polyline, area, scale (µm/px), thickness по двум линиям.
- Сохранение/загрузка проекта (JSON + mask PNG).
- Экспорт CSV измерений и PDF-отчета.

## Установка
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .[dev]
```

## Запуск
```bash
python -m app.main
```

## Линт и форматирование
```bash
ruff check .
black .
```

## Тесты
```bash
pytest -q
```

## Cellpose (опционально)
Если `cellpose` не установлен, кнопка **Run Cellpose** будет отключена с подсказкой.

## Сборка Windows (PyInstaller)
```bat
build_win.bat
```
Результат: `dist\HistoAnalyzer\HistoAnalyzer.exe`

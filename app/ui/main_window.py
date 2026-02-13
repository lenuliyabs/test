from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import QSettings, QThreadPool, Qt
from PySide6.QtGui import QAction, QActionGroup, QDragEnterEvent, QDropEvent, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from app.ai_engine import AIEngine
from app.core.image_ops import EnhanceParams, apply_enhancements
from app.core.mask_ops import threshold_segmentation
from app.core.morphometry import (
    CalibrationProfile,
    calculate_calibration_stats,
    convert_measurement_value,
    polygon_area,
    polyline_length,
    recalc_measurements_with_scale,
    thickness_distribution,
)
from app.core.project_io import load_project, save_project
from app.export.csv_export import export_measurements_csv
from app.export.figure_export import FigureExportSettings, export_figure
from app.export.geojson_export import export_geojson
from app.export.pdf_report import export_pdf_report
from app.modules.segmentation.cellpose_runner import CELLPPOSE_AVAILABLE, run_cellpose
from app.modules.segmentation.onnx_runner import ONNX_AVAILABLE, run_onnx_segmentation
from app.core.acceleration import detect_acceleration
from app.models.model_manager import ensure_default_models
from app.ui.i18n_ru import APP_TITLE, DISCLAIMER, STEP_DESCRIPTIONS, STEP_NAMES
from app.ui.model_manager_dialog import ModelManagerDialog
from app.ui.settings_dialog import SettingsDialog
from app.ui.style import APP_QSS
from app.utils.workers import CancellableWorker
from app.viewer.image_view import ImageView

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
PROJECT_EXTS = {".histo", ".json"}


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1520, 940)
        self.setStyleSheet(APP_QSS)
        self.setAcceptDrops(True)

        self.settings = QSettings("HistoAnalyzer", "HistoAnalyzer")
        self.model_bootstrap_messages = []
        self.ai_engine = AIEngine()
        self.thread_pool = QThreadPool.globalInstance()
        self.active_worker: CancellableWorker | None = None

        self.image_path: str | None = None
        self.original: np.ndarray | None = None
        self.enhanced: np.ndarray | None = None
        self.normalized: np.ndarray | None = None
        self.illumination_corrected: np.ndarray | None = None
        self.artifact_mask: np.ndarray | None = None
        self.roi_mask: np.ndarray | None = None
        self.mask: np.ndarray | None = None

        self.enhance_params = EnhanceParams()
        self.measurements: list[dict] = []
        self.annotations: list[dict] = []
        self.um_per_px: float | None = None
        self.scale_mode = "none"
        self.calibration_profiles = self._load_calibration_profiles_global()
        self.active_profile_name: str | None = None

        self.guided_ai_state: dict = {
            "current_step": 0,
            "report_lang": "Русский",
            "steps": {str(i): {"done": False, "params": {}, "metrics": {}} for i in range(12)},
        }

        self.mask_undo: list[np.ndarray] = []
        self.mask_redo: list[np.ndarray] = []
        self.thickness_line1: list[tuple[float, float]] | None = None
        self.pending_scale_context: dict | None = None

        self._build_ui()
        self._build_menu()
        self._build_toolbar()

        if self.settings.value("auto_fetch_models", True, type=bool):
            try:
                self.model_bootstrap_messages = ensure_default_models()
            except Exception:
                self.model_bootstrap_messages = ["Автозагрузка моделей недоступна"]
        self._build_status_bar()
        self._update_scale_status()
        self._update_accel_status()

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.view = ImageView()
        self.view.measurement_finished.connect(self.on_measurement)
        self.view.scale_line_finished.connect(self.on_scale_line_finished)
        self.view.mask_changed.connect(self.on_mask_changed)
        self.view.cursor_moved.connect(self.on_cursor_moved)
        self.view.zoom_changed.connect(self.on_zoom_changed)
        self.view.compare_changed.connect(self.on_compare_changed)
        splitter.addWidget(self.view)
        splitter.setSizes([1300, 1])
        self.setCentralWidget(splitter)

        self.right_dock = QDockWidget("Панель инструментов", self)
        self.right_dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
        dock_widget = QWidget()
        dock_layout = QVBoxLayout(dock_widget)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._enhance_tab(), "Улучшение")
        self.tabs.addTab(self._segmentation_tab(), "Сегментация")
        self.tabs.addTab(self._morphometry_tab(), "Морфометрия")
        self.tabs.addTab(self._guided_ai_tab(), "🧭 Пошаговый ИИ")
        dock_layout.addWidget(self.tabs)

        disc = QLabel(DISCLAIMER)
        disc.setWordWrap(True)
        dock_layout.addWidget(disc)

        self.right_dock.setWidget(dock_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.right_dock)
        self.right_dock.setMinimumWidth(420)

    def _build_menu(self) -> None:
        menu = self.menuBar()
        file_menu = menu.addMenu("Файл")
        file_menu.addAction("Открыть…", self.open_image, QKeySequence.StandardKey.Open)
        file_menu.addAction("Открыть проект…", self.load_project)
        file_menu.addAction("Сохранить проект…", self.save_project, QKeySequence.StandardKey.Save)

        self.recent_menu = file_menu.addMenu("Недавние файлы")
        self._refresh_recent_menu()

        file_menu.addSeparator()
        export_menu = file_menu.addMenu("Экспорт")
        export_menu.addAction("CSV…", self.export_csv)
        export_menu.addAction("PDF…", self.export_pdf)
        export_menu.addAction("GeoJSON…", self.export_geojson)
        export_menu.addAction("Фигура для статьи…", self.export_figure_dialog)
        file_menu.addSeparator()
        file_menu.addAction("Выход", self.close)

        view_menu = menu.addMenu("Вид")
        self.toggle_panel_action = view_menu.addAction("Правая панель")
        self.toggle_panel_action.setCheckable(True)
        self.toggle_panel_action.setChecked(True)
        self.toggle_panel_action.triggered.connect(self.right_dock.setVisible)
        self.dark_mode_action = view_menu.addAction("Тёмная тема")
        self.dark_mode_action.setCheckable(True)
        self.dark_mode_action.toggled.connect(self.on_dark_mode_toggled)
        self.compare_mode_action = view_menu.addAction("Режим сравнения (шторка)")
        self.compare_mode_action.setCheckable(True)
        self.compare_mode_action.toggled.connect(self.toggle_compare_mode)
        view_menu.addAction("Сброс вида", self.view.fit_image)

        tools_menu = menu.addMenu("Инструменты")
        tools_menu.addAction("Настройки…", self.open_settings)
        tools_menu.addAction("Модели ИИ…", self.open_model_manager)

        help_menu = menu.addMenu("Справка")
        help_menu.addAction("О программе", self.show_about)

    def _build_toolbar(self) -> None:
        tb = QToolBar("Инструменты")
        tb.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)
        self.tools_group = QActionGroup(self)
        self.tools_group.setExclusive(True)

        self.tool_actions: dict[str, QAction] = {}
        for label, key, shortcut, icon in [
            ("Рука", "hand", "V", self.style().standardIcon(self.style().SP_ArrowUp)),
            ("Линия", "line", "L", self.style().standardIcon(self.style().SP_LineEditClearButton)),
            (
                "Полилиния",
                "polyline",
                "P",
                self.style().standardIcon(self.style().SP_FileDialogDetailedView),
            ),
            ("Площадь", "area", "A", self.style().standardIcon(self.style().SP_DialogYesButton)),
            ("Кисть", "brush", "B", self.style().standardIcon(self.style().SP_BrowserReload)),
            ("Ластик", "eraser", "E", self.style().standardIcon(self.style().SP_TrashIcon)),
            (
                "Линия масштаба",
                "scale_line",
                "K",
                self.style().standardIcon(self.style().SP_TitleBarShadeButton),
            ),
        ]:
            action = QAction(icon, label, self)
            action.setCheckable(True)
            action.setShortcut(shortcut)
            action.triggered.connect(lambda checked=False, k=key: self.set_tool(k))
            self.tools_group.addAction(action)
            tb.addAction(action)
            self.tool_actions[key] = action
        self.tool_actions["hand"].setChecked(True)
        tb.addSeparator()

        undo_action = QAction("Отменить", self)
        undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        undo_action.triggered.connect(self.undo_mask)
        tb.addAction(undo_action)
        redo_action = QAction("Повторить", self)
        redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        redo_action.triggered.connect(self.redo_mask)
        tb.addAction(redo_action)

        zoom100_action = QAction("100%", self)
        zoom100_action.setShortcut("Ctrl+0")
        zoom100_action.triggered.connect(self.reset_zoom_100)
        tb.addAction(zoom100_action)

        compare_action = QAction("Сравнение (шторка)", self)
        compare_action.setCheckable(True)
        compare_action.toggled.connect(self.toggle_compare_mode)
        tb.addAction(compare_action)

        export_fig_action = QAction("Экспорт фигуры", self)
        export_fig_action.triggered.connect(self.export_figure_dialog)
        tb.addAction(export_fig_action)

    def _build_status_bar(self) -> None:
        status = QStatusBar()
        self.setStatusBar(status)
        self.zoom_label = QLabel("Масштаб: 100%")
        self.coords_label = QLabel("x: -, y: -")
        self.tool_label = QLabel("Инструмент: рука")
        self.brush_label = QLabel("Кисть: 10")
        self.step_label = QLabel("Шаг мастера: 0")
        self.scale_label = QLabel("Калибровка: не задана")
        self.compare_label = QLabel("Сравнение: OFF")
        self.accel_label = QLabel("Устройство: CPU")
        for w in [
            self.zoom_label,
            self.coords_label,
            self.tool_label,
            self.brush_label,
            self.step_label,
            self.scale_label,
            self.compare_label,
            self.accel_label,
        ]:
            status.addPermanentWidget(w)

    def _enhance_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.enhance_sliders: dict[str, QSlider] = {}
        self.enhance_value_labels: dict[str, QLabel] = {}
        labels = [
            ("brightness", "Экспозиция/яркость", -100, 100),
            ("contrast", "Контраст", -100, 100),
            ("highlights", "Света", -100, 100),
            ("shadows", "Тени", -100, 100),
            ("saturation", "Насыщенность", -100, 100),
            ("warmth", "Теплота", -100, 100),
            ("sharpness", "Резкость", -100, 100),
            ("noise_reduction", "Шумоподавление", 0, 100),
        ]
        for key, text, vmin, vmax in labels:
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            lbl = QLabel(text)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(vmin, vmax)
            slider.valueChanged.connect(self.on_enhance_changed)
            val = QLabel("0")
            reset_btn = QPushButton("Сброс")
            reset_btn.clicked.connect(lambda checked=False, k=key: self.reset_enhance_param(k))
            self.enhance_sliders[key] = slider
            self.enhance_value_labels[key] = val
            rl.addWidget(lbl, 2)
            rl.addWidget(slider, 4)
            rl.addWidget(val, 1)
            rl.addWidget(reset_btn, 1)
            layout.addWidget(row)
        btn = QPushButton("Сбросить всё")
        btn.clicked.connect(self.reset_enhance)
        layout.addWidget(btn)
        layout.addStretch()
        return tab

    def _segmentation_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        form = QFormLayout()
        self.threshold_mode = QComboBox()
        self.threshold_mode.addItems(["otsu", "manual"])
        self.manual_threshold = QSlider(Qt.Orientation.Horizontal)
        self.manual_threshold.setRange(0, 255)
        self.manual_threshold.setValue(128)
        self.close_size = QSlider(Qt.Orientation.Horizontal)
        self.close_size.setRange(1, 15)
        self.close_size.setValue(3)
        self.open_size = QSlider(Qt.Orientation.Horizontal)
        self.open_size.setRange(1, 15)
        self.open_size.setValue(3)
        self.mask_alpha = QSlider(Qt.Orientation.Horizontal)
        self.mask_alpha.setRange(0, 100)
        self.mask_alpha.setValue(40)
        self.mask_alpha.valueChanged.connect(lambda v: self.view.set_mask_opacity(v / 100.0))
        form.addRow("Режим порога", self.threshold_mode)
        form.addRow("Порог вручную", self.manual_threshold)
        form.addRow("Close size", self.close_size)
        form.addRow("Open size", self.open_size)
        form.addRow("Прозрачность оверлея", self.mask_alpha)
        layout.addLayout(form)

        run_btn = QPushButton("Запустить сегментацию")
        run_btn.setProperty("primary", True)
        run_btn.clicked.connect(self.run_threshold)
        layout.addWidget(run_btn)

        self.cellpose_model = QComboBox()
        self.cellpose_model.addItems(["cyto", "nuclei"])
        self.cellpose_diameter = QLineEdit("auto")
        layout.addWidget(QLabel("Cellpose модель"))
        layout.addWidget(self.cellpose_model)
        layout.addWidget(QLabel("Диаметр"))
        layout.addWidget(self.cellpose_diameter)
        self.seg_backend_combo = QComboBox()
        self.seg_backend_combo.addItems(["Fallback CV", "Cellpose", "ONNX"])
        self.seg_backend_combo.setToolTip("Выбор бэкенда сегментации")
        layout.addWidget(QLabel("Бэкенд сегментации"))
        layout.addWidget(self.seg_backend_combo)

        self.onnx_model_path = QLineEdit()
        self.onnx_model_path.setPlaceholderText("Путь к .onnx модели (опционально)")
        layout.addWidget(self.onnx_model_path)

        self.cellpose_btn = QPushButton("Запустить Cellpose")
        self.cellpose_btn.clicked.connect(self.run_cellpose_clicked)
        if not CELLPPOSE_AVAILABLE:
            self.cellpose_btn.setEnabled(False)
            self.cellpose_btn.setToolTip("Установите cellpose для использования")
        layout.addWidget(self.cellpose_btn)

        if not ONNX_AVAILABLE:
            self.onnx_model_path.setToolTip("onnxruntime не установлен — ONNX недоступен")

        layers_box = QGroupBox("Слои")
        lb = QVBoxLayout(layers_box)
        self.layer_checks: dict[str, QCheckBox] = {}
        for key, label in [
            ("original", "Оригинал"),
            ("normalized", "Нормализованное"),
            ("illumination", "Коррекция освещения"),
            ("enhanced", "Улучшенное"),
            ("artifacts", "Маска артефактов"),
            ("roi", "ROI-рамки"),
            ("segmentation", "Оверлей сегментации"),
        ]:
            cb = QCheckBox(label)
            cb.setChecked(self.view.layer_visible.get(key, False))
            cb.toggled.connect(lambda checked, k=key: self.view.set_layer_visibility(k, checked))
            self.layer_checks[key] = cb
            lb.addWidget(cb)
        layout.addWidget(layers_box)

        compare_group = QGroupBox("Сравнение (шторка)")
        cform = QFormLayout(compare_group)
        self.compare_left_combo = QComboBox()
        self.compare_right_combo = QComboBox()
        for n in ["original", "normalized", "illumination", "enhanced"]:
            self.compare_left_combo.addItem(n)
            self.compare_right_combo.addItem(n)
        self.compare_left_combo.setCurrentText("original")
        self.compare_right_combo.setCurrentText("enhanced")
        self.compare_left_combo.currentTextChanged.connect(self.on_compare_layers_changed)
        self.compare_right_combo.currentTextChanged.connect(self.on_compare_layers_changed)
        cform.addRow("Слой слева", self.compare_left_combo)
        cform.addRow("Слой справа", self.compare_right_combo)
        layout.addWidget(compare_group)

        self.brush_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_slider.setRange(1, 80)
        self.brush_slider.setValue(10)
        self.brush_slider.valueChanged.connect(self.on_brush_size_changed)
        layout.addWidget(QLabel("Размер кисти"))
        layout.addWidget(self.brush_slider)
        layout.addStretch()
        return tab

    def _morphometry_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        cal_group = QGroupBox("📏 Калибровка масштаба")
        form = QFormLayout(cal_group)
        self.scale_mode_combo = QComboBox()
        self.scale_mode_combo.addItems(
            [
                "Без калибровки (px)",
                "По линии на изображении",
                "По калибровочному стеклу (микрометр)",
            ]
        )
        self.profile_combo = QComboBox()
        self._refresh_profile_combo()
        self.profile_combo.currentTextChanged.connect(self.on_profile_selected)
        self.um_edit = QLineEdit()
        self.um_edit.setPlaceholderText("мкм на пиксель")

        form.addRow("Режим", self.scale_mode_combo)
        form.addRow("Профиль", self.profile_combo)
        form.addRow("µm/px", self.um_edit)

        btn_line = QPushButton("Калибровать по линии")
        btn_line.clicked.connect(self.start_line_calibration)
        btn_micro = QPushButton("Мастер микрометра")
        btn_micro.clicked.connect(self.run_micrometer_calibration_wizard)
        btn_apply_profile = QPushButton("Применить профиль")
        btn_apply_profile.clicked.connect(self.apply_selected_profile)
        form.addRow(btn_line, btn_micro)
        form.addRow(btn_apply_profile)
        layout.addWidget(cal_group)

        self.results_table = QTableWidget(0, 6)
        self.results_table.setHorizontalHeaderLabels(
            ["Тип", "Значение (px)", "Значение (мкм/мм)", "Единицы", "Детали", "Время"]
        )
        self.results_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.results_table)

        actions = QHBoxLayout()
        del_btn = QPushButton("Удалить выбранное")
        del_btn.clicked.connect(self.delete_selected_measurement)
        clear_btn = QPushButton("Очистить")
        clear_btn.clicked.connect(self.clear_measurements)
        csv_btn = QPushButton("Экспорт CSV")
        csv_btn.clicked.connect(self.export_csv)
        actions.addWidget(del_btn)
        actions.addWidget(clear_btn)
        actions.addWidget(csv_btn)
        layout.addLayout(actions)

        self.summary_label = QLabel("Сводка: отсутствует")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)
        return tab

    def _guided_ai_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.step_list = QListWidget()
        for i, name in enumerate(STEP_NAMES):
            item = QListWidgetItem(name)
            self.step_list.addItem(item)
            if i == 0:
                self.step_list.setCurrentItem(item)
        self.step_list.currentRowChanged.connect(self.on_step_changed)

        self.step_desc_label = QLabel(STEP_DESCRIPTIONS[0])
        self.step_desc_label.setWordWrap(True)
        self.step_reco_label = QLabel("Рекомендации будут показаны после анализа")
        self.step_reco_label.setWordWrap(True)
        self.step_metrics = QTextEdit()
        self.step_metrics.setReadOnly(True)

        btn_row = QHBoxLayout()
        self.btn_preview_step = QPushButton("Предпросмотр")
        self.btn_preview_step.clicked.connect(self.preview_current_step)
        self.btn_apply_step = QPushButton("Применить")
        self.btn_apply_step.clicked.connect(self.apply_current_step)
        self.btn_reset_step = QPushButton("Сбросить шаг")
        self.btn_reset_step.clicked.connect(self.reset_current_step)
        self.btn_done_step = QPushButton("Отметить выполненным")
        self.btn_done_step.clicked.connect(self.mark_step_done)
        btn_row.addWidget(self.btn_preview_step)
        btn_row.addWidget(self.btn_apply_step)
        btn_row.addWidget(self.btn_reset_step)
        btn_row.addWidget(self.btn_done_step)

        nav_row = QHBoxLayout()
        self.btn_next_step = QPushButton("Далее")
        self.btn_next_step.clicked.connect(self.next_step)
        self.btn_prev_step = QPushButton("Назад")
        self.btn_prev_step.clicked.connect(self.prev_step)
        nav_row.addWidget(self.btn_prev_step)
        nav_row.addWidget(self.btn_next_step)

        self.report_lang_combo = QComboBox()
        self.report_lang_combo.addItems(["Русский", "English"])
        self.report_lang_combo.currentTextChanged.connect(self.on_report_lang_changed)

        self.worker_status_label = QLabel("Фоновые задачи: нет")
        self.btn_cancel_worker = QPushButton("Отменить задачу")
        self.btn_cancel_worker.clicked.connect(self.cancel_active_worker)
        self.btn_cancel_worker.setEnabled(False)

        layout.addWidget(QLabel("Этапы мастера"))
        layout.addWidget(self.step_list)
        layout.addWidget(QLabel("Что делает этот шаг"))
        layout.addWidget(self.step_desc_label)
        layout.addWidget(QLabel("Рекомендованные настройки"))
        layout.addWidget(self.step_reco_label)
        layout.addWidget(QLabel("Результаты шага"))
        layout.addWidget(self.step_metrics)
        layout.addWidget(self.worker_status_label)
        layout.addWidget(self.btn_cancel_worker)
        layout.addLayout(btn_row)
        layout.addLayout(nav_row)
        layout.addWidget(QLabel("Язык текста отчёта"))
        layout.addWidget(self.report_lang_combo)
        self.batch_btn = QPushButton("Пакетная обработка по рецепту")
        self.batch_btn.clicked.connect(self.run_batch_mode)
        layout.addWidget(self.batch_btn)
        return tab

    # ---------- базовые обработчики ----------
    def set_tool(self, tool: str) -> None:
        self.view.set_tool(tool)
        self.tool_label.setText(f"Инструмент: {tool}")

    def on_brush_size_changed(self, value: int) -> None:
        self.view.brush_size = value
        self.brush_label.setText(f"Кисть: {value}")

    def on_cursor_moved(self, x: int, y: int) -> None:
        self.coords_label.setText(f"x: {x}, y: {y}")

    def on_zoom_changed(self, zoom: float) -> None:
        self.zoom_label.setText(f"Масштаб: {zoom:.0f}%")

    def reset_zoom_100(self) -> None:
        self.view.resetTransform()
        self.on_zoom_changed(self.view.current_zoom_percent())

    def toggle_compare_mode(self, enabled: bool) -> None:
        self.view.set_compare_mode(enabled)
        if hasattr(self, "compare_mode_action") and self.compare_mode_action.isChecked() != enabled:
            self.compare_mode_action.blockSignals(True)
            self.compare_mode_action.setChecked(enabled)
            self.compare_mode_action.blockSignals(False)
        self.on_compare_changed(enabled, self.view.split_ratio)

    def on_compare_layers_changed(self) -> None:
        if not hasattr(self, "compare_left_combo"):
            return
        self.view.set_compare_layers(
            self.compare_left_combo.currentText(), self.compare_right_combo.currentText()
        )

    def on_compare_changed(self, enabled: bool, ratio: float) -> None:
        if enabled:
            self.compare_label.setText(f"Сравнение: ON ({ratio * 100:.0f}%)")
        else:
            self.compare_label.setText("Сравнение: OFF")

    def _update_accel_status(self) -> None:
        pref = self.settings.value("accel_vendor", "Auto", type=str)
        info = detect_acceleration(pref)
        self.accel_label.setText(f"Устройство: {info.vendor} {info.backend}")

    def open_settings(self) -> None:
        dlg = SettingsDialog(self)
        if dlg.exec():
            self._update_accel_status()

    def open_model_manager(self) -> None:
        dlg = ModelManagerDialog(self)
        dlg.exec()

    def on_dark_mode_toggled(self, checked: bool) -> None:
        from PySide6.QtWidgets import QApplication

        from app.ui.style import apply_dark_palette, apply_light_palette

        app = QApplication.instance()
        if app is None:
            return
        if checked:
            apply_dark_palette(app)
        else:
            apply_light_palette(app)
        app.setStyleSheet(APP_QSS)

    def show_about(self) -> None:
        pref = self.settings.value("accel_vendor", "Auto", type=str)
        info = detect_acceleration(pref)
        QMessageBox.information(
            self,
            "О программе",
            f"{APP_TITLE}\n{DISCLAIMER}\nУстройство: {info.vendor} / {info.backend}",
        )

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        urls = event.mimeData().urls() if event.mimeData().hasUrls() else []
        if any(Path(u.toLocalFile()).suffix.lower() in IMAGE_EXTS | PROJECT_EXTS for u in urls):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:
        urls = event.mimeData().urls() if event.mimeData().hasUrls() else []
        for url in urls:
            if url.isLocalFile() and self.open_path(url.toLocalFile()):
                event.acceptProposedAction()
                return
        super().dropEvent(event)

    # ---------- открытие/сохранение ----------
    def open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Открыть изображение", "", "Изображения (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)"
        )
        if path:
            self._load_image_file(path)

    def _load_image_file(self, path: str) -> None:
        arr = np.array(Image.open(path).convert("RGB"))
        self.image_path = path
        self.original = arr
        self.enhanced = arr.copy()
        self.normalized = None
        self.illumination_corrected = None
        self.artifact_mask = np.zeros(arr.shape[:2], dtype=np.uint8)
        self.roi_mask = np.zeros(arr.shape[:2], dtype=np.uint8)
        self.mask = np.zeros(arr.shape[:2], dtype=np.uint8)
        self.measurements.clear()
        self.annotations.clear()
        self.refresh_measurements_table()
        self.mask_undo.clear()
        self.mask_redo.clear()

        self.view.set_images(self.original, self.enhanced, self.mask)
        self.view.set_layer_image("normalized", self.normalized)
        self.view.set_layer_image("illumination", self.illumination_corrected)
        self.view.set_layer_image("artifacts", self.artifact_mask)
        self.view.set_layer_image("roi", self.roi_mask)
        self.view.fit_image()
        self._add_recent_file(path)

    def save_project(self) -> None:
        if self.image_path is None or self.mask is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить проект", "", "Проект (*.histo *.json)"
        )
        if not path:
            return
        if Path(path).suffix.lower() not in PROJECT_EXTS:
            path = f"{path}.histo"

        save_project(
            path,
            self.image_path,
            self.enhance_params,
            self.mask,
            self.annotations,
            self.measurements,
            self.um_per_px,
            guided_ai=self.guided_ai_state,
            calibration_profiles=[p.to_dict() for p in self.calibration_profiles],
            active_calibration_profile=self.active_profile_name,
        )
        self._add_recent_file(path)

    def load_project(self, path: str | None = None) -> None:
        if path is None:
            path, _ = QFileDialog.getOpenFileName(
                self, "Открыть проект", "", "Проект (*.histo *.json)"
            )
        if not path:
            return
        self._load_project_file(path)

    def _load_project_file(self, path: str) -> None:
        payload, mask = load_project(path)
        image_path = payload["image_path"]
        if not Path(image_path).exists():
            QMessageBox.warning(self, "Ошибка", f"Исходное изображение не найдено: {image_path}")
            return
        self._load_image_file(image_path)
        self.enhance_params = EnhanceParams.from_dict(payload.get("enhance", {}))
        for key, slider in self.enhance_sliders.items():
            slider.setValue(getattr(self.enhance_params, key))
        self.on_enhance_changed()

        self.mask = cv2.resize(
            mask, (self.original.shape[1], self.original.shape[0]), interpolation=cv2.INTER_NEAREST
        )
        self.view.update_mask(self.mask)
        self.annotations = payload.get("annotations", [])
        self.measurements = payload.get("measurements", [])
        self.um_per_px = payload.get("um_per_px")
        self.guided_ai_state = payload.get("guided_ai", self.guided_ai_state)
        self.calibration_profiles = [
            CalibrationProfile.from_dict(p) for p in payload.get("calibration_profiles", [])
        ]
        self.active_profile_name = payload.get("active_calibration_profile")
        self._refresh_profile_combo()
        if self.um_per_px is not None:
            self.um_edit.setText(f"{self.um_per_px:.6f}")
        self.refresh_measurements_table()
        self._update_scale_status()
        self._add_recent_file(path)

    def open_path(self, path: str) -> bool:
        p = Path(path)
        if not p.exists() or not p.is_file():
            return False
        ext = p.suffix.lower()
        try:
            if ext in PROJECT_EXTS:
                self._load_project_file(str(p))
                return True
            if ext in IMAGE_EXTS:
                self._load_image_file(str(p))
                return True
            QMessageBox.warning(self, "Открытие", f"Неподдерживаемый файл: {p.name}")
        except Exception as exc:
            QMessageBox.warning(self, "Ошибка открытия", str(exc))
        return False

    def _add_recent_file(self, path: str) -> None:
        recent = self.settings.value("recent_files", [], type=list)
        path = str(Path(path).resolve())
        recent = [path] + [p for p in recent if p != path]
        self.settings.setValue("recent_files", recent[:10])
        self._refresh_recent_menu()

    def _refresh_recent_menu(self) -> None:
        if not hasattr(self, "recent_menu"):
            return
        self.recent_menu.clear()
        recent = self.settings.value("recent_files", [], type=list)
        existing = [p for p in recent if Path(p).exists()]
        if not existing:
            action = self.recent_menu.addAction("(пусто)")
            action.setEnabled(False)
            return
        for idx, path in enumerate(existing[:10], start=1):
            action = self.recent_menu.addAction(f"{idx}. {Path(path).name}")
            action.setToolTip(path)
            action.triggered.connect(lambda checked=False, p=path: self.open_path(p))
        self.recent_menu.addSeparator()
        self.recent_menu.addAction("Очистить список", self._clear_recent)

    def _clear_recent(self) -> None:
        self.settings.setValue("recent_files", [])
        self._refresh_recent_menu()

    # ---------- Guided AI ----------
    def on_step_changed(self, step_idx: int) -> None:
        if step_idx < 0:
            return
        self.guided_ai_state["current_step"] = step_idx
        self.step_label.setText(f"Шаг мастера: {step_idx}")
        self.step_desc_label.setText(STEP_DESCRIPTIONS.get(step_idx, ""))
        step_state = self.guided_ai_state["steps"].get(str(step_idx), {})
        rec = step_state.get("reco", "Рекомендации появятся после предпросмотра")
        self.step_reco_label.setText(str(rec))
        self.step_metrics.setPlainText(str(step_state.get("metrics", {})))
        self.btn_next_step.setEnabled(bool(step_state.get("done", False) or step_idx == 11))

    def _run_worker(self, fn, on_finished) -> None:
        if self.active_worker is not None:
            QMessageBox.information(self, "Задача", "Дождитесь завершения текущей задачи")
            return
        worker = CancellableWorker(fn)
        self.active_worker = worker
        self.btn_cancel_worker.setEnabled(True)
        self.worker_status_label.setText("Фоновые задачи: выполняется...")

        def _done(result):
            self.active_worker = None
            self.btn_cancel_worker.setEnabled(False)
            self.worker_status_label.setText("Фоновые задачи: завершено")
            on_finished(result)

        def _err(msg: str):
            self.active_worker = None
            self.btn_cancel_worker.setEnabled(False)
            self.worker_status_label.setText("Фоновые задачи: ошибка")
            QMessageBox.warning(self, "Ошибка фоновой задачи", msg)

        worker.signals.finished.connect(_done)
        worker.signals.error.connect(_err)
        self.thread_pool.start(worker)

    def cancel_active_worker(self) -> None:
        if self.active_worker is not None:
            self.active_worker.cancel()
            self.worker_status_label.setText("Фоновые задачи: отмена запрошена")

    def preview_current_step(self) -> None:
        if self.original is None:
            return
        step_idx = self.step_list.currentRow()
        stage_map = {
            1: "quality_check",
            2: "stain_normalization",
            3: "illumination",
            4: "artifacts",
            7: "segmentation",
        }
        stage = stage_map.get(step_idx, "quality_check")

        def _work():
            return self.ai_engine.preview_stage(stage, self.original, None, {})

        def _finish(result):
            if step_idx == 2:
                self.normalized = result.image
                self.view.set_layer_image("normalized", self.normalized)
                self.layer_checks["normalized"].setChecked(True)
            elif step_idx == 3:
                self.illumination_corrected = result.image
                self.view.set_layer_image("illumination", self.illumination_corrected)
                self.layer_checks["illumination"].setChecked(True)
            elif step_idx == 4:
                gray = cv2.cvtColor(result.image, cv2.COLOR_RGB2GRAY)
                self.artifact_mask = (gray > 0).astype(np.uint8)
                self.view.set_layer_image("artifacts", self.artifact_mask)
                self.layer_checks["artifacts"].setChecked(True)
            self.guided_ai_state["steps"][str(step_idx)]["metrics"] = result.metrics
            self.guided_ai_state["steps"][str(step_idx)]["reco"] = result.suggestions
            self.step_reco_label.setText(str(result.suggestions))
            self.step_metrics.setPlainText(str(result.metrics))

        self._run_worker(_work, _finish)

    def apply_current_step(self) -> None:
        if self.original is None:
            return
        step_idx = self.step_list.currentRow()
        stage_map = {
            2: "stain_normalization",
            3: "illumination",
            4: "artifacts",
            6: "quality_check",
            7: "segmentation",
        }
        stage = stage_map.get(step_idx, "quality_check")

        def _work():
            if step_idx == 7:
                backend = self.seg_backend_combo.currentText()
                if backend == "ONNX":
                    model = self.onnx_model_path.text().strip()
                    if model and ONNX_AVAILABLE:
                        mask, conf = run_onnx_segmentation(self.original, model)
                        over = self.original.copy()
                        over[mask > 0] = (
                            over[mask > 0] * 0.6 + np.array([255, 0, 0]) * 0.4
                        ).astype(np.uint8)
                        return type(
                            "R",
                            (),
                            {
                                "image": over,
                                "metrics": {"coverage": float(mask.mean()), "confidence": conf},
                            },
                        )()
                if backend == "Cellpose" and CELLPPOSE_AVAILABLE:
                    c_mask = run_cellpose(self.original)
                    over = self.original.copy()
                    over[c_mask > 0] = (
                        over[c_mask > 0] * 0.6 + np.array([255, 0, 0]) * 0.4
                    ).astype(np.uint8)
                    return type(
                        "R",
                        (),
                        {
                            "image": over,
                            "metrics": {"coverage": float(c_mask.mean()), "confidence": 0.7},
                        },
                    )()
            return self.ai_engine.apply_stage(stage, self.original, [], {})

        def _finish(result):
            if step_idx in {2, 3, 6}:
                self.enhanced = result.image
                self.view.update_enhanced(self.enhanced)
                self.layer_checks["enhanced"].setChecked(True)
            if step_idx == 7:
                gray = cv2.cvtColor(result.image, cv2.COLOR_RGB2GRAY)
                self.mask = (gray > int(gray.mean())).astype(np.uint8)
                self.view.update_mask(self.mask)
            self.guided_ai_state["steps"][str(step_idx)]["applied_at"] = datetime.now().isoformat()
            self.step_metrics.setPlainText(
                f"Применено: {datetime.now().strftime('%H:%M:%S')}\n{result.metrics}"
            )

        self._run_worker(_work, _finish)

    def reset_current_step(self) -> None:
        step_idx = self.step_list.currentRow()
        self.guided_ai_state["steps"][str(step_idx)] = {"done": False, "params": {}, "metrics": {}}
        self.step_reco_label.setText("Рекомендации будут показаны после анализа")
        self.step_metrics.clear()
        self.on_step_changed(step_idx)

    def mark_step_done(self) -> None:
        step_idx = self.step_list.currentRow()
        self.guided_ai_state["steps"][str(step_idx)]["done"] = True
        self.on_step_changed(step_idx)

    def next_step(self) -> None:
        idx = self.step_list.currentRow()
        if not self.guided_ai_state["steps"].get(str(idx), {}).get("done", False):
            answer = QMessageBox.question(
                self,
                "Подтверждение",
                "Шаг не отмечен как выполненный. Продолжить без выполнения?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        if idx < self.step_list.count() - 1:
            self.step_list.setCurrentRow(idx + 1)

    def prev_step(self) -> None:
        idx = self.step_list.currentRow()
        if idx > 0:
            self.step_list.setCurrentRow(idx - 1)

    def on_report_lang_changed(self, value: str) -> None:
        self.guided_ai_state["report_lang"] = value

    # ---------- морфометрия/калибровка ----------
    def on_profile_selected(self, name: str) -> None:
        self.active_profile_name = None if name == "—" else name

    def _refresh_profile_combo(self) -> None:
        if not hasattr(self, "profile_combo"):
            return
        self.profile_combo.clear()
        self.profile_combo.addItem("—")
        for p in self.calibration_profiles:
            self.profile_combo.addItem(p.name)
        if self.active_profile_name:
            idx = self.profile_combo.findText(self.active_profile_name)
            if idx >= 0:
                self.profile_combo.setCurrentIndex(idx)

    def _load_calibration_profiles_global(self) -> list[CalibrationProfile]:
        raw = self.settings.value("calibration_profiles", [], type=list)
        return [CalibrationProfile.from_dict(x) for x in raw]

    def _save_calibration_profiles_global(self) -> None:
        self.settings.setValue(
            "calibration_profiles", [p.to_dict() for p in self.calibration_profiles]
        )

    def start_line_calibration(self) -> None:
        self.pending_scale_context = {"mode": "line"}
        self.set_tool("scale_line")
        QMessageBox.information(self, "Калибровка", "Нарисуйте линию масштаба на изображении")

    def run_micrometer_calibration_wizard(self) -> None:
        if self.original is None:
            QMessageBox.warning(self, "Калибровка", "Сначала откройте изображение микрометра")
            return
        objective, ok = QInputDialog.getItem(
            self,
            "Профиль увеличения",
            "Выберите профиль",
            ["4x", "10x", "20x", "40x", "100x", "свой"],
            1,
            False,
        )
        if not ok:
            return

        scale_type, ok = QInputDialog.getItem(
            self,
            "Тип микрометра",
            "Выберите тип",
            [
                "1 мм / 100 делений (10 мкм)",
                "2 мм / 200 делений (10 мкм)",
                "Другое",
            ],
            0,
            False,
        )
        if not ok:
            return

        um_per_div = 10.0
        if scale_type == "Другое":
            um_per_div, ok = QInputDialog.getDouble(
                self, "Мкм на деление", "Введите мкм/деление", 10.0, 0.001, 10000.0, 3
            )
            if not ok:
                return

        n_div, ok = QInputDialog.getInt(self, "N делений", "Количество делений", 50, 1, 1000)
        if not ok:
            return

        # 3 повтора: пользователь вводит длину в px (или можно измерить инструментом)
        px_distances: list[float] = []
        for i in range(3):
            px, ok = QInputDialog.getDouble(
                self,
                "Повтор измерения",
                f"Введите длину в px для повтора {i + 1} (или по snap)",
                1000.0,
                1.0,
                1_000_000.0,
                3,
            )
            if not ok:
                return
            px_distances.append(px)

        stats = calculate_calibration_stats(px_distances, n_div, um_per_div)
        source_hash = hashlib.sha1(self.original.tobytes()).hexdigest()[:16]
        profile_name, ok = QInputDialog.getText(
            self,
            "Имя профиля",
            "Введите имя профиля",
            text=f"{objective}_{datetime.now():%Y%m%d_%H%M}",
        )
        if not ok or not profile_name.strip():
            return
        profile = CalibrationProfile(
            name=profile_name.strip(),
            objective=objective,
            um_per_px=stats["um_per_px"],
            date=datetime.now().isoformat(),
            source_image_hash=source_hash,
            method="micrometer_manual_or_snap",
            n_repeats=int(stats["n_repeats"]),
            sd=float(stats["sd"]),
        )
        self.calibration_profiles = [p for p in self.calibration_profiles if p.name != profile.name]
        self.calibration_profiles.append(profile)
        self.active_profile_name = profile.name
        self.um_per_px = profile.um_per_px
        self.scale_mode = "micrometer"
        self._save_calibration_profiles_global()
        self._refresh_profile_combo()
        self._update_scale_status()
        self.refresh_measurements_table()
        QMessageBox.information(
            self,
            "Калибровка сохранена",
            f"um/px={profile.um_per_px:.6f}\nSD={profile.sd:.6f}\nПрофиль: {profile.name}",
        )

    def apply_selected_profile(self) -> None:
        name = self.profile_combo.currentText()
        if name == "—":
            self.um_per_px = None
            self.scale_mode = "none"
            self.active_profile_name = None
        else:
            profile = next((p for p in self.calibration_profiles if p.name == name), None)
            if profile is None:
                return
            self.um_per_px = profile.um_per_px
            self.scale_mode = "profile"
            self.active_profile_name = profile.name
        self._update_scale_status()
        self.refresh_measurements_table()

    def on_scale_line_finished(self, px_distance: float) -> None:
        if self.pending_scale_context and self.pending_scale_context.get("mode") == "line":
            value, ok = QInputDialog.getDouble(
                self,
                "Калибровка по линии",
                "Введите реальную длину (мкм)",
                100.0,
                0.0001,
                1_000_000.0,
                4,
            )
            if ok:
                self.um_per_px = value / max(px_distance, 1e-9)
                self.scale_mode = "line"
                self.um_edit.setText(f"{self.um_per_px:.6f}")
                self._update_scale_status()
                self.refresh_measurements_table()
            self.pending_scale_context = None
            self.set_tool("hand")
            return

        real_um, ok = QInputDialog.getDouble(
            self,
            "Масштаб по линии",
            "Введите реальную длину линии (мкм)",
            decimals=3,
            minValue=0.0001,
            value=100.0,
        )
        if ok:
            self.um_per_px = real_um / max(px_distance, 1e-9)
            self.scale_mode = "line"
            self.um_edit.setText(f"{self.um_per_px:.6f}")
            self._update_scale_status()
            self.refresh_measurements_table()

    def _update_scale_status(self) -> None:
        if self.um_per_px is None:
            self.scale_label.setText("Калибровка: не задана")
            return
        prof = self.active_profile_name or "без профиля"
        self.scale_label.setText(f"Калибровка: {self.um_per_px:.6f} мкм/px ({prof})")

    # ---------- enhance/segmentation ----------
    def on_enhance_changed(self) -> None:
        if self.original is None:
            return
        for key, slider in self.enhance_sliders.items():
            val = slider.value()
            setattr(self.enhance_params, key, val)
            self.enhance_value_labels[key].setText(str(val))
        self.enhanced = apply_enhancements(self.original, self.enhance_params)
        self.view.update_enhanced(self.enhanced)

    def reset_enhance_param(self, key: str) -> None:
        self.enhance_sliders[key].setValue(0)

    def reset_enhance(self) -> None:
        for slider in self.enhance_sliders.values():
            slider.setValue(0)
        self.enhance_params = EnhanceParams()
        if self.original is not None:
            self.enhanced = self.original.copy()
            self.view.update_enhanced(self.enhanced)

    def push_undo(self) -> None:
        if self.mask is None:
            return
        self.mask_undo.append(self.mask.copy())
        self.mask_undo = self.mask_undo[-20:]
        self.mask_redo.clear()

    def on_mask_changed(self, new_mask: np.ndarray) -> None:
        if self.mask is not None:
            self.mask_undo.append(self.mask.copy())
            self.mask_undo = self.mask_undo[-20:]
        self.mask = new_mask

    def undo_mask(self) -> None:
        if not self.mask_undo or self.mask is None:
            return
        self.mask_redo.append(self.mask.copy())
        self.mask = self.mask_undo.pop()
        self.view.update_mask(self.mask)

    def redo_mask(self) -> None:
        if not self.mask_redo or self.mask is None:
            return
        self.mask_undo.append(self.mask.copy())
        self.mask = self.mask_redo.pop()
        self.view.update_mask(self.mask)

    def run_threshold(self) -> None:
        if self.enhanced is None:
            return
        self.push_undo()
        self.mask = threshold_segmentation(
            self.enhanced,
            mode=self.threshold_mode.currentText(),
            close_size=self.close_size.value(),
            open_size=self.open_size.value(),
            manual_threshold=self.manual_threshold.value(),
        )
        self.view.update_mask(self.mask)

    def run_cellpose_clicked(self) -> None:
        if self.enhanced is None:
            return
        dia_txt = self.cellpose_diameter.text().strip().lower()
        diameter = None if dia_txt in {"", "auto"} else float(dia_txt)
        try:
            self.push_undo()
            self.mask = run_cellpose(
                self.enhanced,
                model_type=self.cellpose_model.currentText(),
                diameter=diameter,
            )
            self.view.update_mask(self.mask)
        except Exception as exc:
            QMessageBox.warning(self, "Ошибка Cellpose", str(exc))

    # ---------- measurements ----------
    def on_measurement(self, payload: dict) -> None:
        kind = payload["type"]
        points = payload["points"]

        if kind == "line":
            value_px = polyline_length(points[:2])
            self._add_measurement("line", value_px, details=str(points[:2]))
        elif kind == "polyline":
            value_px = polyline_length(points)
            self._add_measurement("polyline", value_px, details=f"n={len(points)}")
        elif kind == "area":
            value_px = polygon_area(points)
            self._add_measurement("area", value_px, details=f"n={len(points)}")
        elif kind == "thickness_1":
            self.thickness_line1 = points
            self.annotations.append({"type": "thickness_1", "points": points})
        elif kind == "thickness_2":
            self.annotations.append({"type": "thickness_2", "points": points})
            if self.thickness_line1 is not None:
                stats = thickness_distribution(self.thickness_line1, points)
                self._add_measurement(
                    "thickness",
                    stats["mean"],
                    details=f"median={stats['median']:.2f}; min={stats['min']:.2f}; max={stats['max']:.2f}",
                )

        self.refresh_measurements_table()

    def _add_measurement(self, kind: str, value_px: float, details: str = "") -> None:
        conv, units = convert_measurement_value(value_px, kind, self.um_per_px)
        self.measurements.append(
            {
                "type": kind,
                "value_px": round(value_px, 6),
                "value_um": None if conv is None else round(float(conv), 6),
                "units": units,
                "details": details,
                "time": datetime.now().strftime("%H:%M:%S"),
            }
        )

    def refresh_measurements_table(self) -> None:
        self.measurements = recalc_measurements_with_scale(self.measurements, self.um_per_px)
        self.results_table.setRowCount(len(self.measurements))
        for i, m in enumerate(self.measurements):
            self.results_table.setItem(i, 0, QTableWidgetItem(str(m.get("type", ""))))
            self.results_table.setItem(i, 1, QTableWidgetItem(str(m.get("value_px", ""))))
            self.results_table.setItem(i, 2, QTableWidgetItem(str(m.get("value_um", "—"))))
            self.results_table.setItem(i, 3, QTableWidgetItem(str(m.get("units", "px"))))
            self.results_table.setItem(i, 4, QTableWidgetItem(str(m.get("details", ""))))
            self.results_table.setItem(i, 5, QTableWidgetItem(str(m.get("time", ""))))
        self._update_summary()

    def _update_summary(self) -> None:
        if not self.measurements:
            self.summary_label.setText("Сводка: нет измерений")
            return
        areas = [m["value_px"] for m in self.measurements if m.get("type") == "area"]
        thickness = [m["value_px"] for m in self.measurements if m.get("type") == "thickness"]
        cells = float(np.mean(self.mask > 0)) if self.mask is not None else 0.0
        text = (
            f"Площадь ткани (сумма area px²): {sum(areas):.2f}; "
            f"Площадь пустот (оценка): {(1.0 - cells) * 100:.1f}% ; "
            f"Клеточность (оценка по маске): {cells * 100:.1f}%"
        )
        if thickness:
            text += (
                f"; Толщина средняя/медиана: {np.mean(thickness):.2f}/{np.median(thickness):.2f} px"
            )
        self.summary_label.setText(text)

    def delete_selected_measurement(self) -> None:
        rows = sorted({idx.row() for idx in self.results_table.selectedIndexes()}, reverse=True)
        for row in rows:
            if 0 <= row < len(self.measurements):
                self.measurements.pop(row)
        self.refresh_measurements_table()

    def clear_measurements(self) -> None:
        self.measurements.clear()
        self.refresh_measurements_table()

    def run_batch_mode(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку для batch")
        if not folder:
            return
        out_csv, _ = QFileDialog.getSaveFileName(
            self, "Сводный CSV batch", "batch_summary.csv", "CSV (*.csv)"
        )
        if not out_csv:
            return
        rows = []
        for p in sorted(Path(folder).glob("*")):
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            try:
                img = np.array(Image.open(p).convert("RGB"))
                qc = self.ai_engine.analyze_quality(img)
                seg = self.ai_engine.preview_stage("segmentation", img, None, {})
                rows.append(
                    {
                        "file": p.name,
                        "sharpness": qc["metrics"].get("sharpness"),
                        "contrast_proxy": qc["metrics"].get("tissue_ratio"),
                        "seg_coverage": seg.metrics.get("coverage"),
                        "seg_confidence": seg.metrics.get("confidence"),
                    }
                )
            except Exception:
                rows.append({"file": p.name, "error": "processing_failed"})
        import csv

        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            fields = sorted({k for r in rows for k in r.keys()})
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        QMessageBox.information(self, "Batch", f"Готово: {out_csv}")

    def export_geojson(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Экспорт GeoJSON", "annotations.geojson", "GeoJSON (*.geojson)"
        )
        if not path:
            return
        export_geojson(path, self.annotations, self.measurements)

    def export_figure_dialog(self) -> None:
        if self.original is None and self.enhanced is None:
            QMessageBox.information(self, "Экспорт фигуры", "Нет данных для экспорта")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Экспорт фигуры для статьи")
        lay = QVBoxLayout(dlg)

        fmt = QComboBox()
        fmt.addItems(["PNG", "TIFF"])
        size_mode = QComboBox()
        size_mode.addItems(["как на экране", "A4 300 dpi", "A4 600 dpi", "пользовательский px"])
        cw = QSpinBox()
        cw.setRange(256, 10000)
        cw.setValue(1920)
        ch = QSpinBox()
        ch.setRange(256, 10000)
        ch.setValue(1080)

        inc_original = QCheckBox("Оригинал")
        inc_original.setChecked(True)
        inc_enhanced = QCheckBox("Улучшенное")
        inc_enhanced.setChecked(True)
        inc_seg = QCheckBox("Сегментация (оверлей)")
        inc_seg.setChecked(True)
        inc_art = QCheckBox("Маска артефактов")
        inc_heat = QCheckBox("Heatmap")
        inc_heat.setEnabled(self.artifact_mask is not None)
        inc_roi = QCheckBox("ROI рамки")
        inc_roi.setChecked(True)
        inc_ann = QCheckBox("Аннотации/измерения")
        inc_ann.setChecked(True)
        alpha = QSlider(Qt.Orientation.Horizontal)
        alpha.setRange(0, 100)
        alpha.setValue(40)

        title_cb = QCheckBox("Заголовок")
        title_cb.setChecked(True)
        title_edit = QLineEdit("HistoAnalyzer — Figure")
        roi_lbl_cb = QCheckBox("Подпись ROI")
        roi_lbl_cb.setChecked(True)
        legend_cb = QCheckBox("Легенда цветов")
        legend_cb.setChecked(True)

        scale_cb = QCheckBox("Показывать масштабную линейку")
        scale_cb.setChecked(True)
        scale_len = QComboBox()
        scale_len.addItems(["50 мкм", "100 мкм", "200 мкм", "500 мкм", "0.5 мм", "1 мм"])
        scale_pos = QComboBox()
        scale_pos.addItems(["левый-низ", "правый-низ"])
        methods_cb = QCheckBox("Вписать параметры Methods")
        methods_cb.setChecked(True)

        form = QFormLayout()
        form.addRow("Формат", fmt)
        form.addRow("Размер", size_mode)
        form.addRow("Ширина px", cw)
        form.addRow("Высота px", ch)
        for w in [inc_original, inc_enhanced, inc_seg, inc_art, inc_heat, inc_roi, inc_ann]:
            form.addRow(w)
        form.addRow("Прозрачность оверлеев", alpha)
        form.addRow(title_cb, title_edit)
        form.addRow(roi_lbl_cb)
        form.addRow(legend_cb)
        form.addRow(scale_cb)
        form.addRow("Длина линейки", scale_len)
        form.addRow("Позиция линейки", scale_pos)
        form.addRow(methods_cb)
        lay.addLayout(form)

        btn_row = QHBoxLayout()
        ok_btn = QPushButton("Экспорт")
        cancel_btn = QPushButton("Отмена")
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)
        lay.addLayout(btn_row)
        cancel_btn.clicked.connect(dlg.reject)

        def _do_export():
            mode_map = {
                "как на экране": "screen",
                "A4 300 dpi": "a4_300",
                "A4 600 dpi": "a4_600",
                "пользовательский px": "custom",
            }
            settings = FigureExportSettings(
                fmt=fmt.currentText(),
                size_mode=mode_map[size_mode.currentText()],
                custom_width=cw.value(),
                custom_height=ch.value(),
                include_original=inc_original.isChecked(),
                include_enhanced=inc_enhanced.isChecked(),
                include_segmentation=inc_seg.isChecked(),
                include_artifacts=inc_art.isChecked(),
                include_heatmap=inc_heat.isChecked(),
                include_roi=inc_roi.isChecked(),
                include_annotations=inc_ann.isChecked(),
                overlay_alpha=alpha.value(),
                show_title=title_cb.isChecked(),
                title_text=title_edit.text().strip() or "HistoAnalyzer — Figure",
                show_roi_labels=roi_lbl_cb.isChecked(),
                show_legend=legend_cb.isChecked(),
                show_scale_bar=scale_cb.isChecked(),
                scale_length=scale_len.currentText(),
                scale_pos=scale_pos.currentText(),
                include_methods=methods_cb.isChecked(),
            )
            self.settings.setValue("figure_export_settings", settings.to_json())
            ext = "png" if settings.fmt.upper() == "PNG" else "tiff"
            path, _ = QFileDialog.getSaveFileName(
                self, "Экспорт фигуры", f"figure.{ext}", f"{settings.fmt} (*.{ext})"
            )
            if not path:
                return
            methods_text = f"normalization={self.guided_ai_state.get('steps', {}).get('2', {}).get('reco', {})}; seg={self.guided_ai_state.get('steps', {}).get('7', {}).get('metrics', {})}; um_per_px={self.um_per_px}"
            scale_drawn = export_figure(
                path=path,
                settings=settings,
                original=self.original,
                enhanced=self.enhanced,
                segmentation=self.mask,
                artifacts=self.artifact_mask,
                heatmap=None,
                roi_mask=self.roi_mask,
                annotations=self.annotations,
                um_per_px=self.um_per_px,
                methods_text=methods_text,
                screen_size=(
                    max(1, self.view.viewport().width()),
                    max(1, self.view.viewport().height()),
                ),
            )
            if settings.show_scale_bar and not scale_drawn:
                QMessageBox.information(
                    self, "Экспорт фигуры", "Масштаб не задан — линейка недоступна"
                )
            dlg.accept()

        ok_btn.clicked.connect(_do_export)

        saved = self.settings.value("figure_export_settings", "", type=str)
        if saved:
            try:
                st = FigureExportSettings.from_json(saved)
                fmt.setCurrentText(st.fmt)
                rev = {
                    "screen": "как на экране",
                    "a4_300": "A4 300 dpi",
                    "a4_600": "A4 600 dpi",
                    "custom": "пользовательский px",
                }
                size_mode.setCurrentText(rev.get(st.size_mode, "как на экране"))
                cw.setValue(st.custom_width)
                ch.setValue(st.custom_height)
                inc_original.setChecked(st.include_original)
                inc_enhanced.setChecked(st.include_enhanced)
                inc_seg.setChecked(st.include_segmentation)
                inc_art.setChecked(st.include_artifacts)
                inc_heat.setChecked(st.include_heatmap)
                inc_roi.setChecked(st.include_roi)
                inc_ann.setChecked(st.include_annotations)
                alpha.setValue(st.overlay_alpha)
                title_cb.setChecked(st.show_title)
                title_edit.setText(st.title_text)
                roi_lbl_cb.setChecked(st.show_roi_labels)
                legend_cb.setChecked(st.show_legend)
                scale_cb.setChecked(st.show_scale_bar)
                scale_len.setCurrentText(st.scale_length)
                scale_pos.setCurrentText(st.scale_pos)
                methods_cb.setChecked(st.include_methods)
            except Exception:
                pass

        dlg.exec()

    # ---------- export ----------
    def export_csv(self) -> None:
        if not self.measurements:
            QMessageBox.information(self, "Экспорт CSV", "Нет измерений")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Экспорт CSV", "measurements.csv", "CSV (*.csv)"
        )
        if path:
            export_measurements_csv(path, self.measurements)

    def export_pdf(self) -> None:
        if self.original is None or self.enhanced is None or self.mask is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Экспорт PDF", "report.pdf", "PDF (*.pdf)")
        if not path:
            return
        active = next(
            (p for p in self.calibration_profiles if p.name == self.active_profile_name), None
        )
        cal_info = {
            "method": self.scale_mode,
            "um_per_px": self.um_per_px,
            "profile": self.active_profile_name,
            "date": None if active is None else active.date,
            "sd": None if active is None else active.sd,
        }
        export_pdf_report(
            path,
            self.original,
            self.enhanced,
            self.mask,
            self.measurements,
            calibration_info=cal_info,
            guided_ai=self.guided_ai_state,
        )

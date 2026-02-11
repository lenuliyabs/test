from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
    QFileDialog,
)

from app.core.image_ops import EnhanceParams, apply_enhancements
from app.core.mask_ops import threshold_segmentation
from app.core.morphometry import polygon_area, polyline_length, thickness_distribution, to_um
from app.core.project_io import load_project, save_project
from app.export.csv_export import export_measurements_csv
from app.export.pdf_report import export_pdf_report
from app.modules.segmentation.cellpose_runner import CELLPPOSE_AVAILABLE, run_cellpose
from app.viewer.image_view import ImageView


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("HistoAnalyzer")
        self.resize(1400, 900)

        self.image_path: str | None = None
        self.original: np.ndarray | None = None
        self.enhanced: np.ndarray | None = None
        self.mask: np.ndarray | None = None

        self.enhance_params = EnhanceParams()
        self.measurements: list[dict] = []
        self.annotations: list[dict] = []
        self.um_per_px: float | None = None

        self.mask_undo: list[np.ndarray] = []
        self.mask_redo: list[np.ndarray] = []

        self.thickness_line1: list[tuple[float, float]] | None = None

        self._build_ui()
        self._build_menu()

    def _build_menu(self) -> None:
        menu = self.menuBar()
        file_menu = menu.addMenu("File")
        file_menu.addAction("Open", self.open_image)
        file_menu.addAction("Save Project", self.save_project)
        file_menu.addAction("Load Project", self.load_project)
        file_menu.addAction("Export CSV", self.export_csv)
        file_menu.addAction("Export PDF", self.export_pdf)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)

        help_menu = menu.addMenu("Help")
        help_menu.addAction("About", self.show_about)

    def _build_ui(self) -> None:
        central = QWidget()
        root = QHBoxLayout(central)

        toolbar = QToolBar("Tools")
        toolbar.setOrientation(Qt.Orientation.Vertical)
        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, toolbar)

        tool_actions = {
            "Selection/Hand": "hand",
            "Measure line": "line",
            "Measure polyline": "polyline",
            "Measure area": "area",
            "Set scale by line": "scale_line",
            "Thickness line 1": "thickness_1",
            "Thickness line 2": "thickness_2",
            "Brush": "brush",
            "Eraser": "eraser",
        }
        for label, key in tool_actions.items():
            action = toolbar.addAction(label)
            action.triggered.connect(lambda checked=False, k=key: self.view.set_tool(k))

        undo_action = toolbar.addAction("Undo")
        undo_action.triggered.connect(self.undo_mask)
        redo_action = toolbar.addAction("Redo")
        redo_action.triggered.connect(self.redo_mask)

        self.view = ImageView()
        self.view.measurement_finished.connect(self.on_measurement)
        self.view.scale_line_finished.connect(self.on_scale_line_finished)
        self.view.mask_changed.connect(self.on_mask_changed)
        root.addWidget(self.view, 1)

        right_panel = QTabWidget()
        right_panel.addTab(self._enhance_tab(), "Enhance")
        right_panel.addTab(self._segmentation_tab(), "Segmentation")
        right_panel.addTab(self._morphometry_tab(), "Morphometry")
        right_panel.setMaximumWidth(380)
        root.addWidget(right_panel)

        self.setCentralWidget(central)

    def _enhance_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.enhance_sliders: dict[str, QSlider] = {}
        labels = [
            ("brightness", "Exposure/Brightness"),
            ("contrast", "Contrast"),
            ("highlights", "Highlights"),
            ("shadows", "Shadows"),
            ("saturation", "Saturation"),
            ("warmth", "Warmth"),
            ("sharpness", "Sharpness"),
            ("noise_reduction", "Noise reduction"),
        ]
        for key, text in labels:
            lbl = QLabel(text)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 100) if key == "noise_reduction" else slider.setRange(-100, 100)
            slider.setValue(0)
            slider.valueChanged.connect(self.on_enhance_changed)
            self.enhance_sliders[key] = slider
            layout.addWidget(lbl)
            layout.addWidget(slider)
        btn_reset = QPushButton("Reset")
        btn_reset.clicked.connect(self.reset_enhance)
        fit_btn = QPushButton("Fit to window")
        fit_btn.clicked.connect(self.view.fit_image)
        layout.addWidget(btn_reset)
        layout.addWidget(fit_btn)
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

        form.addRow("Threshold mode", self.threshold_mode)
        form.addRow("Manual threshold", self.manual_threshold)
        form.addRow("Close size", self.close_size)
        form.addRow("Open size", self.open_size)
        form.addRow("Mask opacity", self.mask_alpha)
        layout.addLayout(form)

        run_thresh = QPushButton("Run threshold segmentation")
        run_thresh.clicked.connect(self.run_threshold)
        layout.addWidget(run_thresh)

        self.cellpose_model = QComboBox()
        self.cellpose_model.addItems(["cyto", "nuclei"])
        self.cellpose_diameter = QLineEdit("auto")
        form2 = QFormLayout()
        form2.addRow("Cellpose model", self.cellpose_model)
        form2.addRow("Diameter", self.cellpose_diameter)
        layout.addLayout(form2)

        self.cellpose_btn = QPushButton("Run Cellpose")
        self.cellpose_btn.clicked.connect(self.run_cellpose_clicked)
        if not CELLPPOSE_AVAILABLE:
            self.cellpose_btn.setEnabled(False)
            self.cellpose_btn.setToolTip("Установите cellpose")
        layout.addWidget(self.cellpose_btn)

        self.brush_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_slider.setRange(1, 80)
        self.brush_slider.setValue(10)
        self.brush_slider.valueChanged.connect(lambda v: setattr(self.view, "brush_size", v))
        layout.addWidget(QLabel("Brush size"))
        layout.addWidget(self.brush_slider)
        layout.addStretch()
        return tab

    def _morphometry_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.um_edit = QLineEdit()
        self.um_edit.setPlaceholderText("µm per pixel")
        set_scale_btn = QPushButton("Apply µm/px")
        set_scale_btn.clicked.connect(self.apply_um_per_px)
        thickness_btn = QPushButton("Compute thickness (line1 vs line2)")
        thickness_btn.clicked.connect(self.compute_thickness)
        layout.addWidget(self.um_edit)
        layout.addWidget(set_scale_btn)
        layout.addWidget(QLabel("Use tool 'Set scale by line' in toolbar."))
        layout.addWidget(QLabel("Use Thickness line 1 / line 2 then click compute."))
        layout.addWidget(thickness_btn)
        self.results_label = QLabel("Measurements: 0")
        layout.addWidget(self.results_label)
        layout.addStretch()
        return tab

    def show_about(self) -> None:
        QMessageBox.information(self, "About", "HistoAnalyzer MVP\nPySide6 desktop tool")

    def open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open image", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff)"
        )
        if not path:
            return
        self.load_image(path)

    def load_image(self, path: str) -> None:
        arr = np.array(Image.open(path).convert("RGB"))
        self.image_path = path
        self.original = arr
        self.enhanced = arr.copy()
        self.mask = np.zeros(arr.shape[:2], dtype=np.uint8)
        self.measurements.clear()
        self.annotations.clear()
        self.mask_undo.clear()
        self.mask_redo.clear()
        self.view.set_images(self.original, self.enhanced, self.mask)
        self.view.fit_image()

    def on_enhance_changed(self) -> None:
        if self.original is None:
            return
        for key, slider in self.enhance_sliders.items():
            setattr(self.enhance_params, key, slider.value())
        self.enhanced = apply_enhancements(self.original, self.enhance_params)
        self.view.update_enhanced(self.enhanced)

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
            QMessageBox.warning(self, "Cellpose error", str(exc))

    def apply_um_per_px(self) -> None:
        txt = self.um_edit.text().strip()
        if not txt:
            return
        self.um_per_px = float(txt)
        self.view.set_um_per_px(self.um_per_px)

    def on_scale_line_finished(self, px_distance: float) -> None:
        real_um, ok = QInputDialog.getDouble(
            self,
            "Scale by line",
            "Введите реальную длину линии (µm):",
            decimals=3,
            minValue=0.0001,
            value=100.0,
        )
        if not ok:
            return
        self.um_per_px = real_um / max(px_distance, 1e-9)
        self.um_edit.setText(f"{self.um_per_px:.6f}")
        self.view.set_um_per_px(self.um_per_px)

    def on_measurement(self, payload: dict) -> None:
        kind = payload["type"]
        points = payload["points"]

        if kind == "line":
            value_px = polyline_length(points[:2])
            self._add_measurement("line", value_px, points=str(points[:2]))
        elif kind == "polyline":
            value_px = polyline_length(points)
            self._add_measurement("polyline", value_px, points=f"n={len(points)}")
        elif kind == "area":
            value_px = polygon_area(points)
            value_um = None if self.um_per_px is None else value_px * (self.um_per_px**2)
            self.measurements.append(
                {
                    "type": "area",
                    "value_px": round(value_px, 3),
                    "value_um": None if value_um is None else round(value_um, 3),
                    "details": f"n={len(points)}",
                }
            )
        elif kind == "thickness_1":
            self.thickness_line1 = points
            self.annotations.append({"type": "thickness_1", "points": points})
        elif kind == "thickness_2":
            self.annotations.append({"type": "thickness_2", "points": points})
            if self.thickness_line1 is not None:
                stats = thickness_distribution(self.thickness_line1, points)
                self.measurements.append(
                    {
                        "type": "thickness_mean",
                        "value_px": round(stats["mean"], 3),
                        "value_um": (
                            None
                            if self.um_per_px is None
                            else round(stats["mean"] * self.um_per_px, 3)
                        ),
                        "details": (
                            f"median={stats['median']:.2f}; min={stats['min']:.2f}; max={stats['max']:.2f}"
                        ),
                    }
                )
        self.results_label.setText(f"Measurements: {len(self.measurements)}")

    def compute_thickness(self) -> None:
        l1 = next((a["points"] for a in self.annotations if a["type"] == "thickness_1"), None)
        l2 = next(
            (a["points"] for a in reversed(self.annotations) if a["type"] == "thickness_2"), None
        )
        if l1 is None or l2 is None:
            QMessageBox.warning(
                self, "Thickness", "Нарисуйте две полилинии thickness_1 и thickness_2"
            )
            return
        stats = thickness_distribution(l1, l2)
        self.measurements.append(
            {
                "type": "thickness_mean",
                "value_px": round(stats["mean"], 3),
                "value_um": (
                    None if self.um_per_px is None else round(stats["mean"] * self.um_per_px, 3)
                ),
                "details": f"median={stats['median']:.2f}; min={stats['min']:.2f}; max={stats['max']:.2f}",
            }
        )
        self.results_label.setText(f"Measurements: {len(self.measurements)}")

    def _add_measurement(self, kind: str, value_px: float, points: str = "") -> None:
        self.measurements.append(
            {
                "type": kind,
                "value_px": round(value_px, 3),
                "value_um": (
                    None if self.um_per_px is None else round(to_um(value_px, self.um_per_px), 3)
                ),
                "details": points,
            }
        )

    def save_project(self) -> None:
        if self.image_path is None or self.mask is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save project", "", "Project (*.json)")
        if not path:
            return
        save_project(
            path,
            self.image_path,
            self.enhance_params,
            self.mask,
            self.annotations,
            self.measurements,
            self.um_per_px,
        )

    def load_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load project", "", "Project (*.json)")
        if not path:
            return
        payload, mask = load_project(path)
        image_path = payload["image_path"]
        if not Path(image_path).exists():
            QMessageBox.warning(self, "Load project", f"Source image not found: {image_path}")
            return
        self.load_image(image_path)
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
        if self.um_per_px is not None:
            self.um_edit.setText(f"{self.um_per_px:.6f}")

    def export_csv(self) -> None:
        if not self.measurements:
            QMessageBox.information(self, "Export CSV", "No measurements")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", "measurements.csv", "CSV (*.csv)")
        if not path:
            return
        export_measurements_csv(path, self.measurements)

    def export_pdf(self) -> None:
        if self.original is None or self.enhanced is None or self.mask is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export PDF", "report.pdf", "PDF (*.pdf)")
        if not path:
            return
        export_pdf_report(path, self.original, self.enhanced, self.mask, self.measurements)

"""Glassmorphism desktop app for face-mask detection (PySide6).

Run:  python app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal, QRectF, QPoint
from PySide6.QtGui import (
    QColor,
    QFont,
    QImage,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QRadialGradient,
)
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGraphicsBlurEffect,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

HERE = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = HERE / "runs" / "detect" / "train-7" / "weights" / "best.pt"

CLASS_COLORS = {
    "with_mask": QColor(52, 211, 153),
    "without_mask": QColor(248, 113, 113),
    "mask_weared_incorrect": QColor(251, 191, 36),
}
FALLBACK_COLOR = QColor(148, 163, 184)

CLASSES = ("with_mask", "without_mask", "mask_weared_incorrect")

ACCENT = "#7dd3fc"
TEXT = "#e8eefc"
MUTED = "#9fb0cc"

STYLESHEET = f"""
QWidget {{
    color: {TEXT};
    font-family: 'Segoe UI', 'Inter', sans-serif;
    font-size: 14px;
}}
QLabel#title {{
    font-size: 26px;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: 0.5px;
}}
QLabel#subtitle {{
    font-size: 13px;
    color: {MUTED};
}}
QLabel#cardTitle {{
    font-size: 12px;
    font-weight: 600;
    color: {MUTED};
    letter-spacing: 1.4px;
}}
QFrame#glass {{
    background-color: rgba(255, 255, 255, 26);
    border: 1px solid rgba(255, 255, 255, 46);
    border-radius: 20px;
}}
QFrame#glassStrong {{
    background-color: rgba(255, 255, 255, 38);
    border: 1px solid rgba(255, 255, 255, 64);
    border-radius: 22px;
}}
QPushButton {{
    background-color: rgba(255, 255, 255, 30);
    border: 1px solid rgba(255, 255, 255, 58);
    border-radius: 13px;
    padding: 11px 20px;
    font-weight: 600;
    color: {TEXT};
}}
QPushButton:hover {{
    background-color: rgba(255, 255, 255, 52);
    border: 1px solid rgba(255, 255, 255, 92);
}}
QPushButton:pressed {{
    background-color: rgba(255, 255, 255, 20);
}}
QPushButton:disabled {{
    background-color: rgba(255, 255, 255, 12);
    border: 1px solid rgba(255, 255, 255, 24);
    color: rgba(232, 238, 252, 90);
}}
QPushButton#primary {{
    background-color: rgba(125, 211, 252, 78);
    border: 1px solid rgba(125, 211, 252, 140);
    color: #f2faff;
}}
QPushButton#primary:hover {{
    background-color: rgba(125, 211, 252, 116);
}}
QSlider::groove:horizontal {{
    height: 5px;
    background: rgba(255, 255, 255, 40);
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: {ACCENT};
    width: 15px;
    margin: -6px 0;
    border-radius: 7px;
}}
QSlider::sub-page:horizontal {{
    background: rgba(125, 211, 252, 130);
    border-radius: 3px;
}}
"""


class Backdrop(QWidget):
    """Paints the gradient + soft colour blobs that the glass panels blur."""

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        rect = self.rect()

        base = QLinearGradient(0, 0, rect.width(), rect.height())
        base.setColorAt(0.0, QColor(15, 23, 42))
        base.setColorAt(0.45, QColor(30, 41, 82))
        base.setColorAt(1.0, QColor(11, 18, 38))
        p.fillRect(rect, base)

        blobs = [
            (0.16, 0.20, 0.55, QColor(56, 189, 248, 108)),
            (0.84, 0.16, 0.48, QColor(167, 139, 250, 104)),
            (0.72, 0.86, 0.62, QColor(45, 212, 191, 82)),
            (0.22, 0.88, 0.46, QColor(244, 114, 182, 74)),
        ]
        for fx, fy, fr, color in blobs:
            cx, cy = rect.width() * fx, rect.height() * fy
            radius = max(rect.width(), rect.height()) * fr
            grad = QRadialGradient(QPoint(int(cx), int(cy)), radius)
            grad.setColorAt(0.0, color)
            transparent = QColor(color)
            transparent.setAlpha(0)
            grad.setColorAt(1.0, transparent)
            p.setBrush(grad)
            p.setPen(Qt.NoPen)
            p.drawEllipse(QRectF(cx - radius, cy - radius, radius * 2, radius * 2))


class GlassFrame(QFrame):
    """A frosted panel: drop shadow + subtle inner highlight."""

    def __init__(self, strong=False, parent=None):
        super().__init__(parent)
        self.setObjectName("glassStrong" if strong else "glass")

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(38)
        shadow.setOffset(0, 10)
        shadow.setColor(QColor(0, 0, 0, 130))
        self.setGraphicsEffect(shadow)


def _upscale_for_plot(result, target: int = 1280):
    """Return a result whose image is at least `target` px on the long edge.

    Labels are rasterised from the image the result carries, so tiny sources
    produce unreadable text. Plot from an upscaled copy with rescaled boxes.
    """
    import cv2

    img = result.orig_img
    if img is None:
        return result

    height, width = img.shape[:2]
    longest = max(height, width)
    if longest >= target:
        return result

    scale = target / float(longest)
    new_size = (max(int(round(width * scale)), 1), max(int(round(height * scale)), 1))
    big = cv2.resize(img, new_size, interpolation=cv2.INTER_CUBIC)

    from ultralytics.engine.results import Results

    boxes_tensor = None
    if result.boxes is not None and len(result.boxes):
        data = result.boxes.data.clone().float()
        data[:, [0, 2]] *= scale
        data[:, [1, 3]] *= scale
        boxes_tensor = data

    return Results(
        orig_img=big,
        path=str(result.path),
        names=result.names,
        boxes=boxes_tensor,
    )


def blur_pixmap(pixmap: QPixmap, radius: float = 34.0) -> QPixmap:
    """True gaussian blur of a pixmap via QGraphicsBlurEffect."""
    from PySide6.QtWidgets import QGraphicsScene

    scene = QGraphicsScene()
    item = scene.addPixmap(pixmap)
    effect = QGraphicsBlurEffect()
    effect.setBlurRadius(radius)
    item.setGraphicsEffect(effect)

    out = QImage(pixmap.size(), QImage.Format_ARGB32_Premultiplied)
    out.fill(Qt.transparent)
    painter = QPainter(out)
    painter.setRenderHint(QPainter.Antialiasing)
    scene.render(painter)
    painter.end()
    return QPixmap.fromImage(out)


class DetectionWorker(QThread):
    """Runs inference off the UI thread."""

    finished_ok = Signal(object, object)
    failed = Signal(str)

    def __init__(self, weights: Path, source: Path, conf: float, parent=None):
        super().__init__(parent)
        self.weights = weights
        self.source = source
        self.conf = conf

    def run(self):
        try:
            from ultralytics import YOLO

            model = YOLO(str(self.weights))
            results = model.predict(source=str(self.source), conf=self.conf,
                                    iou=0.45, verbose=False)
            self.finished_ok.emit(results, None)
        except Exception as exc:  # surfaced in the UI instead of dying silently
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class ImageCanvas(QLabel):
    """Shows the image, or a placeholder, scaled to fit."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(520, 420)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._pixmap: QPixmap | None = None
        self._placeholder = "Choose an image to begin"

    def set_placeholder(self, text: str):
        self._placeholder = text
        self._pixmap = None
        self.update()

    def set_image(self, image: QImage | QPixmap):
        self._pixmap = QPixmap.fromImage(image) if isinstance(image, QImage) else image
        self.update()

    def current_pixmap(self) -> QPixmap | None:
        return self._pixmap

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)

        rect = self.rect()
        path = QPainterPath()
        path.addRoundedRect(QRectF(rect).adjusted(1, 1, -1, -1), 18, 18)
        p.setClipPath(path)
        p.fillRect(rect, QColor(255, 255, 255, 14))

        if self._pixmap and not self._pixmap.isNull():
            scaled = self._pixmap.scaled(rect.size(), Qt.KeepAspectRatio,
                                         Qt.SmoothTransformation)
            x = (rect.width() - scaled.width()) // 2
            y = (rect.height() - scaled.height()) // 2
            p.drawPixmap(x, y, scaled)
        else:
            p.setPen(QPen(QColor(159, 176, 204, 190)))
            font = QFont(self.font())
            font.setPointSize(12)
            p.setFont(font)
            p.drawText(rect, Qt.AlignCenter, self._placeholder)

        p.setClipping(False)
        p.setPen(QPen(QColor(255, 255, 255, 60), 1.4))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(QRectF(rect).adjusted(1, 1, -1, -1), 18, 18)


class MaskApp(QMainWindow):
    def __init__(self, weights: Path):
        super().__init__()
        self.weights = weights
        self.source_path: Path | None = None
        self.worker: DetectionWorker | None = None
        self.class_counts = {name: 0 for name in CLASSES}
        self.total_detections = 0

        self.setWindowTitle("Mask Detection - Glass UI")
        self.resize(1240, 800)
        self.setMinimumSize(1040, 700)

        root = Backdrop()
        self.setCentralWidget(root)

        outer = QVBoxLayout(root)
        outer.setContentsMargins(26, 22, 26, 22)
        outer.setSpacing(18)

        outer.addLayout(self._build_header())

        body = QHBoxLayout()
        body.setSpacing(18)
        body.addWidget(self._build_left_column(), 5)
        body.addWidget(self._build_right_column(), 4)
        outer.addLayout(body, 1)

        self._update_weights_label()
        self._refresh_stats()

    # ------------------------------------------------------------------ layout
    def _build_header(self):
        header = QHBoxLayout()

        titles = QVBoxLayout()
        titles.setSpacing(2)
        title = QLabel("Face Mask Detection")
        title.setObjectName("title")
        subtitle = QLabel("YOLO26n \u00b7 glassmorphism interface")
        subtitle.setObjectName("subtitle")
        titles.addWidget(title)
        titles.addWidget(subtitle)
        header.addLayout(titles)
        header.addStretch(1)

        self.weights_label = QLabel()
        self.weights_label.setObjectName("subtitle")
        self.weights_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        header.addWidget(self.weights_label)
        return header

    def _build_left_column(self):
        card = GlassFrame(strong=True)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        heading = QLabel("PREVIEW")
        heading.setObjectName("cardTitle")
        layout.addWidget(heading)

        self.canvas = ImageCanvas()
        layout.addWidget(self.canvas, 1)

        buttons = QHBoxLayout()
        buttons.setSpacing(10)

        self.open_btn = QPushButton("Open image")
        self.open_btn.clicked.connect(self.open_image)
        buttons.addWidget(self.open_btn, 1)

        self.detect_btn = QPushButton("Detect")
        self.detect_btn.setObjectName("primary")
        self.detect_btn.setEnabled(False)
        self.detect_btn.clicked.connect(self.run_detection)
        buttons.addWidget(self.detect_btn, 1)

        self.save_btn = QPushButton("Save result")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_result)
        buttons.addWidget(self.save_btn, 1)

        layout.addLayout(buttons)
        return card

    def _build_right_column(self):
        card = GlassFrame()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        conf_heading = QLabel("CONFIDENCE THRESHOLD")
        conf_heading.setObjectName("cardTitle")
        layout.addWidget(conf_heading)

        conf_row = QHBoxLayout()
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(5, 95)
        self.conf_slider.setValue(25)
        self.conf_slider.valueChanged.connect(self._on_conf_changed)
        conf_row.addWidget(self.conf_slider, 1)

        self.conf_value = QLabel("0.25")
        self.conf_value.setMinimumWidth(46)
        self.conf_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        conf_row.addWidget(self.conf_value)
        layout.addLayout(conf_row)

        stats_heading = QLabel("DETECTIONS")
        stats_heading.setObjectName("cardTitle")
        layout.addWidget(stats_heading)

        self.stat_labels = {}
        for name in CLASSES:
            row = QHBoxLayout()
            dot = QLabel("\u25cf")
            dot.setStyleSheet(f"color: {CLASS_COLORS[name].name()}; font-size: 15px;")
            row.addWidget(dot)
            label = QLabel(name.replace("_", " "))
            row.addWidget(label, 1)
            count = QLabel("0")
            count.setStyleSheet("font-weight: 700;")
            row.addWidget(count)
            self.stat_labels[name] = count
            layout.addLayout(row)

        self.total_label = QLabel("Total: 0")
        self.total_label.setStyleSheet(f"color: {MUTED};")
        layout.addWidget(self.total_label)

        status_heading = QLabel("STATUS")
        status_heading.setObjectName("cardTitle")
        layout.addWidget(status_heading)

        self.status = QLabel("Load a model and pick an image.")
        self.status.setWordWrap(True)
        self.status.setStyleSheet(f"color: {MUTED};")
        layout.addWidget(self.status)

        layout.addStretch(1)
        return card

    # ----------------------------------------------------------------- actions
    def _update_weights_label(self):
        if self.weights.is_file():
            mb = self.weights.stat().st_size / 1e6
            self.weights_label.setText(f"weights: {self.weights.name} ({mb:.1f} MB)")
            self.weights_label.setStyleSheet(f"color: {MUTED};")
        else:
            self.weights_label.setText("weights: NOT FOUND")
            self.weights_label.setStyleSheet("color: #f87171;")

    def _on_conf_changed(self, value):
        self.conf_value.setText(f"{value / 100:.2f}")

    def _refresh_stats(self):
        for name in CLASSES:
            self.stat_labels[name].setText(str(self.class_counts[name]))
        self.total_label.setText(f"Total: {self.total_detections}")

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open image", str(HERE),
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        if not path:
            return

        self.source_path = Path(path)
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self.status.setText("Could not read that image file.")
            return

        self.canvas.set_image(pixmap)
        self.detect_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        self.class_counts = {name: 0 for name in CLASSES}
        self.total_detections = 0
        self._refresh_stats()
        self.status.setText(f"Loaded {self.source_path.name}. Press Detect.")

    def run_detection(self):
        if self.source_path is None:
            return
        if not self.weights.is_file():
            self.status.setText(
                f"Model weights not found at:\n{self.weights}\n\n"
                "Train a model or set DEFAULT_WEIGHTS in app.py.")
            return

        self.detect_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.status.setText("Running inference...")

        self.worker = DetectionWorker(
            self.weights, self.source_path, self.conf_slider.value() / 100.0, self)
        self.worker.finished_ok.connect(self._on_results)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_failed(self, message):
        self.status.setText(f"Inference failed:\n{message}")
        self.detect_btn.setEnabled(True)

    def _on_results(self, results, _unused):
        self.detect_btn.setEnabled(True)
        if not results:
            self.status.setText("No results returned.")
            return

        result = results[0]
        try:
            # Plot from an upscaled copy: labels are rasterised from the image
            # this dataset's small sources would render as unreadable mush.
            plotted_result = _upscale_for_plot(result)
            plotted = plotted_result.plot(line_width=2, labels=True, conf=True,
                                          font_size=0.8)
        except Exception as exc:
            self.status.setText(f"Could not render result: {exc}")
            return

        plotted = plotted[:, :, ::-1].copy()  # BGR -> RGB
        height, width, channels = plotted.shape
        image = QImage(plotted.data, width, height, channels * width,
                       QImage.Format_RGB888).copy()
        self.canvas.set_image(image)

        self.class_counts = {name: 0 for name in CLASSES}
        self.total_detections = 0
        names = result.names or {}
        if result.boxes is not None and result.boxes.cls is not None:
            for cls_id in result.boxes.cls.tolist():
                label = names.get(int(cls_id), str(int(cls_id)))
                if label in self.class_counts:
                    self.class_counts[label] += 1
                self.total_detections += 1

        self._refresh_stats()
        self.save_btn.setEnabled(True)

        if self.total_detections == 0:
            self.status.setText("No faces detected above the threshold.")
        else:
            summary = ", ".join(
                f"{n.replace('_', ' ')}: {c}"
                for n, c in self.class_counts.items() if c)
            self.status.setText(f"Found {self.total_detections} - {summary}")

    def save_result(self):
        pixmap = self.canvas.current_pixmap()
        if pixmap is None:
            return
        suggested = (f"{self.source_path.stem}_pred.jpg"
                     if self.source_path else "prediction.jpg")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save result", str(HERE / suggested), "Images (*.png *.jpg)")
        if not path:
            return
        if pixmap.save(path):
            self.status.setText(f"Saved to {path}")
        else:
            self.status.setText("Failed to save the image.")


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)

    weights = DEFAULT_WEIGHTS
    if len(sys.argv) > 1:
        weights = Path(sys.argv[1])

    window = MaskApp(weights)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

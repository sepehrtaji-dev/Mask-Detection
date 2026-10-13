from __future__ import annotations

import os
from pathlib import Path

from ultralytics import YOLO


class MaskModel:
    def __init__(self, model: str | None = None, checkpoint_path: str | None = None,
                 device: str | None = None):
        if model is None and checkpoint_path is None:
            model = "yolo26n.pt"
        if model is not None and checkpoint_path is not None:
            raise ValueError("pass either model or checkpoint_path, not both")

        self.checkpoint_path = checkpoint_path
        self.model_name = model
        self.device = device
        self.model: YOLO | None = None

    def _load(self) -> YOLO:
        source = self.checkpoint_path or self.model_name
        if source is None:
            raise ValueError("no model or checkpoint configured")
        if not os.path.isfile(source):
            raise FileNotFoundError(f"weights not found: {source}")
        if self.model is None:
            self.model = YOLO(source)
        return self.model

    def train(self, data_path, epochs=100, imgsize=512, batch=16, device=None,
              project=None, name=None, patience=50, **kwargs):
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"data yaml not found: {data_path}")

        # imgsz must be a multiple of the max stride (32) or Ultralytics silently
        # rewrites it; round here so the value in the log is the value requested.
        if imgsize % 32 != 0:
            imgsize = (imgsize // 32 + 1) * 32

        model = self._load()
        results = model.train(
            data=data_path,
            epochs=epochs,
            imgsz=imgsize,
            batch=batch,
            device=device if device is not None else self.device,
            patience=patience,
            project=project,
            name=name,
            exist_ok=True,
            **kwargs,
        )
        return results

    def val(self, checkpoint_path=None, data_path=None, split="val", device=None):
        path = checkpoint_path or self.checkpoint_path
        if path is None:
            raise ValueError("no checkpoint to validate")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"checkpoint not found: {path}")

        model = YOLO(path)
        metrics = model.val(data=data_path, split=split,
                            device=device if device is not None else self.device)

        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"mAP50   : {metrics.box.map50:.4f}")
        print(f"mAP75   : {metrics.box.map75:.4f}")
        print(f"Precision: {metrics.box.mp:.4f}")
        print(f"Recall   : {metrics.box.mr:.4f}")
        # fitness is a property, not a method
        print(f"Fitness  : {metrics.box.fitness:.4f}")
        return metrics

    def predict(self, source, conf=0.25, iou=0.45, device=None, **kwargs):
        model = self._load()
        return model.predict(
            source=source,
            conf=conf,
            iou=iou,
            device=device if device is not None else self.device,
            **kwargs,
        )

    def export(self, checkpoint_path=None, fmt="onnx", imgsz=512, **kwargs):
        """Export a checkpoint. Prefers the given path, else the trained model."""
        path = checkpoint_path or self.checkpoint_path
        if path is None:
            raise ValueError("no checkpoint to export")

        if not os.path.isfile(path):
            raise FileNotFoundError(f"checkpoint not found: {path}")

        model = YOLO(path)
        out = model.export(format=fmt, imgsz=imgsz, **kwargs)

        out_path = Path(out) if out else None
        if out_path is None or not out_path.exists():
            raise RuntimeError(f"export reported success but no file at {out_path}")
        print(f"exported: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
        return out_path

    def export_best_of(self, run_dir, fmt="onnx", imgsz=512, **kwargs):
        """Export <run_dir>/weights/best.pt if present, else last.pt."""
        weights = Path(run_dir) / "weights"
        for candidate in ("best.pt", "last.pt"):
            path = weights / candidate
            if path.is_file():
                return self.export(str(path), fmt=fmt, imgsz=imgsz, **kwargs)
        raise FileNotFoundError(f"no weights under {weights}")

# Face Mask Detection

YOLO26n detector for face-mask detection with three classes, plus a PySide6
desktop app with a glassmorphism interface.

Trained on the Kaggle **Face Mask Detection** dataset (853 images, 4,072 boxes,
Pascal VOC annotations) and converted to YOLO format.

| Class | Meaning |
|---|---|
| `with_mask` | mask worn correctly |
| `without_mask` | no mask |
| `mask_weared_incorrect` | mask present but worn incorrectly |

---

## Results

Validation split: 183 images, 974 instances (YOLO26n, 60 epochs, `imgsz=512`).

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| **all** | 183 | 974 | 0.794 | 0.737 | 0.794 | 0.538 |
| with_mask | 164 | 761 | 0.906 | 0.908 | 0.934 | 0.622 |
| without_mask | 59 | 183 | 0.812 | 0.770 | 0.815 | 0.534 |
| mask_weared_incorrect | 25 | 30 | 0.664 | 0.533 | 0.634 | 0.458 |

**`mask_weared_incorrect` is the known weak class.** It has only 123 training
boxes (3% of the data). The confusion matrix shows it is genuinely confused with
the other two classes, not merely under-detected: 43% correct, with 16% leaking
to `without_mask` and 9% to `with_mask`.

If your use case only needs mask / no-mask, merge the third class into
`without_mask` — this typically raises overall mAP substantially.

---

## Project layout

```
Mask face/
├── app.py                 # PySide6 glassmorphism desktop app
├── train.py               # training entry point (CLI)
├── model.py               # MaskModel wrapper around Ultralytics YOLO
├── dataset.py             # PyTorch/torchvision detection dataset (VOC XML)
├── convert_to_yolo.py     # VOC XML -> YOLO .txt + data.yaml converter
├── requirements.txt
├── yolo26n.pt             # pretrained starting weights
├── data/
│   ├── images/            # 853 PNGs
│   └── annotations/       # 853 Pascal VOC XMLs
├── yolo_dataset/          # generated
│   ├── data.yaml
│   ├── images/{train,val}/
│   └── labels/{train,val}/
└── runs/detect/           # training runs, metrics, plots
```

`data/` and `yolo_dataset/` are git-ignored because of size. `yolo_dataset/` is
fully regenerable from `data/` with the converter below.

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

A CUDA-capable GPU is optional but strongly recommended for training. Inference
runs fine on CPU.

---

## Usage

### 1. Convert annotations to YOLO format

Only needed if `yolo_dataset/` is missing.

```bash
python convert_to_yolo.py --link-images
```

Options:

| Flag | Effect |
|---|---|
| `--val-split 0.2` | validation fraction (default 0.2) |
| `--seed 42` | split seed; same seed = same split |
| `--copy-images` | copy images instead of hard-linking |
| `--link-images` | hard-link images (near-zero extra disk) |
| `--out DIR` | output directory (default `yolo_dataset/`) |

Hard links are used by default so the ~350 MB of images is not duplicated. The
split is hash-based, so it stays stable when files are added and matches the
split used by `dataset.py`.

### 2. Train

```bash
python train.py --epochs 100 --batch 32 --imgsz 512
```

Defaults: `imgsz=512`, `batch=32`, `patience=50`, export to ONNX after training.
Output goes to `runs/detect/mask_yolo26n/`.

Useful flags:

| Flag | Effect |
|---|---|
| `--device 0` | train on GPU 0 |
| `--patience 50` | early-stop patience (keep below `--epochs`) |
| `--export none` | skip ONNX export |
| `--skip-val` | skip validation of the best checkpoint |
| `--name RUN` | run folder name |

### 3. Run the desktop app

```bash
python app.py
```

Or point it at specific weights:

```bash
python app.py path\to\best.pt
```

By default it loads `runs/detect/train-7/weights/best.pt`. If that file is
missing, the header shows `weights: NOT FOUND` and Detect is disabled with an
explanatory message.

**App workflow:** Open image → adjust the confidence slider → Detect → Save
result. Inference runs on a background thread, so the window stays responsive.
Detections are counted per class and drawn on the preview.

---

## Programmatic use

```python
from model import MaskModel

model = MaskModel(checkpoint_path="runs/detect/train-7/weights/best.pt")
results = model.predict("photo.jpg", conf=0.25)

for box in results[0].boxes:
    print(results[0].names[int(box.cls)], float(box.conf), box.xyxy.tolist())
```

Validate or export a checkpoint:

```python
model.val(checkpoint_path="best.pt", data_path="yolo_dataset/data.yaml")
model.export(checkpoint_path="best.pt", fmt="onnx", imgsz=512)
```

`MaskModel` also exposes `export_best_of(run_dir)`, which prefers `best.pt` and
falls back to `last.pt`.

### torchvision dataset (alternative to YOLO)

`dataset.py` provides the same data as a `torch.utils.data.Dataset` returning
`(image, target)` for torchvision detection models such as Faster R-CNN:

```python
from dataset import build_dataloaders

train_loader, val_loader = build_dataloaders("data", batch_size=4, size=512)
```

It validates that the train/val splits are disjoint and cover the dataset
exactly once, and raises if that invariant is ever broken.

---

## Notes and known limitations

- **Class imbalance.** `mask_weared_incorrect` is 3% of the data and is the
  weakest class by a wide margin. Address it with oversampling, loss weighting,
  or by merging it into `without_mask`.
- **`imgsz` must be a multiple of 32.** Ultralytics silently rounds up otherwise
  (500 becomes 512). Both `train.py` and `MaskModel.train` round explicitly so
  the value logged is the value used.
- **Windows multiprocessing.** Training with `workers>0` requires a
  `if __name__ == "__main__":` guard in the entry script, because Windows spawns
  dataloader workers by re-importing the main module. `train.py` already has it.
  Without the guard you get `RuntimeError: An attempt has been made to start a
  new process before the current process has finished its bootstrapping phase`.
- **Early stopping.** `patience` must be smaller than `epochs` or it can never
  trigger. `train.py` warns when this is the case.
- **Training metrics.** In the reference run the best epoch was 56 of 60, and
  mAP50-95 declined slightly over the final epochs — the run was marginally
  over-trained. More epochs with real early stopping is worth trying.
- **`data/` uses VOC XML bounding boxes, not segmentation masks.** There are no
  per-pixel masks in this dataset.

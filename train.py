from __future__ import annotations

import argparse
import os
from pathlib import Path

from model import MaskModel

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "yolo_dataset" / "data.yaml"


def parse_args():
    ap = argparse.ArgumentParser(description="Train the face-mask detector.")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--checkpoint", default=str(HERE / "yolo26n.pt"),
                    help="pretrained weights to start from")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=512,
                    help="must be a multiple of 32")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--patience", type=int, default=50,
                    help="early-stop patience in epochs")
    ap.add_argument("--device", default=None, help="'0' for GPU, 'cpu' otherwise")
    ap.add_argument("--project", default=str(HERE / "runs" / "detect"))
    ap.add_argument("--name", default="mask_yolo26n")
    ap.add_argument("--export", default="onnx",
                    help="export format after training, or 'none' to skip")
    ap.add_argument("--skip-val", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.data):
        raise SystemExit(f"data yaml not found: {args.data}")

    if args.epochs < args.patience:
        print(f"note: patience={args.patience} exceeds epochs={args.epochs}, "
              f"so early stopping can never trigger")

    model = MaskModel(checkpoint_path=args.checkpoint, device=args.device)

    model.train(
        data_path=args.data,
        epochs=args.epochs,
        imgsize=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        project=args.project,
        name=args.name,
    )

    run_dir = Path(args.project) / args.name
    best = run_dir / "weights" / "best.pt"
    print(f"\nrun dir: {run_dir}")
    print(f"best   : {best}")

    if not args.skip_val and best.is_file():
        print("\nvalidating best checkpoint on the val split...")
        model.val(checkpoint_path=str(best), data_path=args.data)

    if args.export.lower() != "none":
        print(f"\nexporting best checkpoint to {args.export}...")
        model.export_best_of(run_dir, fmt=args.export, imgsz=args.imgsz)

    print("\ndone.")


if __name__ == "__main__":
    main()

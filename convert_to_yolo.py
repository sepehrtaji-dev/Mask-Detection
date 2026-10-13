from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import xml.etree.ElementTree as ET

CLASSES = ("with_mask", "without_mask", "mask_weared_incorrect")
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def _text(node, tag, default=None):
    if node is None:
        return default
    child = node.find(tag)
    if child is None or child.text is None:
        return default
    value = child.text.strip()
    return value if value else default


def parse_voc_xml(xml_path):
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return 0, 0, [], []

    size = root.find("size")
    width = int(float(_text(size, "width", 0)))
    height = int(float(_text(size, "height", 0)))

    boxes, labels = [], []
    for obj in root.findall("object"):
        name = _text(obj, "name")
        if name not in CLASS_TO_IDX:
            continue

        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        try:
            xmin = float(_text(bndbox, "xmin", 0))
            ymin = float(_text(bndbox, "ymin", 0))
            xmax = float(_text(bndbox, "xmax", 0))
            ymax = float(_text(bndbox, "ymax", 0))
        except ValueError:
            continue

        if xmax <= xmin or ymax <= ymin:
            continue

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(CLASS_TO_IDX[name])

    return width, height, boxes, labels


def _find_image(image_dir, stem):
    for ext in IMAGE_EXTS:
        path = os.path.join(image_dir, stem + ext)
        if os.path.isfile(path):
            return path
    return None


def _bucket(stem, seed):
    digest = hashlib.md5(f"{seed}:{stem}".encode()).hexdigest()
    return (int(digest[:8], 16) % 1000) / 1000.0


def convert(image_dir, ann_dir, out_dir, val_split=0.2, seed=42,
            copy_images=False, link_images=False):
    stems = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(image_dir)
        if f.lower().endswith(IMAGE_EXTS)
    )
    stems = [s for s in stems if os.path.isfile(os.path.join(ann_dir, s + ".xml"))]
    if not stems:
        raise RuntimeError(f"no image/annotation pairs found in {image_dir} / {ann_dir}")

    for split in ("train", "val"):
        os.makedirs(os.path.join(out_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", split), exist_ok=True)

    stats = {s: {"images": 0, "boxes": 0, "empty": 0} for s in ("train", "val")}
    per_class = {name: 0 for name in CLASSES}
    skipped = []

    for stem in stems:
        split = "val" if _bucket(stem, seed) < val_split else "train"

        img_path = _find_image(image_dir, stem)
        if img_path is None:
            skipped.append((stem, "no image file"))
            continue

        width, height, boxes, labels = parse_voc_xml(
            os.path.join(ann_dir, stem + ".xml"))

        if width <= 0 or height <= 0:
            skipped.append((stem, "XML has no usable <size>"))
            continue

        lines = []
        for (xmin, ymin, xmax, ymax), label in zip(boxes, labels):
            xmin = max(0.0, min(xmin, width))
            xmax = max(0.0, min(xmax, width))
            ymin = max(0.0, min(ymin, height))
            ymax = max(0.0, min(ymax, height))

            w = xmax - xmin
            h = ymax - ymin
            if w <= 0 or h <= 0:
                continue

            cx = (xmin + xmax) / 2.0 / width
            cy = (ymin + ymax) / 2.0 / height
            nw = w / width
            nh = h / height

            cx = min(max(cx, 0.0), 1.0)
            cy = min(max(cy, 0.0), 1.0)
            nw = min(max(nw, 0.0), 1.0)
            nh = min(max(nh, 0.0), 1.0)

            lines.append(f"{label} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            per_class[CLASSES[label]] += 1

        label_path = os.path.join(out_dir, "labels", split, stem + ".txt")
        with open(label_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
            if lines:
                fh.write("\n")

        base = os.path.basename(img_path)
        dest = os.path.join(out_dir, "images", split, base)
        if copy_images:
            shutil.copy2(img_path, dest)
        elif link_images:
            try:
                os.link(img_path, dest)
            except OSError:
                shutil.copy2(img_path, dest)

        stats[split]["images"] += 1
        stats[split]["boxes"] += len(lines)
        if not lines:
            stats[split]["empty"] += 1

    abs_out = os.path.abspath(out_dir)
    yaml_path = os.path.join(out_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as fh:
        fh.write(f"path: {abs_out}\n")
        fh.write("train: images/train\n")
        fh.write("val: images/val\n")
        fh.write(f"nc: {len(CLASSES)}\n")
        fh.write("names:\n")
        for i, name in enumerate(CLASSES):
            fh.write(f"  {i}: {name}\n")

    print(f"output: {abs_out}")
    print(f"data.yaml: {yaml_path}")
    for split in ("train", "val"):
        s = stats[split]
        print(f"  {split}: {s['images']} images, {s['boxes']} boxes, {s['empty']} empty")
    print("  class boxes:", per_class)
    if skipped:
        print(f"  skipped {len(skipped)} file(s):")
        for stem, why in skipped[:10]:
            print(f"    {stem}: {why}")
        if len(skipped) > 10:
            print(f"    ... and {len(skipped) - 10} more")

    return yaml_path


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=os.path.join(here, "data", "images"))
    ap.add_argument("--annotations", default=os.path.join(here, "data", "annotations"))
    ap.add_argument("--out", default=os.path.join(here, "yolo_dataset"))
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy-images", action="store_true",
                    help="copy images instead of referencing originals")
    ap.add_argument("--link-images", action="store_true",
                    help="hard-link images (falls back to copy)")
    args = ap.parse_args()

    convert(args.images, args.annotations, args.out,
            val_split=args.val_split, seed=args.seed,
            copy_images=args.copy_images, link_images=args.link_images)


if __name__ == "__main__":
    main()

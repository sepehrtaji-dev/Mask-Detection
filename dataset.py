from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import torch
from PIL import Image
from torch.utils.data import Dataset
#imported libs

CLASSES = ("with_mask", "without_mask", "mask_weared_incorrect") #classes in a touple
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)} #transformed classes into indexs

_DEFAULT_SIZE = (400, 400) #image size


def _text(node, tag, default=None):
    child = node.find(tag)
    if child is None or child.text is None:
        return default
    value = child.text.strip()
    return value if value else default
#function to find tags in xml file

def parse_voc_xml(xml_path):
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return _DEFAULT_SIZE[0], _DEFAULT_SIZE[1], [], []

    size = root.find("size")
    width = int(float(_text(size, "width", _DEFAULT_SIZE[0]))) if size is not None else _DEFAULT_SIZE[0]
    height = int(float(_text(size, "height", _DEFAULT_SIZE[1]))) if size is not None else _DEFAULT_SIZE[1]
    #finding size of the image
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


class MaskDetectionDataset(Dataset):
    def __init__(self, root, split="train", transforms=None, val_split=0.2, seed=42):
        self.root = root
        self.image_dir = os.path.join(root, "images")
        self.ann_dir = os.path.join(root, "annotations")
        self.transforms = transforms
        self.split = split

        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"image dir not found: {self.image_dir}")
        if not os.path.isdir(self.ann_dir):
            raise FileNotFoundError(f"annotation dir not found: {self.ann_dir}")

        stems = sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(self.image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        self.stems = [s for s in stems if os.path.isfile(self._xml_path(s))]
        if not self.stems:
            raise RuntimeError(f"no image/annotation pairs found under {root}")

        if split in ("train", "val"):
            self.stems = self._split_stems(self.stems, val_split, seed)

        if not self.stems:
            raise RuntimeError(f"split '{split}' is empty (val_split={val_split})")

    def _xml_path(self, stem):
        return os.path.join(self.ann_dir, stem + ".xml")

    def _img_path(self, stem):
        for ext in (".png", ".jpg", ".jpeg"):
            path = os.path.join(self.image_dir, stem + ext)
            if os.path.isfile(path):
                return path
        raise FileNotFoundError(f"no image file for stem {stem}")

    def _split_stems(self, stems, val_split, seed):
        if not 0.0 <= val_split < 1.0:
            raise ValueError("val_split must be in [0, 1)")

        import hashlib

        def bucket(stem):
            digest = hashlib.md5(f"{seed}:{stem}".encode()).hexdigest()
            return (int(digest[:8], 16) % 1000) / 1000.0

        val = [s for s in stems if bucket(s) < val_split]
        train = [s for s in stems if bucket(s) >= val_split]

        if not val and len(train) > 1:
            val, train = train[:1], train[1:]
        if not train and len(val) > 1:
            train, val = val[:1], val[1:]

        return train if self.split == "train" else val

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, index):
        stem = self.stems[index]
        img_path = self._img_path(stem)

        image = Image.open(img_path).convert("RGB")
        real_w, real_h = image.size

        width, height, boxes, labels = parse_voc_xml(self._xml_path(stem))

        if width <= 0 or height <= 0:
            width, height = real_w, real_h

        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            boxes[:, 0::2].clamp_(min=0, max=real_w)
            boxes[:, 1::2].clamp_(min=0, max=real_h)
            keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[keep]
            labels = torch.tensor(labels, dtype=torch.int64)[keep]
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            "orig_size": torch.tensor([real_h, real_w], dtype=torch.int64),
            "stem": stem,
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        else:
            import numpy as np

            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        return image, target

    def class_counts(self):
        counts = {name: 0 for name in CLASSES}
        for stem in self.stems:
            _, _, _, labels = parse_voc_xml(self._xml_path(stem))
            for label in labels:
                counts[CLASSES[label]] += 1
        return counts


def get_transforms(train=True, size=512):
    try:
        from torchvision.transforms import v2
    except ImportError:
        from torchvision import transforms as T

        return T.Compose([T.Resize((size, size)), T.ToTensor()])

    if train:
        return v2.Compose([
            v2.Resize((size, size), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return v2.Compose([
        v2.Resize((size, size), antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def collate_fn(batch):
    return tuple(zip(*batch))


def build_dataloaders(root, batch_size=4, num_workers=0, size=512, val_split=0.2, seed=42):
    from torch.utils.data import DataLoader

    train_ds = MaskDetectionDataset(root, split="train", val_split=val_split,
                                    seed=seed, transforms=get_transforms(train=True, size=size))
    val_ds = MaskDetectionDataset(root, split="val", val_split=val_split,
                                  seed=seed, transforms=get_transforms(train=False, size=size))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate_fn)

    overlap = set(train_ds.stems) & set(val_ds.stems)
    if overlap:
        raise RuntimeError(
            f"train/val overlap of {len(overlap)} images - split logic is broken")
    if len(train_ds) + len(val_ds) != len(set(train_ds.stems) | set(val_ds.stems)):
        raise RuntimeError("train/val splits do not cover the dataset exactly once")

    return train_loader, val_loader


if __name__ == "__main__":
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    ds = MaskDetectionDataset(root, split="train", transforms=get_transforms(train=True))
    print(f"train split: {len(ds)} images")
    print(f"class counts: {ds.class_counts()}")

    image, target = ds[0]
    print(f"image tensor: {tuple(image.shape)} dtype={image.dtype}")
    print(f"boxes: {tuple(target['boxes'].shape)} labels={target['labels'].tolist()}")
    print(f"stem: {target['stem']}")

    train_loader, val_loader = build_dataloaders(root, batch_size=2)
    images, targets = next(iter(train_loader))
    print(f"batch: {len(images)} images, {len(targets)} targets")
    print(f"train={len(train_loader.dataset)} val={len(val_loader.dataset)}")
    print("OK")

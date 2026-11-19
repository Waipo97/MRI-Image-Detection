"""Dataset utilities: extract, audit labels, clean, split, and write the YOLO folder layout."""
import hashlib, random, shutil, zipfile, collections
from pathlib import Path
import pandas as pd
import yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# --------------------------------------------------------------------------- extract / scan
def extract_zip(zip_path, raw_dir):
    """Extract archive.zip into raw_dir. Skipped if this same zip was already extracted.
    Returns the dataset root (the folder whose sub-folders are the class folders)."""
    zip_path, raw_dir = Path(zip_path), Path(raw_dir)
    marker = raw_dir / "_extracted"
    stamp = str(zip_path.stat().st_size)             # a different zip means the old extraction is stale
    if not marker.exists() or marker.read_text() != stamp:
        for old in raw_dir.iterdir():
            if old.resolve() != zip_path.resolve():
                shutil.rmtree(old) if old.is_dir() else old.unlink()
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(raw_dir)
        marker.write_text(stamp)
    return find_dataset_root(raw_dir)


def find_dataset_root(raw_dir):
    """The folder whose sub-folders all contain images (the class folders), searched up to 3 levels deep."""
    raw_dir = Path(raw_dir)
    for d in [raw_dir, *sorted(p for p in raw_dir.rglob("*") if p.is_dir() and "__MACOSX" not in p.parts
                               and len(p.relative_to(raw_dir).parts) <= 3)]:
        subs = [s for s in d.iterdir() if s.is_dir()]
        if subs and all(any(f.suffix.lower() in IMG_EXT for f in s.iterdir()) for s in subs):
            return d
    raise FileNotFoundError(f"No class folders with images found under {raw_dir}")


def scan_dataset(root):
    """One row per image: its class folder, parsed YOLO boxes and a label-quality 'issue' (or None)."""
    root = Path(root)
    rows = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        for img in sorted(f for f in folder.iterdir() if f.suffix.lower() in IMG_EXT):
            lbl = img.with_suffix(".txt")
            boxes, issue = [], None
            if not lbl.exists():
                issue = "no_label_file"
            else:
                for line in lbl.read_text().splitlines():
                    parts = line.split()
                    if not parts:
                        continue
                    try:
                        if len(parts) < 5:
                            raise ValueError
                        c = int(float(parts[0])); xywh = [float(v) for v in parts[1:5]]
                    except ValueError:
                        issue = "malformed_label"; break
                    if not all(0 <= v <= 1 for v in xywh) or xywh[2] <= 0 or xywh[3] <= 0:
                        issue = "bad_coordinates"
                    boxes.append((c, *xywh))
                if issue is None and not boxes:
                    issue = "empty_label"
            rows.append(dict(image=str(img), folder=folder.name, label=str(lbl), boxes=boxes,
                             n_boxes=len(boxes), issue=issue,
                             md5=hashlib.md5(img.read_bytes()).hexdigest()))
    return pd.DataFrame(rows)


def find_orphan_labels(root):
    """Label-like .txt files that have no image next to them (e.g. classes.txt, stray copies)."""
    root = Path(root)
    return sorted(str(t.relative_to(root)) for t in root.rglob("*.txt")
                  if not any(t.with_suffix(e).exists() for e in IMG_EXT))


# --------------------------------------------------------------------------- clean
def infer_class_map(df):
    """The class id of each folder = the most common id in its (valid) labels.
    Returns ({folder: id}, [names ordered by id])."""
    folder2id = {}
    for folder, g in df[df.issue.isna()].groupby("folder"):
        cnt = collections.Counter(b[0] for boxes in g.boxes for b in boxes)
        folder2id[folder] = cnt.most_common(1)[0][0]
    if len(set(folder2id.values())) != len(folder2id):
        raise ValueError(f"Two folders map to the same class id: {folder2id}")
    names = [f for f, _ in sorted(folder2id.items(), key=lambda kv: kv[1])]
    if sorted(folder2id.values()) != list(range(len(names))):
        raise ValueError(f"Class ids are not 0..N-1: {folder2id}")
    return folder2id, names


def clean_dataset(df):
    """Flag images that cannot be trusted. Adds/updates the 'issue' column:
    class_mismatch (a box class differs from the folder's class) and cross_class_duplicate
    (identical image bytes appear in different class folders)."""
    df = df.copy()
    folder2id, names = infer_class_map(df)
    mism = [i for i, r in df.iterrows()
            if pd.isna(r.issue) and any(b[0] != folder2id[r.folder] for b in r.boxes)]
    df.loc[mism, "issue"] = "class_mismatch"
    ok = df[df.issue.isna()]
    multi = ok.groupby("md5").folder.nunique()
    bad_md5 = set(multi[multi > 1].index)
    df.loc[df.md5.isin(bad_md5) & df.issue.isna(), "issue"] = "cross_class_duplicate"
    return df, folder2id, names


# --------------------------------------------------------------------------- split
def split_dataset(df, ratios=(0.70, 0.20, 0.10), seed=0):
    """Stratified (per class folder) split. Byte-identical images always stay in the same split,
    so exact duplicates cannot leak from train into val/test."""
    df = df.copy()
    df["split"] = None
    r_train, r_val = ratios[0], ratios[0] + ratios[1]
    rng = random.Random(seed)
    for folder, g in df[df.issue.isna()].groupby("folder"):
        groups = [list(ix) for _, ix in g.groupby("md5").groups.items()]
        groups.sort(key=lambda x: (min(x)))          # deterministic order before shuffling
        rng.shuffle(groups)
        total, seen = len(g), 0
        for grp in groups:
            f = seen / total
            df.loc[grp, "split"] = "train" if f < r_train else "val" if f < r_val else "test"
            seen += len(grp)
    return df


# --------------------------------------------------------------------------- write YOLO layout
def build_yolo_dataset(df, names, yolo_dir):
    """Copy images + cleaned labels into <yolo_dir>/{train,val,test}/{images,labels} and write data.yaml."""
    yolo_dir = Path(yolo_dir)
    if yolo_dir.exists():
        shutil.rmtree(yolo_dir)
    use = df[df.split.notna()]
    if use.image.map(lambda p: Path(p).name).duplicated().any():
        raise ValueError("Duplicate image file names across folders; rename before building the dataset.")
    for _, r in use.iterrows():
        img = Path(r.image)
        (yolo_dir / r.split / "images").mkdir(parents=True, exist_ok=True)
        (yolo_dir / r.split / "labels").mkdir(parents=True, exist_ok=True)
        shutil.copy(img, yolo_dir / r.split / "images" / img.name)
        with open(yolo_dir / r.split / "labels" / (img.stem + ".txt"), "w") as f:
            for c, x, y, w, h in r.boxes:
                f.write(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
    cfg = {"path": str(yolo_dir.resolve()), "train": "train/images", "val": "val/images",
           "test": "test/images", "names": {i: n for i, n in enumerate(names)}}
    yaml.safe_dump(cfg, open(yolo_dir / "data.yaml", "w"), sort_keys=False)
    return yolo_dir / "data.yaml"


def split_summary(df, names):
    """Images and boxes per split and class."""
    rows = []
    for split, g in df[df.split.notna()].groupby("split"):
        cnt = collections.Counter(b[0] for boxes in g.boxes for b in boxes)
        rows.append({"split": split, "images": len(g), **{names[k]: cnt.get(k, 0) for k in range(len(names))}})
    return pd.DataFrame(rows).set_index("split").loc[["train", "val", "test"]]

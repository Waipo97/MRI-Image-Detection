import cv2
import matplotlib.pyplot as plt

PALETTE = [(255, 99, 71), (65, 105, 225), (60, 179, 113), (255, 215, 0), (186, 85, 211)]


def draw_boxes(img_path, boxes, names, thickness=2):
    """boxes: iterable of (class_id, xc, yc, w, h) normalised. Returns an RGB array."""
    im = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    h, w = im.shape[:2]
    for c, x, y, bw, bh in boxes:
        p1 = (int((x - bw / 2) * w), int((y - bh / 2) * h)); p2 = (int((x + bw / 2) * w), int((y + bh / 2) * h))
        col = PALETTE[int(c) % len(PALETTE)]
        cv2.rectangle(im, p1, p2, col, thickness)
        cv2.putText(im, names[int(c)], (p1[0], max(p1[1] - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, .55, col, 2)
    return im


def read_yolo_labels(path):
    try:
        return [tuple([int(float(l.split()[0])), *map(float, l.split()[1:5])]) for l in open(path) if l.strip()]
    except FileNotFoundError:
        return []


def show_grid(items, ncols=3, figsize=(12, 4), titles=None):
    """items: list of RGB arrays."""
    nrows = -(-len(items) // ncols)
    fig, ax = plt.subplots(nrows, ncols, figsize=(figsize[0], figsize[1] * nrows), squeeze=False)
    for a in ax.ravel():
        a.axis("off")
    for i, im in enumerate(items):
        ax.ravel()[i].imshow(im)
        if titles:
            ax.ravel()[i].set_title(titles[i], fontsize=9)
    plt.tight_layout()
    return fig

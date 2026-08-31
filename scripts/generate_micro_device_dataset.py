import argparse
import math
import random
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from tqdm import tqdm


IMG_SIZE = 128
TRAIN_SAMPLES_PER_CLASS = 3000
VAL_SAMPLES_PER_CLASS = 600

DATA_ROOT = Path("data/micro_devices")
PREVIEW_ROOT = Path("outputs/micro_device_dataset_preview")
REPORT_PATH = Path("reports/micro_device_dataset_guide_cn.md")

CLASSES = [
    ("0_diode", "diode"),
    ("1_bjt", "bjt"),
    ("2_nmos", "nmos"),
    ("3_pmos", "pmos"),
    ("4_resistor", "resistor"),
]

BACKGROUND = 0
FOREGROUND = 255


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a five-class synthetic micro-device symbol dataset for ImageFolder."
    )
    parser.add_argument("--data-root", type=str, default=str(DATA_ROOT))
    parser.add_argument("--preview-root", type=str, default=str(PREVIEW_ROOT))
    parser.add_argument("--report-path", type=str, default=str(REPORT_PATH))
    parser.add_argument("--train-per-class", type=int, default=TRAIN_SAMPLES_PER_CLASS)
    parser.add_argument("--val-per-class", type=int, default=VAL_SAMPLES_PER_CLASS)
    parser.add_argument("--preview-per-class", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing generated micro-device dataset and preview directories before generation.",
    )
    return parser.parse_args()


def line(draw, points, width, fill=FOREGROUND):
    draw.line(points, fill=fill, width=width, joint="curve")


def polygon(draw, points, fill=None, outline=FOREGROUND, width=3):
    draw.polygon(points, fill=fill, outline=outline)
    if outline is not None and width > 1:
        closed = points + [points[0]]
        line(draw, closed, width, fill=outline)


def draw_arrow(draw, start, end, width, fill=FOREGROUND):
    line(draw, [start, end], width, fill=fill)
    sx, sy = start
    ex, ey = end
    angle = math.atan2(ey - sy, ex - sx)
    head_len = 9
    spread = math.radians(28)
    p1 = (
        ex - head_len * math.cos(angle - spread),
        ey - head_len * math.sin(angle - spread),
    )
    p2 = (
        ex - head_len * math.cos(angle + spread),
        ey - head_len * math.sin(angle + spread),
    )
    polygon(draw, [(ex, ey), p1, p2], fill=fill, outline=fill, width=1)


def draw_diode(draw, width):
    line(draw, [(16, 64), (38, 64)], width)
    line(draw, [(90, 64), (112, 64)], width)
    polygon(draw, [(38, 35), (38, 93), (80, 64)], fill=None, width=width)
    line(draw, [(84, 34), (84, 94)], width)


def draw_resistor(draw, width):
    line(draw, [(14, 64), (28, 64)], width)
    pts = [(28, 64)]
    x = 28
    amp = 18
    step = 9
    for i in range(8):
        x += step
        y = 64 - amp if i % 2 == 0 else 64 + amp
        pts.append((x, y))
    pts.append((100, 64))
    line(draw, pts, width)
    line(draw, [(100, 64), (114, 64)], width)


def draw_bjt(draw, width):
    line(draw, [(30, 28), (30, 100)], width)
    line(draw, [(12, 64), (30, 64)], width)
    line(draw, [(30, 45), (84, 22)], width)
    line(draw, [(84, 22), (108, 22)], width)
    line(draw, [(30, 83), (76, 106)], width)
    line(draw, [(76, 106), (108, 106)], width)
    draw_arrow(draw, (52, 94), (75, 105), width)
    draw.ellipse((20, 18, 98, 110), outline=FOREGROUND, width=max(1, width // 2))


def draw_nmos(draw, width):
    line(draw, [(22, 34), (22, 94)], width)
    line(draw, [(8, 64), (22, 64)], width)
    line(draw, [(44, 34), (44, 94)], width)
    line(draw, [(44, 34), (94, 34)], width)
    line(draw, [(44, 94), (94, 94)], width)
    line(draw, [(94, 18), (94, 34)], width)
    line(draw, [(94, 94), (94, 110)], width)
    line(draw, [(58, 47), (58, 81)], max(1, width - 1))
    draw_arrow(draw, (72, 78), (55, 78), max(1, width - 1))


def draw_pmos(draw, width):
    line(draw, [(18, 34), (18, 94)], width)
    draw.ellipse((23, 56, 39, 72), outline=FOREGROUND, width=width)
    line(draw, [(4, 64), (18, 64)], width)
    line(draw, [(44, 34), (44, 94)], width)
    line(draw, [(44, 34), (94, 34)], width)
    line(draw, [(44, 94), (94, 94)], width)
    line(draw, [(94, 18), (94, 34)], width)
    line(draw, [(94, 94), (94, 110)], width)
    line(draw, [(58, 47), (58, 81)], max(1, width - 1))
    draw_arrow(draw, (55, 78), (72, 78), max(1, width - 1))


DRAW_FUNCS = {
    "diode": draw_diode,
    "bjt": draw_bjt,
    "nmos": draw_nmos,
    "pmos": draw_pmos,
    "resistor": draw_resistor,
}


def draw_clean_symbol(class_key, rng):
    img = Image.new("L", (IMG_SIZE, IMG_SIZE), BACKGROUND)
    draw = ImageDraw.Draw(img)
    width = int(rng.integers(3, 7))
    DRAW_FUNCS[class_key](draw, width)
    return img


def affine_augment(img, rng):
    scale = float(rng.uniform(0.82, 1.15))
    angle = float(rng.uniform(-14.0, 14.0))
    tx = int(rng.integers(-10, 11))
    ty = int(rng.integers(-10, 11))

    scaled_size = max(24, int(round(IMG_SIZE * scale)))
    scaled = img.resize((scaled_size, scaled_size), Image.Resampling.BICUBIC)
    canvas = Image.new("L", (IMG_SIZE, IMG_SIZE), BACKGROUND)
    x0 = (IMG_SIZE - scaled_size) // 2 + tx
    y0 = (IMG_SIZE - scaled_size) // 2 + ty
    canvas.paste(scaled, (x0, y0))
    return canvas.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=BACKGROUND)


def apply_line_breaks(img, rng):
    if rng.random() > 0.35:
        return img
    arr_img = img.copy()
    draw = ImageDraw.Draw(arr_img)
    n_breaks = int(rng.integers(1, 4))
    for _ in range(n_breaks):
        x = int(rng.integers(18, 110))
        y = int(rng.integers(18, 110))
        w = int(rng.integers(4, 11))
        h = int(rng.integers(3, 9))
        draw.rectangle((x - w, y - h, x + w, y + h), fill=BACKGROUND)
    return arr_img


def apply_noise_blur_brightness(img, rng):
    if rng.random() < 0.55:
        radius = float(rng.uniform(0.2, 0.8))
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(float(rng.uniform(0.82, 1.18)))

    arr = np.asarray(img, dtype=np.float32) / 255.0
    noise_sigma = float(rng.uniform(0.0, 0.035))
    if noise_sigma > 0:
        arr += rng.normal(0.0, noise_sigma, size=arr.shape).astype(np.float32)
    arr = np.clip(arr, 0.0, 1.0)
    return Image.fromarray((arr * 255.0).astype(np.uint8), mode="L")


def make_sample(class_key, rng):
    img = draw_clean_symbol(class_key, rng)
    img = affine_augment(img, rng)
    img = apply_line_breaks(img, rng)
    img = apply_noise_blur_brightness(img, rng)
    return img


def save_png_01(img, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def prepare_dirs(data_root, preview_root, clean):
    if clean:
        for path in [data_root, preview_root]:
            if path.exists():
                shutil.rmtree(path)
    for split in ["train", "val"]:
        for class_dir, _ in CLASSES:
            (data_root / split / class_dir).mkdir(parents=True, exist_ok=True)
    preview_root.mkdir(parents=True, exist_ok=True)


def generate_split(split, samples_per_class, data_root, seed):
    counts = {}
    for class_index, (class_dir, class_key) in enumerate(CLASSES):
        out_dir = data_root / split / class_dir
        rng = np.random.default_rng(seed + class_index * 100_000 + (0 if split == "train" else 50_000))
        for idx in tqdm(range(samples_per_class), desc=f"{split}/{class_dir}"):
            img = make_sample(class_key, rng)
            save_png_01(img, out_dir / f"{class_key}_{idx:05d}.png")
        counts[class_dir] = samples_per_class
    return counts


def generate_previews(preview_root, preview_per_class, seed):
    preview_paths = []
    grid_images = []
    for class_index, (class_dir, class_key) in enumerate(CLASSES):
        class_preview_dir = preview_root / class_dir
        class_preview_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed + 900_000 + class_index * 10_000)
        for idx in range(preview_per_class):
            img = make_sample(class_key, rng)
            path = class_preview_dir / f"preview_{idx:02d}.png"
            save_png_01(img, path)
            preview_paths.append(path)
            if idx < 5:
                grid_images.append((class_dir, img))

    rows = len(CLASSES)
    cols = 5
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.7, rows * 1.7), constrained_layout=True)
    for r, (class_dir, _) in enumerate(CLASSES):
        class_imgs = [img for label, img in grid_images if label == class_dir]
        for c in range(cols):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(class_imgs):
                ax.imshow(class_imgs[c], cmap="gray", vmin=0, vmax=255)
            if c == 0:
                ax.set_title(class_dir, fontsize=9)
    grid_path = preview_root / "preview_grid.png"
    fig.savefig(grid_path, dpi=180)
    plt.close(fig)
    return preview_paths, grid_path


def write_report(report_path, data_root, preview_root, train_count, val_count):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    text = f"""# 微电子器件结构图案数据集生成说明

生成脚本：`generate_micro_device_dataset.py`

## 1. 数据集目的

本数据集用于启动第二条代码主线：微电子器件结构图案识别。它生成一个可由 `torchvision.datasets.ImageFolder` 直接读取的五分类灰度图数据集，后续可以替换 VCSEL near-light 输入，作为 D2NN 图案识别任务的初始 baseline 数据。

需要注意：这是规则生成的简化微电子器件符号/结构图案数据集，不是真实芯片版图，也不是真实显微图。

## 2. 五类图案含义

| 类别目录 | 含义 | 图案特征 |
|---|---|---|
| `0_diode` | 二极管 | 三角/箭头感结构和竖线 |
| `1_bjt` | 三极管 | 三端结构，含发射极箭头 |
| `2_nmos` | NMOS 管 | gate、source、drain 三端结构，不带 PMOS 圆圈 |
| `3_pmos` | PMOS 管 | 与 NMOS 相似，但 gate 处带小圆圈 |
| `4_resistor` | 电阻 | 锯齿形电阻符号 |

## 3. 目录结构

输出根目录：

```text
{data_root}
```

完整结构：

```text
{data_root}/train/0_diode
{data_root}/train/1_bjt
{data_root}/train/2_nmos
{data_root}/train/3_pmos
{data_root}/train/4_resistor
{data_root}/val/0_diode
{data_root}/val/1_bjt
{data_root}/val/2_nmos
{data_root}/val/3_pmos
{data_root}/val/4_resistor
```

类别目录使用数字前缀，保证 `ImageFolder` 的类别排序稳定。

默认样本数：

| split | 每类样本数 |
|---|---:|
| train | {train_count} |
| val | {val_count} |

## 4. 如何运行脚本

默认生成完整数据集：

```powershell
python scripts/generate_micro_device_dataset.py
```

如果想先快速测试：

```powershell
python scripts/generate_micro_device_dataset.py --train-per-class 20 --val-per-class 5 --clean
```

如果需要重新生成并清空旧数据：

```powershell
python scripts/generate_micro_device_dataset.py --clean
```

## 5. 如何用 ImageFolder 读取

```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder("data/micro_devices/train", transform=transform)
val_dataset = datasets.ImageFolder("data/micro_devices/val", transform=transform)

print(train_dataset.classes)
print(train_dataset.class_to_idx)
print(len(train_dataset), len(val_dataset))
```

预期类别顺序：

```text
['0_diode', '1_bjt', '2_nmos', '3_pmos', '4_resistor']
```

## 6. 后续如何接入 D2NN

后续可以参考 V2 baseline 的数据读取方式，把 `data_root` 从 `data/vcsel_near_synth` 替换为 `data/micro_devices`。输入仍是 `[B, 1, 128, 128]` 灰度图，通过 `sqrt(I)` 转为电场振幅后进入 D2NN。

建议先做以下步骤：

1. 用 `ImageFolder` 验证类别顺序和样本数。
2. 抽取一个 batch，确认 shape 为 `[B, 1, 128, 128]`。
3. 检查像素范围是否在 `[0, 1]`。
4. 复制 V2 baseline 训练脚本为微电子器件专用脚本，再替换数据路径和输出目录。
5. 先做连续相位 baseline，再做 PTQ、QAT 和误差扰动鲁棒性评估。

## 7. 预览图

每类样例图保存到：

```text
{preview_root}
```

总览图：

```text
{preview_root / "preview_grid.png"}
```

## 8. 注意事项

- 图像为灰度 PNG。
- 图像尺寸为 `128 x 128`。
- 整个数据集统一使用黑底白线。
- 保存为 PNG 时像素为 8-bit 灰度；用 `transforms.ToTensor()` 读取后范围为 `[0, 1]`。
- 当前图案是规则生成的简化符号，主要用于算法链路验证，不代表真实微电子器件版图或真实显微图。
"""
    report_path.write_text(text, encoding="utf-8")


def print_counts(title, counts):
    print(title)
    for class_dir, _ in CLASSES:
        print(f"  {class_dir}: {counts[class_dir]}")


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    preview_root = Path(args.preview_root)
    report_path = Path(args.report_path)

    random.seed(args.seed)
    np.random.seed(args.seed)

    prepare_dirs(data_root, preview_root, args.clean)
    train_counts = generate_split("train", args.train_per_class, data_root, args.seed)
    val_counts = generate_split("val", args.val_per_class, data_root, args.seed)
    _, grid_path = generate_previews(preview_root, args.preview_per_class, args.seed)
    write_report(report_path, data_root, preview_root, args.train_per_class, args.val_per_class)

    print_counts("Train samples per class:", train_counts)
    print_counts("Val samples per class:", val_counts)
    print("Dataset output dir:", data_root)
    print("Preview output dir:", preview_root)
    print("Preview grid:", grid_path)
    print("Report:", report_path)
    print("micro device dataset generation finished")


if __name__ == "__main__":
    main()

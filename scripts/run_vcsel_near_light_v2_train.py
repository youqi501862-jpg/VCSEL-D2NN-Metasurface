import argparse
import copy
import csv
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

IMG_SIZE = 128
N_pixels = 128
DATA_ROOT = "./data/vcsel_near_synth"
OUTPUT_DIR = Path("outputs/vcsel_near_v2_train")

DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 30
DEFAULT_LR = 0.002


def parse_args():
    parser = argparse.ArgumentParser(description="Train VCSEL near-light V2 baseline.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--data-root", type=str, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=8)
    return parser.parse_args()


def ensure_output_dirs(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "samples").mkdir(parents=True, exist_ok=True)


def build_dataloaders(data_root, batch_size, num_workers):
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ]
    )

    train_dataset = torchvision.datasets.ImageFolder(
        os.path.join(data_root, "train"), transform=transform
    )
    val_dataset = torchvision.datasets.ImageFolder(
        os.path.join(data_root, "val"), transform=transform
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_dataset, val_dataset, train_dataloader, val_dataloader


class Diffractive_Layer(torch.nn.Module):
    def __init__(
        self,
        wavelength=830e-9,
        N_pixels=128,
        pixel_size=2e-6,
        distance=torch.tensor([0.005]),
        device="cpu",
    ):
        super(Diffractive_Layer, self).__init__()
        fx = np.fft.fftshift(np.fft.fftfreq(N_pixels, d=pixel_size))
        fy = np.fft.fftshift(np.fft.fftfreq(N_pixels, d=pixel_size))
        fxx, fyy = np.meshgrid(fx, fy)
        argument = (2 * np.pi) ** 2 * ((1.0 / wavelength) ** 2 - fxx**2 - fyy**2)
        tmp = np.sqrt(np.abs(argument))
        self.distance = distance.to(device)
        self.kz = torch.tensor(np.where(argument >= 0, tmp, 1j * tmp)).to(device)
        self.device = device

    def forward(self, E):
        fft_c = torch.fft.fft2(E)
        c = torch.fft.fftshift(fft_c).to(self.device)
        phase = torch.exp(1j * self.kz * self.distance).to(self.device)
        angular_spectrum = torch.fft.ifft2(torch.fft.ifftshift(c * phase))
        return angular_spectrum


class Propagation_Layer(torch.nn.Module):
    def __init__(
        self,
        wavelength=830e-9,
        N_pixels=128,
        pixel_size=2e-6,
        distance=torch.tensor([0.001]),
        device="cpu",
    ):
        super(Propagation_Layer, self).__init__()
        fx = np.fft.fftshift(np.fft.fftfreq(N_pixels, d=pixel_size))
        fy = np.fft.fftshift(np.fft.fftfreq(N_pixels, d=pixel_size))
        fxx, fyy = np.meshgrid(fx, fy)
        argument = (2 * np.pi) ** 2 * ((1.0 / wavelength) ** 2 - fxx**2 - fyy**2)
        tmp = np.sqrt(np.abs(argument))
        self.distance = distance.to(device)
        self.kz = torch.tensor(np.where(argument >= 0, tmp, 1j * tmp)).to(device)
        self.device = device

    def forward(self, E):
        fft_c = torch.fft.fft2(E)
        c = torch.fft.fftshift(fft_c)
        phase = torch.exp(1j * self.kz * self.distance).to(self.device)
        angular_spectrum = torch.fft.ifft2(torch.fft.ifftshift(c * phase))
        return angular_spectrum


class Lens_Layer(torch.nn.Module):
    def __init__(
        self,
        f=5e-3,
        wl=830e-9,
        N_pixels=128,
        pixel_size=2e-6,
        distance=torch.tensor([0.005]),
        device="cpu",
    ):
        super(Lens_Layer, self).__init__()
        coord_limit = (N_pixels // 2) * pixel_size
        mesh = np.arange(-coord_limit, coord_limit, pixel_size)
        x, y = np.meshgrid(mesh, mesh)
        self.phase_lens = torch.tensor(
            np.exp(-1j * np.pi / (wl * f) * (x**2 + y**2))
        ).to(device)

        fx = np.fft.fftshift(np.fft.fftfreq(N_pixels, d=pixel_size))
        fy = np.fft.fftshift(np.fft.fftfreq(N_pixels, d=pixel_size))
        fxx, fyy = np.meshgrid(fx, fy)
        argument = (2 * np.pi) ** 2 * ((1.0 / wl) ** 2 - fxx**2 - fyy**2)
        tmp = np.sqrt(np.abs(argument))
        self.distance = distance.to(device)
        self.kz = torch.tensor(np.where(argument >= 0, tmp, 1j * tmp)).to(device)
        self.device = device

    def forward(self, E):
        fft_c = torch.fft.fft2(E)
        c = torch.fft.fftshift(fft_c).to(self.device)
        phase = torch.exp(1j * self.kz * self.distance).to(self.device)
        angular_spectrum = torch.fft.ifft2(torch.fft.ifftshift(c * phase))
        return angular_spectrum * self.phase_lens


def detector_region_logits(Int, detector_pos):
    region_scores = []
    for up, down, left, right in detector_pos:
        region_sum = Int[:, up:down, left:right].sum(dim=(1, 2))
        region_scores.append(region_sum)
    return torch.stack(region_scores, dim=1)


def build_detector_pos():
    det_size = 24
    gap = 8
    grid_n = 3
    total_width = grid_n * det_size + (grid_n - 1) * gap
    start_pos_x = (N_pixels - total_width) // 2
    start_pos_y = (N_pixels - total_width) // 2
    layout = [
        (0, [0, 2]),
        (1, [1]),
        (2, [0, 2]),
    ]

    detector_pos = []
    for row, cols in layout:
        for col in cols:
            left = start_pos_x + col * (det_size + gap)
            right = left + det_size
            up = start_pos_y + row * (det_size + gap)
            down = up + det_size
            detector_pos.append((up, down, left, right))
    return detector_pos


def param_to_phase(param, phi_max=np.pi, mapping="sigmoid"):
    if mapping == "sigmoid":
        return phi_max * torch.sigmoid(param)
    if mapping == "linear":
        return phi_max * torch.clamp(param, 0.0, 1.0)
    raise ValueError(f"Unknown mapping: {mapping}")


def ste_quantize_phase_from_param(param, phi_max=np.pi, n_levels=4, mapping="sigmoid"):
    phi_cont = param_to_phase(param, phi_max=phi_max, mapping=mapping)
    levels = torch.linspace(
        0.0,
        phi_max * (n_levels - 1) / n_levels,
        steps=n_levels,
        device=phi_cont.device,
        dtype=phi_cont.dtype,
    )
    idx = torch.argmin(torch.abs(phi_cont.unsqueeze(-1) - levels), dim=-1)
    phi_q = levels[idx]
    return phi_cont + (phi_q - phi_cont).detach()


class DNN(torch.nn.Module):
    def __init__(
        self,
        phase=None,
        num_layers=3,
        wl=830e-9,
        N_pixels=128,
        pixel_size=2e-6,
        distance=None,
        detector_pos=None,
        phase_mapping="sigmoid",
        quantize_during_train=False,
        quant_levels=4,
        device="cpu",
    ):
        super(DNN, self).__init__()
        phase = phase or []
        distance = distance or []
        self.detector_pos = detector_pos or []
        self.phi_max = np.pi
        self.phase_mapping = phase_mapping
        self.quantize_during_train = quantize_during_train
        self.quant_levels = quant_levels
        self.device = device

        for i in range(num_layers):
            self.register_parameter("phase" + "_" + str(i), phase[i])
        for i in range(num_layers + 2):
            self.register_parameter("distance" + "_" + str(i), distance[i])

        self.lens_layer1 = Lens_Layer(device=device)
        self.diffractive_layers = torch.nn.ModuleList(
            [
                Diffractive_Layer(wl, N_pixels, pixel_size, distance[i], device=device)
                for i in range(num_layers)
            ]
        )
        self.lens_layer2 = Lens_Layer(distance=distance[-2], device=device)
        self.last_diffractive_layer = Propagation_Layer(
            wl, N_pixels, pixel_size, distance[-1], device=device
        )

    def forward(self, E):
        E = self.lens_layer1(E)
        for index, layer in enumerate(self.diffractive_layers):
            temp = layer(E)
            registered_phase = getattr(self, f"phase_{index}")
            if self.quantize_during_train:
                constr_phase = ste_quantize_phase_from_param(
                    registered_phase,
                    phi_max=self.phi_max,
                    n_levels=self.quant_levels,
                    mapping=self.phase_mapping,
                )
            else:
                constr_phase = param_to_phase(
                    registered_phase,
                    phi_max=self.phi_max,
                    mapping=self.phase_mapping,
                )
            exp_j_phase = torch.exp(1j * constr_phase)
            E = temp * exp_j_phase
            E_phase = E / (torch.abs(E) + 1e-12)
            I = torch.abs(E) ** 2
            I_th = torch.mean(I, dim=(-2, -1), keepdim=True) / 2.0
            I_out = F.relu(I - I_th)
            E = torch.sqrt(I_out + 1e-12) * E_phase

        E = self.lens_layer2(E)
        E = self.last_diffractive_layer(E)
        Int = torch.abs(E) ** 2
        output = detector_region_logits(Int, self.detector_pos)
        return output, Int


def initialize_model(device, detector_pos):
    num_layers = 3
    phase = [
        torch.nn.Parameter(torch.zeros((N_pixels, N_pixels), dtype=torch.float32))
        for _ in range(num_layers)
    ]
    distance_between_layers = [0.004, 0.004, 0.004, 0.004, 0.005]
    distance = [
        torch.nn.Parameter(torch.tensor([distance_between_layers[i]], dtype=torch.float32))
        for i in range(num_layers + 2)
    ]

    model = DNN(
        phase=phase,
        num_layers=num_layers,
        wl=830e-9,
        pixel_size=2e-6,
        distance=distance,
        detector_pos=detector_pos,
        phase_mapping="sigmoid",
        quantize_during_train=False,
        quant_levels=4,
        device=device,
    ).to(device)
    return model


def set_trainable_phase_only(model):
    for p in model.parameters():
        p.requires_grad = False

    for name, p in model.named_parameters():
        if name.startswith("phase_"):
            p.requires_grad_(True)

    params_to_update = [p for p in model.parameters() if p.requires_grad]
    assert len(params_to_update) > 0, "No trainable phase parameters found."
    assert len({str(p.device) for p in params_to_update}) == 1, "Trainable parameters cross devices."
    return params_to_update


def train(
    model,
    loss_function,
    optimizer,
    trainloader,
    testloader,
    epochs,
    device,
    output_dir,
    num_classes,
):
    history = []
    best_acc = -1.0
    best_model_wts = copy.deepcopy(model.state_dict())
    best_path = output_dir / "best.pt"

    opt_devices = {str(p.device) for g in optimizer.param_groups for p in g["params"]}
    if len(opt_devices) != 1:
        raise RuntimeError(f"Optimizer parameters are on mixed devices: {opt_devices}")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for images, labels in tqdm(trainloader, desc=f"Train {epoch + 1}/{epochs}"):
            images = torch.sqrt(images.to(device).squeeze(1))
            labels = labels.to(device).long()

            optimizer.zero_grad()
            out_label, _ = model(images)
            out_label = out_label.to(dtype=torch.float32)

            loss = loss_function(out_label, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(out_label, 1)
            running_correct += (predicted == labels).sum().item()
            running_total += labels.size(0)

        epoch_train_loss = running_loss / max(1, len(trainloader))
        epoch_train_acc = running_correct / max(1, running_total)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        pred_hist = torch.zeros(num_classes, dtype=torch.int64)

        with torch.no_grad():
            for images, labels in tqdm(testloader, desc=f"Val   {epoch + 1}/{epochs}"):
                images = torch.sqrt(images.to(device).squeeze(1))
                labels = labels.to(device).long()

                out_label, _ = model(images)
                out_label = out_label.to(dtype=torch.float32)

                loss = loss_function(out_label, labels)
                val_loss += loss.item()

                _, predicted = torch.max(out_label, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

                pred_cpu = predicted.detach().cpu()
                for c in range(num_classes):
                    pred_hist[c] += (pred_cpu == c).sum()

        epoch_val_loss = val_loss / max(1, len(testloader))
        epoch_val_acc = val_correct / max(1, val_total)
        pred_hist_list = pred_hist.tolist()

        row = {
            "epoch": epoch + 1,
            "train_loss": epoch_train_loss,
            "val_loss": epoch_val_loss,
            "train_acc": epoch_train_acc,
            "val_acc": epoch_val_acc,
            "val_pred_hist": pred_hist_list,
        }
        history.append(row)

        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            state = {
                "state_dict": model.state_dict(),
                "best_acc": best_acc,
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "history": history,
            }
            torch.save(state, best_path)

        print(
            f"Epoch={epoch + 1}/{epochs} "
            f"train_loss={epoch_train_loss:.4f}, val_loss={epoch_val_loss:.4f}"
        )
        print(f"train_acc={epoch_train_acc:.4f}, val_acc={epoch_val_acc:.4f}")
        print("val pred hist:", pred_hist_list)
        print("-----------------------")

    model.load_state_dict(best_model_wts)
    save_history_csv(history, output_dir / "training_history.csv")
    save_curves(history, output_dir / "training_curves.png")
    return model, history, best_path


def save_history_csv(history, path):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "val_loss", "train_acc", "val_acc", "val_pred_hist"],
        )
        writer.writeheader()
        for row in history:
            out = dict(row)
            out["val_pred_hist"] = " ".join(str(x) for x in row["val_pred_hist"])
            writer.writerow(out)


def save_curves(history, path):
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    train_acc = [row["train_acc"] for row in history]
    val_acc = [row["val_acc"] for row in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    ax1.plot(epochs, train_loss, label="train_loss")
    ax1.plot(epochs, val_loss, label="val_loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_acc, label="train_acc")
    ax2.plot(epochs, val_acc, label="val_acc")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_sample_outputs(model, dataloader, device, output_dir, sample_count, detector_pos, classes):
    model.eval()
    sample_dir = output_dir / "samples"
    saved = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images_device = torch.sqrt(images.to(device).squeeze(1))
            logits, intensity = model(images_device)
            probs = torch.softmax(logits.float(), dim=1)
            pred = torch.argmax(logits, dim=1)

            for b in range(images.shape[0]):
                if saved >= sample_count:
                    return

                input_img = images[b, 0].detach().cpu().numpy()
                out_int = intensity[b].detach().cpu().numpy()
                det_energy = logits[b].detach().cpu().numpy()
                true_label = int(labels[b].item())
                pred_label = int(pred[b].item())

                fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
                axes[0].imshow(input_img, cmap="gray", vmin=0, vmax=1)
                axes[0].set_title(f"input true={true_label}")
                axes[0].axis("off")

                axes[1].imshow(out_int, cmap="inferno")
                axes[1].set_title(f"output intensity pred={pred_label}")
                axes[1].axis("off")

                axes[2].bar(np.arange(len(det_energy)), det_energy)
                axes[2].set_xticks(np.arange(len(det_energy)))
                axes[2].set_xticklabels([str(i) for i in range(len(det_energy))])
                axes[2].set_title("detector energy")
                axes[2].set_xlabel("class")
                axes[2].set_ylabel("sum intensity")

                fig.savefig(sample_dir / f"sample_{saved:02d}.png", dpi=150)
                plt.close(fig)

                np.save(sample_dir / f"sample_{saved:02d}_input.npy", input_img.astype(np.float32))
                np.save(sample_dir / f"sample_{saved:02d}_output_intensity.npy", out_int.astype(np.float32))
                np.save(sample_dir / f"sample_{saved:02d}_detector_energy.npy", det_energy.astype(np.float32))
                np.save(sample_dir / f"sample_{saved:02d}_detector_prob.npy", probs[b].detach().cpu().numpy().astype(np.float32))
                saved += 1


def collect_phase_map(model):
    params = dict(model.named_parameters())
    phase_keys = sorted(
        [k for k in params.keys() if k.startswith("phase_")],
        key=lambda x: int(x.split("_", 1)[1]),
    )
    phases = []
    for key in phase_keys:
        raw = params[key].detach().cpu()
        phi = np.pi * torch.sigmoid(raw)
        phases.append(phi.numpy().astype(np.float32))
    return np.stack(phases, axis=0)


T_OXIDE = 676.6e-9


def quantize_phase_4level(phi_rad):
    levels = np.array([0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], dtype=np.float32)
    idx = np.argmin(np.abs(phi_rad[..., None] - levels), axis=-1)
    return levels[idx]


def phase_to_etch_depth(phi_rad, t_oxide=T_OXIDE, phi_max=3 * np.pi / 4):
    phi_clamped = np.clip(phi_rad, 0, phi_max)
    return t_oxide * (1.0 - phi_clamped / phi_max)


def export_phase_height_distance(model, output_dir):
    phase_map = collect_phase_map(model)
    phase_quantized = quantize_phase_4level(phase_map)
    height_map = phase_to_etch_depth(phase_quantized).astype(np.float32)

    np.save(output_dir / "phase_map.npy", phase_map.astype(np.float32))
    np.save(output_dir / "height_map.npy", height_map)

    distances = []
    num_layers = phase_map.shape[0]
    for i in range(num_layers + 2):
        d = getattr(model, f"distance_{i}").detach().cpu().item()
        distances.append(d)
    np.save(output_dir / "trained_distances.npy", np.array(distances, dtype=np.float32))

    print("Saved:", output_dir / "phase_map.npy")
    print("Saved:", output_dir / "height_map.npy")
    print("Saved:", output_dir / "trained_distances.npy")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_output_dirs(output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using Device:", device)
    print("Output dir:", output_dir)
    print("Batch size:", args.batch_size)
    print("Epochs:", args.epochs)

    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders(
        args.data_root, args.batch_size, args.num_workers
    )
    num_classes = len(train_dataset.classes)
    class_names = ["fundamental", "first_order", "second_order", "third_order", "forth_order"]

    print("Using classes:", class_names)
    print("ImageFolder classes:", train_dataset.classes)
    print("train classes:", train_dataset.class_to_idx)
    print("val classes:", val_dataset.class_to_idx)
    print("full train samples:", len(train_dataset))
    print("full val samples:", len(val_dataset))

    sample_images, sample_labels = next(iter(train_loader))
    print("sample batch images shape:", tuple(sample_images.shape))
    print("sample batch labels shape:", tuple(sample_labels.shape))
    print("sample image min/max:", float(sample_images.min()), float(sample_images.max()))

    detector_pos = build_detector_pos()
    print("Detector positions (up, down, left, right):")
    for i, p in enumerate(detector_pos):
        print(f"  det[{i}] = {p}")

    model = initialize_model(device, detector_pos)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    params_to_update = set_trainable_phase_only(model)

    print("Params to learn:")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print("  ", name)
    print("num trainable params =", len(params_to_update))
    print("trainable devices =", {str(p.device) for p in params_to_update})

    optimizer = torch.optim.Adam(params_to_update, lr=args.lr)

    with torch.no_grad():
        quick_images, _ = next(iter(train_loader))
        quick_E = torch.sqrt(quick_images.to(device).squeeze(1))
        quick_logits, quick_Int = model(quick_E)
        print("field input after sqrt shape:", tuple(quick_E.shape))
        print("logits shape:", tuple(quick_logits.shape))
        print("output intensity shape:", tuple(quick_Int.shape))

    best_model, history, best_path = train(
        model,
        criterion,
        optimizer,
        train_loader,
        val_loader,
        epochs=args.epochs,
        device=device,
        output_dir=output_dir,
        num_classes=num_classes,
    )

    print("Best checkpoint:", best_path)
    save_sample_outputs(
        best_model,
        val_loader,
        device,
        output_dir,
        args.sample_count,
        detector_pos,
        train_dataset.classes,
    )
    export_phase_height_distance(best_model, output_dir)
    print("baseline training finished")


if __name__ == "__main__":
    main()

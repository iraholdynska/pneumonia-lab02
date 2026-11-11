"""Спільний код для 05_improvements.ipynb.

Завантаження даних, передобробка (CLAHE), аугментація, побудова моделей,
навчання з налаштуваннями, метрики, Grad-CAM.
"""
import copy
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision import transforms as T
from skimage import exposure
from tqdm import tqdm

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

DATA_DIR = Path("../data/chest_xray")
RESULTS_DIR = Path("../results")
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def load_data():
    df = pd.read_csv(RESULTS_DIR / "splits.csv")
    df["y"] = (df["label"] == "PNEUMONIA").astype(int)

    X = np.load(DATA_DIR.parent / "cache_224.npz")["X"]
    assert len(X) == len(df), "cache_224.npz і splits.csv не збігаються"

    tr = (df["new_split"] == "train").values
    va = (df["new_split"] == "val").values
    te = (df["new_split"] == "test").values
    return df, X, tr, va, te, df["y"].values


def make_clahe(X, cache_name="cache_224_clahe.npz", clip_limit=0.02):
    path = DATA_DIR.parent / cache_name
    if path.exists():
        return np.load(path)["X"]
    out = np.empty_like(X)
    for i in tqdm(range(len(X)), desc="CLAHE"):
        out[i] = (exposure.equalize_adapthist(X[i], clip_limit=clip_limit) * 255).astype(np.uint8)
    np.savez_compressed(path, X=out)
    return out


def get_augment(strong=True):
    if not strong:
        return T.Compose([
            T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            T.ColorJitter(brightness=0.2, contrast=0.2),
        ])
    return T.Compose([
        T.RandomResizedCrop(224, scale=(0.75, 1.0), ratio=(0.9, 1.1), antialias=True),
        T.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        T.ColorJitter(brightness=0.4, contrast=0.4),
    ])


class XrayDS(Dataset):
    def __init__(self, X, y, transform=None, per_image_norm=False):
        self.X, self.y = X, y
        self.transform, self.per_image_norm = transform, per_image_norm

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = torch.from_numpy(self.X[i]).float().div(255).unsqueeze(0) 
        if self.transform is not None:
            x = self.transform(x)
        if self.per_image_norm:
            x = ((x - x.mean()) / (x.std() + 1e-6) * 0.25 + 0.5).clamp(0, 1)
        x = x.repeat(3, 1, 1)
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        return x, torch.tensor(self.y[i], dtype=torch.float32)


def build_model(arch="resnet18", unfreeze="last1", pretrained=True):
    if arch == "resnet18":
        w = models.ResNet18_Weights.DEFAULT if pretrained else None
        m = models.resnet18(weights=w)
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 1))
        head = m.fc
        blocks = {"head": [], "last1": [m.layer4], "last2": [m.layer3, m.layer4]}
    elif arch == "densenet121":
        w = models.DenseNet121_Weights.DEFAULT if pretrained else None
        m = models.densenet121(weights=w)
        m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1024, 1))
        head = m.classifier
        f = m.features
        blocks = {"head": [],
                  "last1": [f.denseblock4, f.norm5],
                  "last2": [f.denseblock3, f.transition3, f.denseblock4, f.norm5]}
    elif arch == "efficientnet_b0":
        w = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        m = models.efficientnet_b0(weights=w)
        m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1280, 1))
        head = m.classifier
        blocks = {"head": [],
                  "last1": [m.features[7], m.features[8]],
                  "last2": [m.features[6], m.features[7], m.features[8]]}
    else:
        raise ValueError(f"Невідома архітектура: {arch}")

    for p in m.parameters():
        p.requires_grad = False
    for p in head.parameters():
        p.requires_grad = True
    for blk in blocks[unfreeze]:
        for p in blk.parameters():
            p.requires_grad = True
    return m, head


def set_train_mode(model):
    model.train()
    for mod in model.modules():
        if isinstance(mod, nn.modules.batchnorm._BatchNorm) and not any(p.requires_grad for p in mod.parameters()):
            mod.eval()


def load_trained(arch, path):
    m, _ = build_model(arch, "head", pretrained=False)
    m.load_state_dict(torch.load(path, map_location=device))
    return m.to(device).eval()


@torch.no_grad()
def predict(model, loader):
    model.eval()
    probs, ys = [], []
    for xb, yb in loader:
        probs.append(torch.sigmoid(model(xb.to(device)).squeeze(1)).cpu().numpy())
        ys.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(ys).astype(int)


def get_probs(model, X, y, mask, per_image_norm=False, batch_size=64):
    loader = DataLoader(XrayDS(X[mask], y[mask], None, per_image_norm), batch_size=batch_size)
    return predict(model, loader)


def compute_metrics(y_true, probs, threshold=0.5):
    pred = (probs >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "specificity": recall_score(y_true, pred, pos_label=0),
        "f1": f1_score(y_true, pred),
        "auc": roc_auc_score(y_true, probs),
    }


def run_experiment(name, arch, unfreeze, X, y, tr, va, te, *,
                   strong_aug=True, per_image_norm=False, use_weights=True,
                   label_smoothing=0.0, epochs=10, patience=4,
                   lr_head=1e-3, lr_backbone=1e-4, seed=SEED,
                   pretrained=True, save=True):
    set_seed(seed)

    train_ds = XrayDS(X[tr], y[tr], get_augment(strong_aug) if strong_aug is not None else None, per_image_norm)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(XrayDS(X[va], y[va], None, per_image_norm), batch_size=64)
    test_loader = DataLoader(XrayDS(X[te], y[te], None, per_image_norm), batch_size=64)

    model, head = build_model(arch, unfreeze, pretrained)
    model = model.to(device)

    head_ids = {id(p) for p in head.parameters()}
    backbone_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    groups = [{"params": list(head.parameters()), "lr": lr_head}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr_backbone})
    optimizer = torch.optim.Adam(groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if use_weights:
        pw = torch.tensor([(y[tr] == 0).sum() / (y[tr] == 1).sum()], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    else:
        criterion = nn.BCEWithLogitsLoss()

    history = {"train_loss": [], "val_loss": [], "val_f1": []}
    best_f1, best_state, best_epoch, bad_epochs = -1.0, None, 0, 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        set_train_mode(model)
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            target = yb * (1 - label_smoothing) + 0.5 * label_smoothing   
            optimizer.zero_grad()
            loss = criterion(model(xb).squeeze(1), target)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(xb)
        scheduler.step()
        train_loss = running / len(train_ds)

        val_probs, val_y = predict(model, val_loader)
        eps = 1e-7
        val_loss = float(-np.mean(val_y * np.log(val_probs + eps) + (1 - val_y) * np.log(1 - val_probs + eps)))
        val_f1 = f1_score(val_y, (val_probs >= 0.5).astype(int))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        print(f"[{name}] епоха {epoch:2d} | train {train_loss:.4f} | val loss {val_loss:.4f} | "
              f"val F1 {val_f1:.4f} | {time.time() - t0:.0f} с")

        if val_f1 > best_f1:
            best_f1, best_epoch, bad_epochs = val_f1, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Рання зупинка: {patience} епох без покращення")
                break

    model.load_state_dict(best_state)
    if save:
        torch.save(best_state, RESULTS_DIR / f"{name}.pth")

    val_probs, val_y = predict(model, val_loader)
    test_probs, test_y = predict(model, test_loader)
    return {
        "name": name, "arch": arch, "history": history, "best_epoch": best_epoch,
        "per_image_norm": per_image_norm,
        "val": compute_metrics(val_y, val_probs), "test": compute_metrics(test_y, test_probs),
        "val_probs": val_probs, "val_y": val_y, "test_probs": test_probs, "test_y": test_y,
    }


def grad_cam(model, x, layer):
    store = {}

    def fwd_hook(module, inputs, output):
        store["act"] = output
        output.register_hook(lambda g: store.__setitem__("grad", g))

    handle = layer.register_forward_hook(fwd_hook)
    model.eval()
    model.zero_grad()
    x = x.to(device).requires_grad_(True)          
    logit = model(x).squeeze(1)
    logit.sum().backward()
    handle.remove()

    w = store["grad"].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * store["act"]).sum(dim=1, keepdim=True))
    cam = torch.nn.functional.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam[0, 0].detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam, float(torch.sigmoid(logit).item())

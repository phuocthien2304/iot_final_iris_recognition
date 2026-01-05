# train_iris_cnn.py
# CNN training pipeline for iris recognition
import argparse
import os
import sys

import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils.metrics import compute_far_frr, evaluate_model
from utils.model import (
    ImprovedIrisCNN,
    SimpleIrisCNN,
)


class CosineClassifier(nn.Module):
    def __init__(self, in_features, num_classes, scale=30.0, margin=0.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.margin = margin

    def forward(self, features, labels=None):
        weight_norm = torch.nn.functional.normalize(self.weight, dim=1)
        feats_norm = torch.nn.functional.normalize(features, dim=1)
        cosine = torch.matmul(feats_norm, weight_norm.t())
        if labels is not None and self.margin > 0:
            margin_tensor = torch.zeros_like(cosine)
            margin_tensor[torch.arange(features.size(0)), labels] = self.margin
            cosine = cosine - margin_tensor
        return self.scale * cosine


def _strip_classifier(backbone):
    """Remove final linear layer to expose embedding features."""
    if not hasattr(backbone, "c") or not isinstance(backbone.c, nn.Sequential):
        raise ValueError("Backbone does not expose expected classifier sequence")

    modules = list(backbone.c.children())
    if not modules or not isinstance(modules[-1], nn.Linear):
        raise ValueError("Expected last module of backbone classifier to be nn.Linear")

    last_linear = modules.pop()
    feature_dim = last_linear.in_features
    backbone.c = nn.Sequential(*modules)
    return feature_dim


class IrisNet(nn.Module):
    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x, labels=None):
        features = self.backbone(x)
        return self.classifier(features, labels=labels)

    def forward_features(self, x):
        return self.backbone(x)

# --- Path Setup ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_ROOT = os.path.join(
    PROJECT_ROOT,
    "data",
    "casia-iris-preprocessed",
    "CASIA_thousand_norm_256_64_e_nn_stacked",
)
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
# --- End Path Setup ---

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument("--arch", choices=["simple", "improved"], default="simple")
parser.add_argument("--augment", choices=["none", "basic", "strong"], default="strong")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--label_smoothing", type=float, default=0.05)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument(
    "--data-root",
    type=str,
    default=DEFAULT_DATA_ROOT,
    help="Path to preprocessed CASIA dataset (expects train/val[/test] subfolders)",
)
parser.add_argument("--num-workers", type=int, default=0)
parser.add_argument("--debug-dataloader-only", action="store_true",
                    help="Only iterate dataloader to trigger preprocess debug saving, skip training")
parser.add_argument("--debug-batches", type=int, default=1,
                    help="When --debug-dataloader-only, number of train batches to iterate")
args = parser.parse_args()
IMG_SIZE = (128,128); BATCH_SIZE = args.batch; EPOCHS = args.epochs; LR = args.lr

if not os.path.isdir(args.data_root):
    raise FileNotFoundError(f"[ERROR] Dataset root not found: {args.data_root}")


class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        path = self.samples[index][0]
        return img, target, {"path": path, "label": self.classes[target]}


def _build_transforms(img_size, augment_mode, is_train):
    ops = [transforms.Resize(img_size)]
    if is_train:
        if augment_mode == "basic":
            ops.append(transforms.RandomHorizontalFlip(p=0.5))
        elif augment_mode == "strong":
            ops.extend(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.98, 1.02)),
                    transforms.ColorJitter(brightness=0.15, contrast=0.15),
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
                ]
            )
    ops.extend(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    return transforms.Compose(ops)


def _align_eval_class_indices(train_ds, eval_ds):
    """Ensure eval dataset shares the same class_to_idx mapping as train."""
    ref_map = train_ds.class_to_idx
    ref_classes = train_ds.classes

    remapped_samples = []
    missing_classes = set()
    for path, _ in eval_ds.samples:
        class_name = os.path.basename(os.path.dirname(path))
        if class_name not in ref_map:
            missing_classes.add(class_name)
            continue
        remapped_samples.append((path, ref_map[class_name]))

    if missing_classes:
        missing_str = ", ".join(sorted(missing_classes)[:5])
        raise ValueError(
            "[ERROR] Evaluation split contains classes absent from training split: "
            f"{missing_str}{' ...' if len(missing_classes) > 5 else ''}"
        )

    eval_ds.samples = remapped_samples
    eval_ds.targets = [target for _, target in remapped_samples]
    eval_ds.class_to_idx = ref_map
    eval_ds.classes = ref_classes


def load_casia_dataset(data_root, img_size, batch_size, augment_mode, num_workers):
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    test_dir = os.path.join(data_root, "test")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"[ERROR] Expected train split at {train_dir}")

    eval_dir = val_dir if os.path.isdir(val_dir) else test_dir
    if eval_dir is None or not os.path.isdir(eval_dir):
        raise FileNotFoundError(
            f"[ERROR] Could not find validation/test split under {data_root}. Expected 'val' or 'test' folder."
        )

    train_tf = _build_transforms(img_size, augment_mode, is_train=True)
    eval_tf = _build_transforms(img_size, augment_mode="none", is_train=False)

    train_ds = ImageFolderWithPaths(train_dir, transform=train_tf)
    eval_ds = ImageFolderWithPaths(eval_dir, transform=eval_tf)
    _align_eval_class_indices(train_ds, eval_ds)

    common_loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": max(0, int(num_workers)),
        "pin_memory": torch.cuda.is_available(),
    }

    train_loader = DataLoader(train_ds, shuffle=True, **common_loader_kwargs)
    eval_loader = DataLoader(eval_ds, shuffle=False, **common_loader_kwargs)

    class_names = train_ds.classes
    return train_loader, eval_loader, len(class_names), class_names, eval_dir


train_loader, test_loader, num_classes, class_names, eval_dir = load_casia_dataset(
    args.data_root,
    IMG_SIZE,
    BATCH_SIZE,
    args.augment,
    args.num_workers,
)

print(f"[INFO] Loaded CASIA dataset: train={len(train_loader.dataset)}, eval={len(test_loader.dataset)} from {eval_dir}")

if args.debug_dataloader_only:
    max_batches = max(1, int(args.debug_batches))
    processed = 0
    for batch_idx, (x, y, meta) in enumerate(train_loader):
        processed += len(x)
        if batch_idx + 1 >= max_batches:
            break
    print(f"[DEBUG] Iterated {batch_idx + 1} batches ({processed} samples). Exiting without training.")
    sys.exit(0)

model_map = {
    "simple": SimpleIrisCNN,
    "improved": ImprovedIrisCNN,
}
model_cls = model_map[args.arch]
backbone = model_cls(num_classes)
feature_dim = _strip_classifier(backbone)
cosine_head = CosineClassifier(feature_dim, num_classes, scale=30.0, margin=0.2)
model = IrisNet(backbone, cosine_head).to(DEVICE)
criterion = nn.CrossEntropyLoss(label_smoothing=float(args.label_smoothing))
opt = optim.Adam(model.parameters(), lr=LR, weight_decay=float(args.weight_decay))
scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

train_losses=[]; val_losses=[]; train_accs=[]; val_accs=[]; best_acc=0
for e in range(1,EPOCHS+1):
    model.train(); tot=0; corr=0; loss_sum=0
    for x,y,_ in train_loader:
        x,y=x.to(DEVICE),y.to(DEVICE).long()
        opt.zero_grad(); out=model(x, labels=y); loss=criterion(out,y); loss.backward(); opt.step()
        loss_sum+=loss.item()*y.size(0); corr+=(out.argmax(1)==y).sum().item(); tot+=y.size(0)
    train_losses.append(loss_sum/tot); train_accs.append(corr/tot)
    vloss,vacc,vlog,vlabel,_=evaluate_model(model,test_loader,DEVICE)
    val_losses.append(vloss); val_accs.append(vacc)
    print(f"Epoch {e}/{EPOCHS} TrainAcc={corr/tot:.3f} ValAcc={vacc:.3f}")
    if vacc>best_acc:
        best_acc=vacc
        base_name = {
            "simple": "best_iris_cnn.pth",
            "improved": "best_iris_cnn_improved.pth",
        }[args.arch]
        torch.save(model.state_dict(), os.path.join(OUTPUTS_DIR, base_name))
    scheduler.step()

eer,thr,FARs,FRRs,ths=compute_far_frr(vlog,vlabel)
print(f"EER={eer:.3f}@{thr:.2f}")

with open(os.path.join(OUTPUTS_DIR, "label_encoder.pkl"),"wb") as f:
    pickle.dump({"class_names": class_names}, f)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1);plt.plot(train_losses,label="Train");plt.plot(val_losses,label="Val");plt.legend();plt.title("Loss")
plt.subplot(1,2,2);plt.plot(train_accs,label="Train");plt.plot(val_accs,label="Val");plt.legend();plt.title("Accuracy")
plt.tight_layout();plt.savefig(os.path.join(OUTPUTS_DIR, "training_curves.png"))

plt.figure();plt.plot(ths,FARs,label="FAR");plt.plot(ths,FRRs,label="FRR")
plt.axvline(thr,ls="--",label=f"EER@{thr:.2f}");plt.legend();plt.savefig(os.path.join(OUTPUTS_DIR, "far_frr_curve.png"))
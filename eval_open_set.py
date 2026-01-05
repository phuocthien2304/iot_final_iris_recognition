import json
import pathlib
from collections import Counter
import os
import argparse
import csv
import matplotlib.pyplot as plt

import torch
import numpy as np

from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import DenseNet161Iris, ResNet101Iris, InceptionV3Iris


def get_model(model_name, checkpoint_path, num_classes=1500):

    model = None
    input_size = 0

    if model_name == "resnet101":
        model = ResNet101Iris(num_classes=num_classes)
        input_size = 224
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))

    elif model_name == "densenet161":
        model = DenseNet161Iris(num_classes=num_classes)
        input_size = 224
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))

    elif model_name == "inception":
        model = InceptionV3Iris(num_classes=num_classes)
        input_size = 299
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))

    else:
        print("Invalid model name, exiting...")
        exit()

    return model, input_size

def get_dataloader(data_path, input_size, batch_size=32, num_workers=4):

    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(data_path, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

def enroll_identities(feature_extract_func, dataloader, device):
    enrolled = {}
    with torch.no_grad():
        for input, labels in dataloader:
            inputs = input.to(device)
            labels = labels.cpu().detach().numpy()

            # Extract the features using the CNN
            predictions = feature_extract_func(inputs).cpu().detach().numpy()

            # Create a matrix for each users, where a row represents a feature vector extracted from the enrollment image and
            # normalize the matrix bx rows (to reduce the amount of computation in the recognition phase)
            # Results is a dictionary, where a key is a specific identity with the entry containing a matrix where a row is x / ||x||,
            # where x is a feature vector for a given image
            unique_labels = np.unique(labels)
            for i in unique_labels:
                user_features = predictions[labels == i, :]
                if i in enrolled:
                    enrolled[i] = np.vstack((enrolled[i], normalize(user_features, axis=1, norm='l2')))
                else:
                    enrolled[i] = normalize(user_features, axis=1, norm='l2')

    return enrolled

def evaluate(enrolled, feature_extract_func, dataloader, device, rank_n=50, return_preds=False, return_details=False):
    total = 0
    rank_n_correct = np.zeros(rank_n)
    preds_top1 = []
    best_sims = []
    labels_out = []

    with torch.no_grad():
        for input, labels in dataloader:
            inputs = input.to(device)
            labels = labels.cpu().detach().numpy()
            predictions = feature_extract_func(inputs).cpu().detach().numpy()
            for idx, label in enumerate(labels):
                pred = predictions[idx, :].reshape(-1, 1)
                pred_norm = normalize(pred, axis=0, norm="l2")
                similarities_id = {}
                for key in enrolled.keys():
                    cosine_similarities = np.matmul(enrolled[key], pred_norm)
                    similarities_id[key] = np.max(cosine_similarities)

                counter = Counter(similarities_id)
                if return_preds:
                    preds_top1.append(list(dict(counter.most_common(1)).keys())[0])
                if return_details:
                    best_sims.append(float(max(similarities_id.values())))
                    labels_out.append(int(label))
                for i in range(1, rank_n + 1):
                    rank_n_vals = list(dict(counter.most_common(i)).keys())
                    rank_n_correct[i-1] += 1 if label in rank_n_vals else 0
                total +=1

    rank_n_correct /= total
    rank_1_accuracy =  rank_n_correct[0]
    rank_5_accuracy = rank_n_correct[4]
    print(f"Rank 1 accuracy: {rank_1_accuracy}, rank 5 accuracy: {rank_5_accuracy}")
    if return_details:
        return rank_1_accuracy, rank_5_accuracy, rank_n_correct, preds_top1, best_sims, labels_out
    if return_preds:
        return rank_1_accuracy, rank_5_accuracy, rank_n_correct, preds_top1
    return rank_1_accuracy, rank_5_accuracy, rank_n_correct

def _compute_far_frr(best_sims, genuine_mask, thr):
    best_sims = np.asarray(best_sims)
    genuine_mask = np.asarray(genuine_mask).astype(bool)
    accept_mask = best_sims >= thr
    n_genuine = int(genuine_mask.sum())
    n_impostor = int((~genuine_mask).sum())
    frr = float((~accept_mask & genuine_mask).sum()) / n_genuine if n_genuine > 0 else 0.0
    far = float((accept_mask & (~genuine_mask)).sum()) / n_impostor if n_impostor > 0 else 0.0
    tpr = 1.0 - frr
    tnr = 1.0 - far
    return far, frr, tpr, tnr

def _sweep_thresholds(best_sims, genuine_mask, start=0.5, end=0.99, step=0.01):
    thrs = []
    fars = []
    frrs = []
    tprs = []
    tnrs = []
    t = start
    # ensure numeric stability
    while t <= end + 1e-9:
        far, frr, tpr, tnr = _compute_far_frr(best_sims, genuine_mask, t)
        thrs.append(round(float(t), 6))
        fars.append(far)
        frrs.append(frr)
        tprs.append(tpr)
        tnrs.append(tnr)
        t += step
    # EER estimation: choose thr with minimal |FAR-FRR|
    diffs = np.abs(np.array(fars) - np.array(frrs))
    idx = int(diffs.argmin())
    eer_thr = thrs[idx]
    eer_val = float((fars[idx] + frrs[idx]) / 2.0)
    return {
        'thresholds': thrs,
        'FARs': fars,
        'FRRs': frrs,
        'TPRs': tprs,
        'TNRs': tnrs,
        'EER': {'threshold': eer_thr, 'value': eer_val}
    }

def _save_cm_csv(cm, labels, path):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([''] + list(labels))
        for i, row in enumerate(cm):
            writer.writerow([labels[i]] + list(map(int, row)))

def _plot_cm(cm, labels, path, title, vmax=None):
    if len(labels) == 0:
        return
    fig, ax = plt.subplots(figsize=(min(12, 0.28*len(labels)+2), min(10, 0.22*len(labels)+2)), dpi=140)
    im = ax.imshow(cm, interpolation='nearest', cmap='Reds', vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=6, rotation=90)
    ax.set_yticklabels(labels, fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

if __name__ == '__main__':


    print("Loading model...")
    checkpoint_path = r"D:\wordspace\IoT Final\iris-recognition-cnn\models\resnet101_e_80_lr_2e-05_best.pth"
    model_name = "resnet101"
    enrollment_data_path = r"D:\wordspace\IoT Final\iris-recognition-cnn\data\casia-iris-preprocessed\CASIA_thousand_norm_256_64_e_nn_open_set_stacked\enrollment"
    test_data_path = r"D:\wordspace\IoT Final\iris-recognition-cnn\data\casia-iris-preprocessed\CASIA_thousand_norm_256_64_e_nn_open_set_stacked\test"
    batch_size = 196
    default_thr = 0.90
    default_num_workers = 4

    parser = argparse.ArgumentParser(description="Open-set evaluation with FAR/FRR and threshold sweep")
    parser.add_argument('--checkpoint', default=checkpoint_path)
    parser.add_argument('--arch', default=model_name, choices=['resnet101','densenet161','inception'])
    parser.add_argument('--enroll', default=enrollment_data_path)
    parser.add_argument('--test', default=test_data_path)
    parser.add_argument('--batch-size', type=int, default=batch_size)
    parser.add_argument('--thr', type=float, default=default_thr)
    parser.add_argument('--num-workers', type=int, default=default_num_workers)
    parser.add_argument('--sweep-start', type=float, default=0.50)
    parser.add_argument('--sweep-end', type=float, default=0.99)
    parser.add_argument('--sweep-step', type=float, default=0.01)
    parser.add_argument('--save-prefix', default='results')
    parser.add_argument('--print-preds', action='store_true', help='Print per-image top-1 predictions')
    parser.add_argument('--plot', action='store_true', help='Save DET plot (FAR vs FRR) to <save-prefix>/<model>_det.png')
    parser.add_argument('--cm-splits', type=int, default=4, help='Number of progressive confusion matrices (e.g., 4 => 25%,50%,75%,100%)')
    parser.add_argument('--cm-max-labels', type=int, default=80, help='Max number of labels to render as PNG; always save CSV')
    args = parser.parse_args()

    def _norm(p: str) -> str:
        if p is None:
            return p
        # strip quotes/spaces and normalize
        p2 = p.strip().strip('"').strip("'")
        return os.path.normpath(os.path.expanduser(p2))

    checkpoint_path = _norm(args.checkpoint)
    model_name = args.arch
    enrollment_data_path = _norm(args.enroll)
    test_data_path = _norm(args.test)
    batch_size = args.batch_size
    thr = args.thr
    num_workers = args.num_workers
    sweep_start = args.sweep_start
    sweep_end = args.sweep_end
    sweep_step = args.sweep_step
    save_prefix = args.save_prefix
    print_preds = args.print_preds
    plot_flag = args.plot
    cm_splits = max(1, int(args.cm_splits))
    cm_max_labels = int(args.cm_max_labels)

    # Validate directories early
    def _validate_dir(path):
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Directory not found: {path}")
        subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        if len(subdirs) == 0:
            raise RuntimeError(f"No class subfolders found under: {path}")
        return subdirs

    print(f"Enrollment dir: {repr(enrollment_data_path)}")
    print(f"Test dir      : {repr(test_data_path)}")
    _validate_dir(enrollment_data_path)
    _validate_dir(test_data_path)

    model, input_size = get_model(model_name, checkpoint_path)

    # device = torch.device('cuda')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    


    enrollment_dataloader = get_dataloader(enrollment_data_path, input_size, batch_size=batch_size, num_workers=num_workers)
    test_dataloader = get_dataloader(test_data_path, input_size, batch_size=batch_size, num_workers=num_workers)

    print("Enrolling identities...")
    enrolled = enroll_identities(model.feature_extract_avg_pool, enrollment_dataloader, device)

    print("Running recognition evaluation...")
    rank_1_accuracy, rank_5_accuracy, rank_n_accuracy, preds_top1, best_sims, test_labels = evaluate(
        enrolled, model.feature_extract_avg_pool, test_dataloader, device, return_preds=True, return_details=True)

    enroll_class_names = enrollment_dataloader.dataset.classes
    test_class_names = test_dataloader.dataset.classes
    test_paths = [p for p, _ in test_dataloader.dataset.samples]
    if print_preds:
        for i, pred_idx in enumerate(preds_top1):
            print(f"{test_paths[i]} -> {enroll_class_names[pred_idx]}")

    # Compute FAR/FRR at a chosen threshold
    genuine_mask = []
    for i, lbl in enumerate(test_labels):
        true_name = test_class_names[lbl]
        is_genuine = true_name in enroll_class_names
        genuine_mask.append(is_genuine)
    far, frr, tpr, tnr = _compute_far_frr(best_sims, genuine_mask, thr)
    print(f"Threshold: {thr:.3f} | FAR: {far:.4f} | FRR: {frr:.4f}")

    # Threshold sweep and EER
    sweep = _sweep_thresholds(best_sims, genuine_mask, start=sweep_start, end=sweep_end, step=sweep_step)
    print(f"EER ~ {sweep['EER']['value']:.4f} at thr {sweep['EER']['threshold']:.3f}")

    # Optional DET plot (FAR vs FRR)
    det_path = None
    if plot_flag:
        det_path = f'{save_prefix}/{model_name}_det.png'
        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        ax.plot(sweep['FARs'], sweep['FRRs'], '-', color='C0', label='DET curve')
        ax.scatter([far], [frr], color='C1', s=20, label=f'Thr={thr:.2f}')
        eer_v = sweep['EER']['value']
        eer_t = sweep['EER']['threshold']
        ax.scatter([eer_v], [eer_v], color='C3', s=30, label=f'EER~{eer_v:.3f} @ {eer_t:.2f}')
        ax.set_xlabel('FAR')
        ax.set_ylabel('FRR')
        ax.set_title(f'DET curve - {model_name}')
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, which='both', linestyle='--', alpha=0.4)
        ax.legend()
        fig.tight_layout()
        fig.savefig(det_path)
        plt.close(fig)

    # Build names for confusion matrix (map open-set)
    y_true_names = []
    y_pred_names = []
    for i, lbl in enumerate(test_labels):
        true_name = test_class_names[lbl]
        if true_name in enroll_class_names:
            true_out = true_name
        else:
            true_out = 'IMPOSTOR'
        pred_out = 'UNKNOWN' if best_sims[i] < thr else enroll_class_names[preds_top1[i]]
        y_true_names.append(true_out)
        y_pred_names.append(pred_out)

    # Label order: enrolled classes + UNKNOWN + IMPOSTOR
    cm_labels = list(enroll_class_names) + ['UNKNOWN', 'IMPOSTOR']

    # Helper to compute and save CM
    def _compute_and_save_cm(sub_n: int, suffix: str):
        y_t = y_true_names[:sub_n]
        y_p = y_pred_names[:sub_n]
        cm = confusion_matrix(y_t, y_p, labels=cm_labels)
        csv_path = f'{save_prefix}/{model_name}_cm_{suffix}.csv'
        _save_cm_csv(cm, cm_labels, csv_path)
        if len(cm_labels) <= cm_max_labels:
            # Full matrix
            png_path = f'{save_prefix}/{model_name}_cm_{suffix}.png'
            _plot_cm(cm, cm_labels, png_path, f'Confusion Matrix ({suffix})')
            # Errors-only (zero diagonal)
            cm_err = cm.copy()
            np.fill_diagonal(cm_err, 0)
            vmax = cm_err.max() if cm_err.size else None
            png_err = f'{save_prefix}/{model_name}_cm_{suffix}_errors.png'
            _plot_cm(cm_err, cm_labels, png_err, f'Errors only ({suffix})', vmax=vmax)

    # Save progressive CMs
    total_n = len(y_true_names)
    steps = [int(np.ceil(total_n * (i / cm_splits))) for i in range(1, cm_splits + 1)]
    for i, n in enumerate(steps, start=1):
        _compute_and_save_cm(n, f'step{i}')
    # Also save full explicitly
    _compute_and_save_cm(total_n, 'full')

    results = {
        "rank_1_acc": rank_1_accuracy,
        "rank_5_acc": rank_5_accuracy,
        "rank_n_accuracies": list(rank_n_accuracy),
        "predictions": [enroll_class_names[idx] for idx in preds_top1],
        "threshold": thr,
        "FAR": far,
        "FRR": frr,
        "TPR": tpr,
        "TNR": tnr,
        "sweep": {
            "start": sweep_start,
            "end": sweep_end,
            "step": sweep_step,
            "EER": sweep['EER']
        },
        "det_plot": det_path
    }

    pathlib.Path(save_prefix).mkdir(parents=True, exist_ok=True)

    with open(f'{save_prefix}/{model_name}_results.json', 'w') as f:
        json.dump(results, f)

    # Save sweep CSV
    sweep_csv = f'{save_prefix}/{model_name}_sweep.csv'
    with open(sweep_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['threshold','FAR','FRR','TPR','TNR'])
        for t, fa, fr, tp, tn in zip(sweep['thresholds'], sweep['FARs'], sweep['FRRs'], sweep['TPRs'], sweep['TNRs']):
            writer.writerow([t, fa, fr, tp, tn])

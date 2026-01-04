import os
import argparse
import pathlib
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import torch
import sys

# Ensure project root is on sys.path so we can import eval_open_set when running from scripts/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.normpath(os.path.join(CURRENT_DIR, os.pardir))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# Reuse core utilities from eval_open_set without running its __main__
from eval_open_set import (
    get_model,
    get_dataloader,
    enroll_identities,
    evaluate,
)


def _norm(p: str) -> str:
    if p is None:
        return p
    p2 = str(p).strip().strip('"').strip("'")
    return os.path.normpath(os.path.expanduser(p2))


def _choose_grid(n: int, preferred_cols: int = 0):
    # If user specifies columns > 0, use it; otherwise auto choose near-square with horizontal bias
    if preferred_cols and preferred_cols > 0:
        cols = int(preferred_cols)
    else:
        # Nice presets to match common sample sizes for easier visual inspection
        if n in (100, 400, 900):
            s = int(np.sqrt(n))
            return s, s
        # Large final set: prefer 30x50 style (rows fixed 30, widen horizontally)
        if n >= 1200:
            rows = 30
            cols = int(np.ceil(n / rows))
            return rows, cols

        root = int(np.ceil(np.sqrt(max(1, n))))
        cols = root
        # Try to widen a bit for a horizontal rectangle if many blanks on last row
        while cols < n and (n % cols) > (cols // 3):
            # stop if making too wide
            if cols >= root + 4:
                break
            cols += 1
    rows = int(np.ceil(n / cols)) if n > 0 else 1
    return rows, cols


def _save_error_grid(correct_flags, n, cols, path, title):
    vals = list(correct_flags[:n])
    rows, cols = _choose_grid(len(vals), cols)
    grid = -1 * np.ones((rows, cols), dtype=int)
    for i, ok in enumerate(vals):
        r = i // cols
        c = i % cols
        grid[r, c] = 1 if ok else 0
    grid_idx = grid + 1  # -1 -> 0 (white), 0 -> 1 (red), 1 -> 2 (blue)
    cmap = ListedColormap(['#ffffff', '#ff4d4d', '#6cc4ff'])
    # Keep cells square: figure size proportional to cols x rows
    cell = 0.45
    fig_w = max(4, cols * cell + 1.4)
    fig_h = max(3, rows * cell + 1.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    ax.imshow(grid_idx, cmap=cmap, vmin=0, vmax=2, interpolation='none', aspect='equal')
    # Black gridlines (clear)
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.7)

    # Axis numbering (like spreadsheet). Reduce density for large grids.
    if cols <= 15:
        x_ticks = np.arange(cols)
        x_labels = [str(i) for i in range(1, cols + 1)]
    else:
        step = 5
        x_ticks = np.arange(0, cols, step)
        x_labels = [str(i + 1) for i in range(0, cols, step)]
    if rows <= 15:
        y_ticks = np.arange(rows)
        y_labels = [str(i) for i in range(1, rows + 1)]
    else:
        step = 5
        y_ticks = np.arange(0, rows, step)
        y_labels = [str(i + 1) for i in range(0, rows, step)]

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels(x_labels, fontsize=7)
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.tick_params(axis='x', top=True, bottom=False, labeltop=True, labelbottom=False, length=0)
    ax.tick_params(axis='y', left=True, right=False, labelleft=True, length=0)

    # Put summary like the sample image
    ax.set_title(title, fontsize=10, pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser("Per-sample error grid (progressive)")
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--arch', required=True, choices=['resnet101','resnet152','densenet161','densenet201','inception'])
    parser.add_argument('--enroll', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('--batch-size', type=int, default=196)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--thr', type=float, default=0.90)
    parser.add_argument('--splits', type=int, default=4, help='Number of progressive steps (e.g. 4 => 25%,50%,75%,100%)')
    parser.add_argument('--grid-cols', type=int, default=0, help='0=auto near-square; >0 = fixed columns (e.g., 10 for 10x10 when N=100)')
    parser.add_argument('--save-prefix', default='results')
    args = parser.parse_args()

    checkpoint_path = _norm(args.checkpoint)
    model_name = args.arch
    enrollment_data_path = _norm(args.enroll)
    test_data_path = _norm(args.test)
    batch_size = args.batch_size
    num_workers = args.num_workers
    thr = args.thr
    splits = max(1, int(args.splits))
    grid_cols = int(args.grid_cols)
    if grid_cols < 0:
        grid_cols = 0
    save_prefix = args.save_prefix

    pathlib.Path(save_prefix).mkdir(parents=True, exist_ok=True)

    model, input_size = get_model(model_name, checkpoint_path)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    enrollment_dataloader = get_dataloader(enrollment_data_path, input_size, batch_size=batch_size, num_workers=num_workers)
    test_dataloader = get_dataloader(test_data_path, input_size, batch_size=batch_size, num_workers=num_workers)

    print('Enrolling identities...')
    enrolled = enroll_identities(model.feature_extract_avg_pool, enrollment_dataloader, device)

    print('Evaluating to collect per-sample predictions...')
    _, _, _, preds_top1, best_sims, test_labels = evaluate(
        enrolled, model.feature_extract_avg_pool, test_dataloader, device, return_preds=True, return_details=True
    )

    enroll_class_names = enrollment_dataloader.dataset.classes
    test_class_names = test_dataloader.dataset.classes

    # Open-set aware correctness flags: True if (known & correctly identified & above thr) OR (impostor & rejected)
    correct_flags = []
    for i, lbl in enumerate(test_labels):
        true_name = test_class_names[lbl]
        is_genuine = true_name in enroll_class_names
        if is_genuine:
            pred_ok = (best_sims[i] >= thr) and (enroll_class_names[preds_top1[i]] == true_name)
            correct_flags.append(pred_ok)
        else:
            # Impostor should be rejected
            correct_flags.append(best_sims[i] < thr)

    total_n = len(correct_flags)
    steps = [int(np.ceil(total_n * (i / splits))) for i in range(1, splits + 1)]

    # Save progressive grids
    summary_rows = []
    for idx, n in enumerate(steps, start=1):
        errors = int(n - int(np.sum(np.array(correct_flags[:n], dtype=int))))
        err_rate = float(errors / n) if n > 0 else 0.0
        grid_path = f'{save_prefix}/{model_name}_grid_step{idx}.png'
        _save_error_grid(correct_flags, n, grid_cols, grid_path, f'Per-sample grid (step {idx}/{splits}) - N={n}, errors={errors}')
        summary_rows.append([f'step{idx}', n, errors, err_rate])

    # Save full grid
    errors_full = int(total_n - int(np.sum(np.array(correct_flags, dtype=int))))
    err_rate_full = float(errors_full / total_n) if total_n > 0 else 0.0
    full_grid = f'{save_prefix}/{model_name}_grid_full.png'
    _save_error_grid(correct_flags, total_n, grid_cols, full_grid, f'Per-sample grid (full) - N={total_n}, errors={errors_full}')
    summary_rows.append(['full', total_n, errors_full, err_rate_full])

    # Milestone grids with fixed target shapes
    milestones = [100, 400, 900, total_n]
    for m in milestones:
        if m <= 0 or m > total_n:
            continue
        errors_m = int(m - int(np.sum(np.array(correct_flags[:m], dtype=int))))
        err_rate_m = float(errors_m / m) if m > 0 else 0.0
        grid_path = f'{save_prefix}/{model_name}_grid_N{m}.png'
        _save_error_grid(correct_flags, m, grid_cols, grid_path, f'N={m}, errors={errors_m}')
        summary_rows.append([f'N{m}', m, errors_m, err_rate_m])

    # CSV summary
    with open(f'{save_prefix}/{model_name}_grid_summary.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'N', 'errors', 'error_rate'])
        for row in summary_rows:
            writer.writerow(row)

    print('Saved grids and summary to', save_prefix)


if __name__ == '__main__':
    main()

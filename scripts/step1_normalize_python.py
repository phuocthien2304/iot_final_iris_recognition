"""
Step 1: Normalize raw CASIA-Iris-Thousand images using pure Python
Alternative to iris_segm_norm.exe - uses OpenCV HoughCircles + polar unwrap + CLAHE
Creates two intermediate datasets matching the preprocessed structure
"""
import argparse
import itertools
import math
import os
import pathlib
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import sys

sys.path.insert(0, os.path.abspath('.'))
from utils.utils import casia_train_val_test_split, casia_enrollment_test_split, parse_casia_thousand_filename


def _detect_circles(gray):
    """Detect pupil and iris circles using HoughCircles"""
    h, w = gray.shape[:2]
    pupil = None
    iris = None
    
    try:
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.5, minDist=max(1, h//8),
            param1=120, param2=15, minRadius=max(8, h//40), maxRadius=max(9, h//4)
        )
        if circles is not None:
            circles = np.around(circles[0]).astype(int)
            best = None
            best_mean = 1e9
            for x, y, r in circles:
                x, y, r = int(x), int(y), int(r)
                if r <= 0:
                    continue
                x0, x1 = max(0, x-r), min(w, x+r)
                y0, y1 = max(0, y-r), min(h, y+r)
                roi = gray[y0:y1, x0:x1]
                m = float(roi.mean()) if roi.size else 1e9
                if m < best_mean:
                    best_mean = m
                    best = (x, y, r)
            pupil = best
    except Exception:
        pass
    
    try:
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(1, h//8),
            param1=150, param2=30,
            minRadius=(pupil[2]+10 if pupil else max(12, h//8)),
            maxRadius=max(13, h//2)
        )
        if circles is not None:
            x, y, r = np.around(circles[0][0]).astype(int)
            iris = (int(x), int(y), int(r))
    except Exception:
        pass
    
    if pupil is None:
        cx, cy = w//2, h//2
        pupil = (cx, cy, max(8, h//20))
    if iris is None:
        iris = (pupil[0], pupil[1], min(h//2-1, int(pupil[2]*3)))
    
    cx = int(0.5*(pupil[0] + iris[0]))
    cy = int(0.5*(pupil[1] + iris[1]))
    rp = int(pupil[2])
    ri = int(iris[2])
    if ri <= rp:
        ri = rp + max(5, h//12)
    
    return (cx, cy, rp), (cx, cy, ri)


def normalize_iris(img_path, output_path, radial_res=64, angular_res=256, enhance=True):
    """Normalize iris image to polar coordinates with optional enhancement"""
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False, "Failed to read image"
        
        img_blur = cv2.medianBlur(img, 5)
        pupil, iris = _detect_circles(img_blur)
        
        cx, cy, rp = pupil
        _, _, ri = iris
        
        thetas = np.linspace(0, 2*math.pi, angular_res, endpoint=False)
        rs = np.linspace(rp, ri, radial_res)
        map_x = np.zeros((radial_res, angular_res), dtype=np.float32)
        map_y = np.zeros((radial_res, angular_res), dtype=np.float32)
        for j, th in enumerate(thetas):
            map_x[:, j] = cx + rs * np.cos(th)
            map_y[:, j] = cy + rs * np.sin(th)
        
        polar = cv2.remap(
            img_blur, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        if enhance:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            polar = clahe.apply(polar)
        
        stacked = np.tile(polar, (4, 1))
        stacked = np.clip(stacked, 0, 255).astype(np.uint8)
        stacked_rgb = cv2.cvtColor(stacked, cv2.COLOR_GRAY2RGB)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, stacked_rgb)
        
        return True, None
    except Exception as e:
        return False, str(e)


def process_one(args):
    """Process single image"""
    input_path, output_path, radial_res, angular_res, enhance = args
    success, error = normalize_iris(input_path, output_path, radial_res, angular_res, enhance)
    if success:
        return 0, input_path
    else:
        return -1, f"{input_path} ({error})"


def create_dataset(input_dir, output_dir, from_id, to_id, width, height, enhancement, workers, is_open_set=False):
    print(f"\nCreating dataset: {output_dir}")
    print(f"  ID range: {from_id}-{to_id-1}")
    print(f"  Size: {width}x{height}, Enhancement: {enhancement}")
    print(f"  Type: {'Open-set (enrollment/test)' if is_open_set else 'Closed-set (train/val/test)'}")
    
    if is_open_set:
        # Open-set: enrollment (7 files) + test (3 files)
        enrollment_dict, test_dict = casia_enrollment_test_split(
            input_dir,
            parse_func=parse_casia_thousand_filename,
            from_=from_id,
            to=to_id
        )
        
        train_files = list(itertools.chain.from_iterable(enrollment_dict.values()))
        val_files = []
        test_files = list(itertools.chain.from_iterable(test_dict.values()))
        
        pathlib.Path(os.path.join(output_dir, "train")).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(output_dir, "test")).mkdir(parents=True, exist_ok=True)
    else:
        # Closed-set: train (7 files) + val (2 files) + test (1 file)
        train_dict, val_dict, test_dict = casia_train_val_test_split(
            input_dir,
            parse_func=parse_casia_thousand_filename,
            from_=from_id,
            to=to_id
        )
        
        train_files = list(itertools.chain.from_iterable(train_dict.values()))
        val_files = list(itertools.chain.from_iterable(val_dict.values()))
        test_files = list(itertools.chain.from_iterable(test_dict.values()))
        
        pathlib.Path(os.path.join(output_dir, "train")).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(output_dir, "val")).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(output_dir, "test")).mkdir(parents=True, exist_ok=True)
    
    tasks = []
    for f in train_files:
        filename = os.path.basename(f)
        dest = os.path.join(output_dir, "train", filename.replace(".jpg", ".png"))
        tasks.append((f, dest, height, width, enhancement))
    
    for f in val_files:
        filename = os.path.basename(f)
        dest = os.path.join(output_dir, "val", filename.replace(".jpg", ".png"))
        tasks.append((f, dest, height, width, enhancement))
    
    for f in test_files:
        filename = os.path.basename(f)
        dest = os.path.join(output_dir, "test", filename.replace(".jpg", ".png"))
        tasks.append((f, dest, height, width, enhancement))
    
    print(f"  Total files to process: {len(tasks)}")
    
    failures = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(process_one, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            ret_code, info = fut.result()
            if ret_code != 0:
                failures.append(info)
    
    if failures:
        print(f"\n  WARNING: {len(failures)} files failed")
        for f in failures[:10]:
            print(f"    {f}")
        if len(failures) > len(tasks) * 0.5:
            return False
    
    print(f"  ✓ Processed {len(tasks) - len(failures)}/{len(tasks)} files")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="./data/CASIA-Iris-Thousand")
    ap.add_argument("--output-base", default="./data/casia-iris-preproscressed-v2")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    
    input_dir = os.path.abspath(args.input)
    output_base = os.path.abspath(args.output_base)
    
    if not os.path.isdir(input_dir):
        print(f"ERROR: Input directory not found: {input_dir}")
        return 1
    
    print("="*60)
    print("STEP 1: Normalize images using Python (HoughCircles)")
    print("="*60)
    
    success1 = create_dataset(
        input_dir,
        os.path.join(output_base, "CASIA_thousand_norm_256_64_e_nn_open_set"),
        from_id=750,
        to_id=1000,
        width=256,
        height=64,
        enhancement=True,
        workers=args.workers,
        is_open_set=True
    )
    
    success2 = create_dataset(
        input_dir,
        os.path.join(output_base, "CASIA_thousand_norm_256_64_e_nn"),
        from_id=0,
        to_id=750,
        width=256,
        height=64,
        enhancement=True,
        workers=args.workers,
        is_open_set=False
    )
    
    if success1 and success2:
        print("\n" + "="*60)
        print("✓ STEP 1 COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nNext: Run step2_organize_imagefolder.py")
        return 0
    else:
        print("\n" + "="*60)
        print("✗ STEP 1 FAILED - Too many files could not be processed")
        print("="*60)
        return 1


if __name__ == '__main__':
    exit(main())

"""
Step 1: Normalize raw CASIA-Iris-Thousand images using iris_segm_norm.exe
Creates two intermediate datasets matching the preprocessed structure:
- CASIA_thousand_norm_256_64_e_nn_open_set (ID 750-999, for open-set)
- CASIA_thousand_norm_256_64_e_nn (ID 0-749, for closed-set)
"""
import argparse
import itertools
import os
import pathlib
import subprocess
from multiprocessing import Pool
from tqdm import tqdm
import sys

sys.path.insert(0, os.path.abspath('.'))
from utils.utils import casia_train_val_test_split, parse_casia_thousand_filename


def pool_func_generic(args):
    file_path, dest_file_path, width, height, enhancement, quiet_mode = args
    exe_path = "./scripts/iris_segm_norm.exe"
    
    os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)
    
    command = [
        exe_path,
        "-i", file_path,
        "-o", dest_file_path,
        "-s", str(width), str(height),
        "-m", dest_file_path.replace('.png', '_mask.png')
    ]
    if enhancement:
        command.append("-e")
    if quiet_mode:
        command.append("-q")
    
    try:
        result = subprocess.run(command, capture_output=True, timeout=30)
        return result.returncode, file_path
    except Exception as e:
        return -1, f"{file_path} (error: {e})"


def create_dataset(input_dir, output_dir, from_id, to_id, width, height, enhancement, quiet_mode, workers):
    print(f"\nCreating dataset: {output_dir}")
    print(f"  ID range: {from_id}-{to_id-1}")
    print(f"  Size: {width}x{height}, Enhancement: {enhancement}")
    
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
        tasks.append((f, dest, width, height, enhancement, quiet_mode))
    
    for f in val_files:
        filename = os.path.basename(f)
        dest = os.path.join(output_dir, "val", filename.replace(".jpg", ".png"))
        tasks.append((f, dest, width, height, enhancement, quiet_mode))
    
    for f in test_files:
        filename = os.path.basename(f)
        dest = os.path.join(output_dir, "test", filename.replace(".jpg", ".png"))
        tasks.append((f, dest, width, height, enhancement, quiet_mode))
    
    print(f"  Total files to process: {len(tasks)}")
    
    failures = []
    with Pool(processes=workers) as pool:
        for ret_code, info in tqdm(pool.imap_unordered(pool_func_generic, tasks), total=len(tasks)):
            if ret_code != 0:
                failures.append(info)
    
    if failures:
        print(f"\n  WARNING: {len(failures)} files failed")
        for f in failures[:10]:
            print(f"    {f}")
    else:
        print(f"  ✓ All files processed successfully")
    
    return len(failures) == 0


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
    
    if not os.path.isfile("./scripts/iris_segm_norm.exe"):
        print("ERROR: iris_segm_norm.exe not found in scripts/")
        return 1
    
    print("="*60)
    print("STEP 1: Normalize images using iris_segm_norm.exe")
    print("="*60)
    
    # Dataset 1: Open-set (ID 750-999)
    success1 = create_dataset(
        input_dir,
        os.path.join(output_base, "CASIA_thousand_norm_256_64_e_nn_open_set"),
        from_id=750,
        to_id=1000,
        width=256,
        height=64,
        enhancement=True,
        quiet_mode=True,
        workers=args.workers
    )
    
    # Dataset 2: Closed-set (ID 0-749)
    success2 = create_dataset(
        input_dir,
        os.path.join(output_base, "CASIA_thousand_norm_256_64_e_nn"),
        from_id=0,
        to_id=750,
        width=256,
        height=64,
        enhancement=True,
        quiet_mode=True,
        workers=args.workers
    )
    
    if success1 and success2:
        print("\n" + "="*60)
        print("✓ STEP 1 COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nNext: Run step2_organize_imagefolder.py")
        return 0
    else:
        print("\n" + "="*60)
        print("✗ STEP 1 FAILED - Some files could not be processed")
        print("="*60)
        return 1


if __name__ == '__main__':
    exit(main())

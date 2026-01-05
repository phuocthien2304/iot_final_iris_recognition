"""
Step 2: Organize normalized images into ImageFolder structure
Converts flat train/val/test directories into class-based structure (xxx_L/xxx_R)
Also renames 'train' to 'enrollment' for open-set dataset
"""
import argparse
import os
import pathlib
import shutil
from tqdm import tqdm
import sys

sys.path.insert(0, os.path.abspath('.'))
from utils.utils import parse_casia_thousand_filename


def organize_split(split_dir, parse_func):
    """Organize files in split_dir into class subdirectories"""
    if not os.path.isdir(split_dir):
        print(f"  Skipping {split_dir} (not found)")
        return 0
    
    files = [f for f in os.listdir(split_dir) if os.path.isfile(os.path.join(split_dir, f))]
    files = [f for f in files if not f.endswith('_mask.png')]
    
    print(f"  Processing {len(files)} files...")
    
    for file in tqdm(files, desc=f"  {os.path.basename(split_dir)}"):
        if "_mask" in file:
            continue
        
        full_file_path = os.path.join(split_dir, file)
        
        try:
            identifier, side, index = parse_func(file)
            class_name = f"{identifier}_{side}"
            class_dir = os.path.join(split_dir, class_name)
            pathlib.Path(class_dir).mkdir(parents=True, exist_ok=True)
            
            dest_path = os.path.join(class_dir, file)
            shutil.move(full_file_path, dest_path)
            
            # Also move mask file if exists
            mask_file = file.replace('.png', '_mask.png')
            mask_path = os.path.join(split_dir, mask_file)
            if os.path.isfile(mask_path):
                shutil.move(mask_path, os.path.join(class_dir, mask_file))
        except Exception as e:
            print(f"    Error processing {file}: {e}")
    
    # Count classes
    classes = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
    return len(classes)


def organize_dataset(dataset_dir, rename_train_to_enrollment=False):
    """Organize all splits in a dataset"""
    print(f"\nOrganizing: {os.path.basename(dataset_dir)}")
    
    if not os.path.isdir(dataset_dir):
        print(f"  ERROR: Dataset not found: {dataset_dir}")
        return False
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        split_dir = os.path.join(dataset_dir, split)
        if os.path.isdir(split_dir):
            num_classes = organize_split(split_dir, parse_casia_thousand_filename)
            print(f"  ✓ {split}: {num_classes} classes")
    
    # Rename train -> enrollment for open-set
    if rename_train_to_enrollment:
        train_dir = os.path.join(dataset_dir, 'train')
        enrollment_dir = os.path.join(dataset_dir, 'enrollment')
        if os.path.isdir(train_dir) and not os.path.isdir(enrollment_dir):
            print(f"  Renaming train -> enrollment")
            shutil.move(train_dir, enrollment_dir)
        
        # Remove val split for open-set (only keep enrollment + test)
        val_dir = os.path.join(dataset_dir, 'val')
        if os.path.isdir(val_dir):
            print(f"  Removing val split (open-set only needs enrollment + test)")
            shutil.rmtree(val_dir)
    
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="./data/casia-iris-preproscressed-v2")
    args = ap.parse_args()
    
    base_dir = os.path.abspath(args.base)
    
    print("="*60)
    print("STEP 2: Organize into ImageFolder structure")
    print("="*60)
    
    # Organize open-set dataset (rename train -> enrollment)
    success1 = organize_dataset(
        os.path.join(base_dir, "CASIA_thousand_norm_256_64_e_nn_open_set"),
        rename_train_to_enrollment=True
    )
    
    # Organize closed-set dataset (keep train/val/test)
    success2 = organize_dataset(
        os.path.join(base_dir, "CASIA_thousand_norm_256_64_e_nn"),
        rename_train_to_enrollment=False
    )
    
    if success1 and success2:
        print("\n" + "="*60)
        print("✓ STEP 2 COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nNext: Run step3_stack_images.py")
        return 0
    else:
        print("\n" + "="*60)
        print("✗ STEP 2 FAILED")
        print("="*60)
        return 1


if __name__ == '__main__':
    exit(main())

"""
Step 3: Stack 64x256 normalized images into 256x256 RGB images
Creates final _stacked versions of both datasets
"""
import argparse
import os
import shutil
from PIL import Image
import numpy as np
from tqdm import tqdm


def get_all_image_files(directory):
    """Recursively get all image files"""
    image_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.png') and '_mask' not in file:
                image_files.append(os.path.join(root, file))
    return image_files


def stack_image(input_path, output_path):
    """Stack 64x256 image 4 times vertically to create 256x256 RGB"""
    try:
        img = np.array(Image.open(input_path))
        
        # Handle different input formats
        if len(img.shape) == 3:
            img = img[:, :, 0]  # Take first channel if RGB
        
        h, w = img.shape
        
        # Create 256x256 by stacking 4 times
        new_img = np.zeros((256, 256), dtype=img.dtype)
        new_img[0:64, :] = img
        new_img[64:128, :] = img
        new_img[128:192, :] = img
        new_img[192:256, :] = img
        
        # Convert to RGB
        rgb_img = Image.fromarray(new_img).convert("RGB")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        rgb_img.save(output_path)
        return True
    except Exception as e:
        print(f"  Error stacking {input_path}: {e}")
        return False


def create_stacked_dataset(src_dir, dst_dir):
    """Create stacked version of a dataset"""
    print(f"\nCreating stacked dataset:")
    print(f"  Source: {os.path.basename(src_dir)}")
    print(f"  Destination: {os.path.basename(dst_dir)}")
    
    if not os.path.isdir(src_dir):
        print(f"  ERROR: Source not found: {src_dir}")
        return False
    
    # Get all image files
    image_files = get_all_image_files(src_dir)
    print(f"  Found {len(image_files)} images to stack")
    
    if len(image_files) == 0:
        print(f"  WARNING: No images found in {src_dir}")
        return False
    
    # Process each image
    failures = 0
    for img_path in tqdm(image_files, desc="  Stacking"):
        # Compute relative path and output path
        rel_path = os.path.relpath(img_path, src_dir)
        out_path = os.path.join(dst_dir, rel_path)
        
        if not stack_image(img_path, out_path):
            failures += 1
        
        # Also copy mask if exists
        mask_path = img_path.replace('.png', '_mask.png')
        if os.path.isfile(mask_path):
            out_mask_path = out_path.replace('.png', '_mask.png')
            os.makedirs(os.path.dirname(out_mask_path), exist_ok=True)
            shutil.copy2(mask_path, out_mask_path)
    
    if failures > 0:
        print(f"  WARNING: {failures} images failed to stack")
        return False
    else:
        print(f"  ✓ All images stacked successfully")
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="./data/casia-iris-preproscressed-v2")
    args = ap.parse_args()
    
    base_dir = os.path.abspath(args.base)
    
    print("="*60)
    print("STEP 3: Stack images (64x256 -> 256x256 RGB)")
    print("="*60)
    
    # Create stacked version of open-set
    success1 = create_stacked_dataset(
        os.path.join(base_dir, "CASIA_thousand_norm_256_64_e_nn_open_set"),
        os.path.join(base_dir, "CASIA_thousand_norm_256_64_e_nn_open_set_stacked")
    )
    
    # Create stacked version of closed-set
    success2 = create_stacked_dataset(
        os.path.join(base_dir, "CASIA_thousand_norm_256_64_e_nn"),
        os.path.join(base_dir, "CASIA_thousand_norm_256_64_e_nn_stacked")
    )
    
    if success1 and success2:
        print("\n" + "="*60)
        print("✓ STEP 3 COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nAll steps completed! Verifying final structure...")
        
        # Quick verification
        print("\nFinal structure:")
        for ds in ["CASIA_thousand_norm_256_64_e_nn_open_set_stacked", 
                   "CASIA_thousand_norm_256_64_e_nn_stacked"]:
            ds_path = os.path.join(base_dir, ds)
            if os.path.isdir(ds_path):
                splits = [d for d in os.listdir(ds_path) if os.path.isdir(os.path.join(ds_path, d))]
                print(f"  {ds}:")
                for split in splits:
                    split_path = os.path.join(ds_path, split)
                    classes = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
                    print(f"    {split}: {len(classes)} classes")
        
        return 0
    else:
        print("\n" + "="*60)
        print("✗ STEP 3 FAILED")
        print("="*60)
        return 1


if __name__ == '__main__':
    exit(main())

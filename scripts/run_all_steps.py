"""
Master script to run all 3 steps in sequence
Creates casia-iris-preproscressed-v2 from CASIA-Iris-Thousand matching preprocessed exactly
"""
import subprocess
import sys
import os

def run_step(script_name, step_num, total_steps):
    print("\n" + "="*70)
    print(f"RUNNING STEP {step_num}/{total_steps}: {script_name}")
    print("="*70 + "\n")
    
    result = subprocess.run([sys.executable, f"scripts/{script_name}"], cwd=".")
    
    if result.returncode != 0:
        print(f"\n✗ Step {step_num} failed with exit code {result.returncode}")
        return False
    
    return True

def main():
    print("="*70)
    print("CREATING casia-iris-preproscressed-v2 FROM CASIA-Iris-Thousand")
    print("="*70)
    print("\nThis will run 3 steps:")
    print("  1. Normalize images using Python (HoughCircles + CLAHE)")
    print("  2. Organize into ImageFolder structure")
    print("  3. Stack images (64x256 -> 256x256 RGB)")
    print("\nEstimated time: 10-30 minutes depending on your CPU")
    print()
    
    response = input("Continue? (yes/no): ").strip().lower()
    if response != "yes":
        print("Aborted.")
        return 1
    
    steps = [
        "step1_normalize_python.py",
        "step2_organize_imagefolder.py",
        "step3_stack_images.py"
    ]
    
    for i, step in enumerate(steps, 1):
        if not run_step(step, i, len(steps)):
            print("\n" + "="*70)
            print("✗ PIPELINE FAILED")
            print("="*70)
            return 1
    
    print("\n" + "="*70)
    print("✓ ALL STEPS COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nYour preprocessed dataset is ready at:")
    print("  data/casia-iris-preproscressed-v2/")
    print("\nYou can now use it for training/evaluation just like the original.")
    
    return 0

if __name__ == '__main__':
    exit(main())

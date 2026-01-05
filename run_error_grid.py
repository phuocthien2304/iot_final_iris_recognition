"""
Script to run error grid evaluation with the new Original dataset
"""
import subprocess
import sys

def main():
    checkpoint = "./models/best_iris_cnn_improved.pth"
    arch = "resnet101"
    enroll = "./data/casia-iris-preprocessed/CASIA_thousand_norm_256_64_e_nn_open_set_stacked/enrollment"
    test = "./data/casia-iris-preprocessed/CASIA_thousand_norm_256_64_e_nn_open_set_stacked/test"
    
    cmd = [
        sys.executable,
        "scripts/error_grid_eval.py",
        "--checkpoint", checkpoint,
        "--arch", arch,
        "--enroll", enroll,
        "--test", test,
        "--batch-size", "196",
        "--num-workers", "4",
        "--thr", "0.90",
        "--splits", "4",
        "--grid-cols", "0",
        "--save-prefix", "results/error_grids"
    ]
    
    print("Running error grid evaluation...")
    print(" ".join(cmd))
    print()
    
    result = subprocess.run(cmd)
    return result.returncode

if __name__ == '__main__':
    exit(main())

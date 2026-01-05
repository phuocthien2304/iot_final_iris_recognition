"""
Evaluate open-set FAR (False Accept Rate) and FRR (False Reject Rate) with best_iris_cnn_improved.pth
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.abspath('.'))

from models import ResNet101Iris

def get_model(model_name, checkpoint_path, num_classes=1500):
    """Load model and checkpoint"""
    model = None
    input_size = 224
    
    if model_name == "resnet101":
        model = ResNet101Iris(num_classes=num_classes)
        input_size = 224
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    state = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(state)
    return model, input_size

def get_dataloader(data_path, input_size, batch_size=196, shuffle=False, num_workers=4):
    """Create dataloader"""
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(data_path, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def enroll_identities(model, dataloader, device):
    """Enroll identities by computing mean feature vectors"""
    model.eval()
    class_features = {}
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            features = model.feature_extract_avg_pool(inputs)
            features = features.cpu().numpy()
            
            for feat, lbl in zip(features, labels.numpy()):
                lbl = int(lbl)
                if lbl not in class_features:
                    class_features[lbl] = []
                class_features[lbl].append(feat)
    
    enrolled = {}
    for lbl, feats in class_features.items():
        enrolled[lbl] = np.mean(feats, axis=0)
    
    return enrolled

def evaluate_open_set(enrolled, model, test_loader, device, enroll_classes, test_classes):
    """Evaluate open-set with FAR/FRR calculation"""
    model.eval()
    
    all_similarities = []
    all_predictions = []
    all_labels = []
    all_is_genuine = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            features = model.feature_extract_avg_pool(inputs)
            features = features.cpu().numpy()
            
            for feat, lbl in zip(features, labels.numpy()):
                lbl = int(lbl)
                true_class_name = test_classes[lbl]
                is_genuine = true_class_name in enroll_classes
                
                # Compute similarity with all enrolled identities
                sims = []
                for enrolled_lbl, enrolled_feat in enrolled.items():
                    sim = np.dot(feat, enrolled_feat) / (np.linalg.norm(feat) * np.linalg.norm(enrolled_feat))
                    sims.append((enrolled_lbl, sim))
                
                # Get best match
                best_lbl, best_sim = max(sims, key=lambda x: x[1])
                
                all_similarities.append(best_sim)
                all_predictions.append(best_lbl)
                all_labels.append(lbl)
                all_is_genuine.append(is_genuine)
    
    return np.array(all_similarities), np.array(all_predictions), np.array(all_labels), np.array(all_is_genuine)

def compute_far_frr(similarities, predictions, labels, is_genuine, enroll_classes, test_classes, thresholds):
    """Compute FAR and FRR for different thresholds"""
    far_list = []
    frr_list = []
    
    for thr in thresholds:
        false_accepts = 0
        false_rejects = 0
        total_impostors = 0
        total_genuines = 0
        
        for i in range(len(similarities)):
            sim = similarities[i]
            pred_lbl = predictions[i]
            true_lbl = labels[i]
            genuine = is_genuine[i]
            
            if genuine:
                total_genuines += 1
                true_class_name = test_classes[true_lbl]
                pred_class_name = enroll_classes[pred_lbl]
                
                # False reject: genuine but rejected (sim < thr) or wrong identity
                if sim < thr or pred_class_name != true_class_name:
                    false_rejects += 1
            else:
                total_impostors += 1
                # False accept: impostor but accepted (sim >= thr)
                if sim >= thr:
                    false_accepts += 1
        
        far = (false_accepts / total_impostors * 100) if total_impostors > 0 else 0.0
        frr = (false_rejects / total_genuines * 100) if total_genuines > 0 else 0.0
        
        far_list.append(far)
        frr_list.append(frr)
    
    return np.array(far_list), np.array(frr_list)

def find_eer(far_list, frr_list, thresholds):
    """Find Equal Error Rate (EER) where FAR = FRR"""
    diff = np.abs(far_list - frr_list)
    eer_idx = np.argmin(diff)
    eer = (far_list[eer_idx] + frr_list[eer_idx]) / 2.0
    eer_threshold = thresholds[eer_idx]
    return eer, eer_threshold, eer_idx

def plot_far_frr(thresholds, far_list, frr_list, eer, eer_threshold, save_path):
    """Plot FAR/FRR curves"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(thresholds, far_list, 'b-', label='FAR (False Accept Rate)', linewidth=2)
    ax.plot(thresholds, frr_list, 'r-', label='FRR (False Reject Rate)', linewidth=2)
    ax.axvline(eer_threshold, color='g', linestyle='--', label=f'EER = {eer:.2f}% @ thr={eer_threshold:.3f}', linewidth=1.5)
    
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Error Rate (%)', fontsize=12)
    ax.set_title('FAR/FRR vs Threshold', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

def main():
    print("="*80)
    print("OPEN-SET FAR/FRR EVALUATION")
    print("="*80)
    
    # Configuration
    checkpoint_path = "./models/best_iris_cnn_improved.pth"
    model_name = "best_iris_cnn_improved"
    arch_name = "resnet101"
    enrollment_path = "./data/casia-iris-preprocessed/CASIA_thousand_norm_256_64_e_nn_open_set_stacked/enrollment"
    test_path = "./data/casia-iris-preprocessed/CASIA_thousand_norm_256_64_e_nn_open_set_stacked/test"
    batch_size = 196
    num_workers = 4
    
    print(f"\nModel: {model_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Enrollment data: {enrollment_path}")
    print(f"Test data: {test_path}")
    
    # Load model
    print("\nLoading model...")
    model, input_size = get_model(arch_name, checkpoint_path, num_classes=1500)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model.to(device)
    model.eval()
    
    # Load data
    print("\nLoading enrollment data...")
    enroll_loader = get_dataloader(enrollment_path, input_size, batch_size=batch_size, num_workers=num_workers)
    print(f"Enrollment samples: {len(enroll_loader.dataset)}")
    print(f"Enrollment classes: {len(enroll_loader.dataset.classes)}")
    
    print("\nLoading test data...")
    test_loader = get_dataloader(test_path, input_size, batch_size=batch_size, num_workers=num_workers)
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Test classes: {len(test_loader.dataset.classes)}")
    
    enroll_classes = enroll_loader.dataset.classes
    test_classes = test_loader.dataset.classes
    
    # Enroll identities
    print("\nEnrolling identities...")
    enrolled = enroll_identities(model, enroll_loader, device)
    print(f"Enrolled {len(enrolled)} identities")
    
    # Evaluate
    print("\nEvaluating open-set...")
    similarities, predictions, labels, is_genuine = evaluate_open_set(
        enrolled, model, test_loader, device, enroll_classes, test_classes
    )
    
    print(f"Total test samples: {len(similarities)}")
    print(f"Genuine samples: {np.sum(is_genuine)}")
    print(f"Impostor samples: {np.sum(~is_genuine)}")
    
    # Compute FAR/FRR for different thresholds
    print("\nComputing FAR/FRR curves...")
    thresholds = np.linspace(0.0, 1.0, 101)
    far_list, frr_list = compute_far_frr(similarities, predictions, labels, is_genuine, enroll_classes, test_classes, thresholds)
    
    # Find EER
    eer, eer_threshold, eer_idx = find_eer(far_list, frr_list, thresholds)
    
    # Results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\nEqual Error Rate (EER): {eer:.2f}%")
    print(f"EER Threshold: {eer_threshold:.3f}")
    print(f"\nAt threshold = 0.90:")
    idx_90 = np.argmin(np.abs(thresholds - 0.90))
    print(f"  FAR: {far_list[idx_90]:.2f}%")
    print(f"  FRR: {frr_list[idx_90]:.2f}%")
    print("\n" + "="*80)
    
    # Save results
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save plot
    plot_path = f"{results_dir}/far_frr_curve.png"
    plot_far_frr(thresholds, far_list, frr_list, eer, eer_threshold, plot_path)
    print(f"\nFAR/FRR curve saved to {plot_path}")
    
    # Save text results
    with open(f"{results_dir}/open_set_far_frr.txt", "w") as f:
        f.write("OPEN-SET FAR/FRR EVALUATION\n")
        f.write("="*80 + "\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Enrollment data: {enrollment_path}\n")
        f.write(f"Test data: {test_path}\n")
        f.write(f"Device: {device}\n\n")
        f.write(f"Total test samples: {len(similarities)}\n")
        f.write(f"Genuine samples: {np.sum(is_genuine)}\n")
        f.write(f"Impostor samples: {np.sum(~is_genuine)}\n\n")
        f.write(f"Equal Error Rate (EER): {eer:.2f}%\n")
        f.write(f"EER Threshold: {eer_threshold:.3f}\n\n")
        f.write(f"At threshold = 0.90:\n")
        f.write(f"  FAR: {far_list[idx_90]:.2f}%\n")
        f.write(f"  FRR: {frr_list[idx_90]:.2f}%\n\n")
        f.write("Threshold\tFAR(%)\tFRR(%)\n")
        f.write("-"*40 + "\n")
        for i in range(0, len(thresholds), 5):
            f.write(f"{thresholds[i]:.2f}\t\t{far_list[i]:.2f}\t{frr_list[i]:.2f}\n")
    
    print(f"Results saved to {results_dir}/open_set_far_frr.txt")

if __name__ == '__main__':
    main()

"""
Evaluate closed-set accuracy with the best_iris_cnn_improved.pth model
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

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

def get_dataloader(data_path, input_size, batch_size=128, shuffle=False, num_workers=4):
    """Create dataloader for evaluation"""
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(data_path, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def evaluate_accuracy(model, dataloader, device):
    """Evaluate top-1 and top-5 accuracy and collect predictions"""
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            
            # Top-1 accuracy
            _, pred_top1 = outputs.max(1)
            correct_top1 += pred_top1.eq(labels).sum().item()
            
            # Top-5 accuracy
            _, pred_top5 = outputs.topk(5, 1, True, True)
            pred_top5 = pred_top5.t()
            correct_top5 += pred_top5.eq(labels.view(1, -1).expand_as(pred_top5)).sum().item()
            
            total += labels.size(0)
            
            # Collect for confusion matrix
            all_preds.extend(pred_top1.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    top1_acc = 100.0 * correct_top1 / total
    top5_acc = 100.0 * correct_top5 / total
    
    return top1_acc, top5_acc, total, np.array(all_preds), np.array(all_labels)

def main():
    print("="*80)
    print("CLOSED-SET ACCURACY EVALUATION")
    print("="*80)
    
    # Configuration
    checkpoint_path = "./models/best_iris_cnn_improved.pth"
    model_name = "best_iris_cnn_improved"
    arch_name = "resnet101"
    test_data_path = "./data/casia-iris-preprocessed/CASIA_thousand_norm_256_64_e_nn_stacked/test"
    batch_size = 128
    num_workers = 4
    
    print(f"\nModel: {model_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test data: {test_data_path}")
    print(f"Batch size: {batch_size}")
    
    # Load model
    print("\nLoading model...")
    model, input_size = get_model(arch_name, checkpoint_path, num_classes=1500)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model.to(device)
    model.eval()
    
    # Load test data
    print("\nLoading test data...")
    test_loader = get_dataloader(test_data_path, input_size, batch_size=batch_size, num_workers=num_workers)
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Test classes: {len(test_loader.dataset.classes)}")
    
    # Evaluate
    print("\nEvaluating...")
    top1_acc, top5_acc, total, all_preds, all_labels = evaluate_accuracy(model, test_loader, device)
    
    # Results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\nTotal test samples: {total}")
    print(f"Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc:.2f}%")
    print("\n" + "="*80)
    
    # Save results
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    with open(f"{results_dir}/closed_set_accuracy.txt", "w") as f:
        f.write("CLOSED-SET ACCURACY EVALUATION\n")
        f.write("="*80 + "\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Test data: {test_data_path}\n")
        f.write(f"Device: {device}\n\n")
        f.write(f"Total test samples: {total}\n")
        f.write(f"Top-1 Accuracy: {top1_acc:.2f}%\n")
        f.write(f"Top-5 Accuracy: {top5_acc:.2f}%\n")
    
    print(f"\nResults saved to {results_dir}/closed_set_accuracy.txt")
    
    # Generate confusion matrix (sample only first 100 classes for visualization)
    print("\nGenerating confusion matrix visualization...")
    num_classes_to_show = min(100, len(test_loader.dataset.classes))
    
    # Filter predictions and labels for first N classes
    mask = all_labels < num_classes_to_show
    filtered_preds = all_preds[mask]
    filtered_labels = all_labels[mask]
    
    if len(filtered_labels) > 0:
        cm = confusion_matrix(filtered_labels, filtered_preds, labels=range(num_classes_to_show))
        
        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', ax=ax, cbar_kws={'label': 'Count'})
        ax.set_xlabel('Predicted Class', fontsize=12)
        ax.set_ylabel('True Class', fontsize=12)
        ax.set_title(f'Confusion Matrix (First {num_classes_to_show} Classes)\nTop-1 Accuracy: {top1_acc:.2f}%', fontsize=14, fontweight='bold')
        
        # Add accuracy text
        textstr = f'Total Samples: {len(filtered_labels)}\nCorrect: {np.sum(filtered_preds == filtered_labels)}\nIncorrect: {np.sum(filtered_preds != filtered_labels)}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        fig.tight_layout()
        cm_path = f"{results_dir}/confusion_matrix.png"
        fig.savefig(cm_path, dpi=150)
        plt.close(fig)
        
        print(f"Confusion matrix saved to {cm_path}")
    
    # Generate per-class accuracy bar chart
    print("\nGenerating per-class accuracy chart...")
    class_correct = np.zeros(len(test_loader.dataset.classes))
    class_total = np.zeros(len(test_loader.dataset.classes))
    
    for pred, label in zip(all_preds, all_labels):
        class_total[label] += 1
        if pred == label:
            class_correct[label] += 1
    
    class_accuracy = np.divide(class_correct, class_total, out=np.zeros_like(class_correct), where=class_total!=0) * 100
    
    # Plot top 50 and bottom 50 classes
    sorted_indices = np.argsort(class_accuracy)
    worst_50 = sorted_indices[:50]
    best_50 = sorted_indices[-50:]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Bottom 50
    ax1.bar(range(50), class_accuracy[worst_50], color='red', alpha=0.7)
    ax1.set_xlabel('Class Index (Worst 50)', fontsize=10)
    ax1.set_ylabel('Accuracy (%)', fontsize=10)
    ax1.set_title('Worst 50 Classes by Accuracy', fontsize=12, fontweight='bold')
    ax1.set_ylim([0, 100])
    ax1.grid(axis='y', alpha=0.3)
    
    # Top 50
    ax2.bar(range(50), class_accuracy[best_50], color='green', alpha=0.7)
    ax2.set_xlabel('Class Index (Best 50)', fontsize=10)
    ax2.set_ylabel('Accuracy (%)', fontsize=10)
    ax2.set_title('Best 50 Classes by Accuracy', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 100])
    ax2.grid(axis='y', alpha=0.3)
    
    fig.tight_layout()
    acc_chart_path = f"{results_dir}/per_class_accuracy.png"
    fig.savefig(acc_chart_path, dpi=150)
    plt.close(fig)
    
    print(f"Per-class accuracy chart saved to {acc_chart_path}")
    print(f"\nAll results saved to {results_dir}/")

if __name__ == '__main__':
    main()

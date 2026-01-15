import torch
from dataloader import get_dataloaders
from model import TemporalJerseyNet
from train import train
from validate import validate
from inference import predict
import matplotlib.pyplot as plt
import os

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load data
    train_loader, val_loader, test_loader = get_dataloaders(
        root_dir="temporal_jersey_nr_recognition_dataset_subset",
        seq_len=8,
        batch_size=4
    )

    # Initialize model
    model = TemporalJerseyNet().to(device)
    print("[INFO] Model initialized")

    # Train model and get history
    history = train(model, train_loader, val_loader, device, epochs=10)

    # Create directory to save curves
    curves_dir = "training_curves"
    os.makedirs(curves_dir, exist_ok=True)

    # Plot and save loss curves
    plt.figure(figsize=(10, 4))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    loss_curve_path = os.path.join(curves_dir, "loss_curve.png")
    plt.savefig(loss_curve_path)
    print(f"[INFO] Loss curve saved to {loss_curve_path}")
    plt.close()

    # Plot and save accuracy curves
    plt.figure(figsize=(10, 4))
    plt.plot(history["val_acc1"], label="Digit 1 Accuracy")
    plt.plot(history["val_acc2"], label="Digit 2 Accuracy")
    plt.plot(history["val_jersey_acc"], label="Jersey Number Accuracy")
    plt.title("Validation Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    acc_curve_path = os.path.join(curves_dir, "accuracy_curve.png")
    plt.savefig(acc_curve_path)
    print(f"[INFO] Accuracy curve saved to {acc_curve_path}")
    plt.close()

    # Evaluate metrics on the test set
    print("\n[TEST SET METRICS]")
    test_results = validate(model, test_loader, device)
    
    print("Digit 1 Metrics:")
    print(f"  Accuracy:  {test_results['acc1']:.4f}")
    print(f"  Precision: {test_results['prec1']:.4f}")
    print(f"  Recall:    {test_results['rec1']:.4f}")
    print(f"  F1-Score:  {test_results['f1_1']:.4f}")
    print(f"  ROC-AUC:   {test_results['roc1'] if test_results['roc1'] is not None else 'N/A'}\n")

    print("Digit 2 Metrics:")
    print(f"  Accuracy:  {test_results['acc2']:.4f}")
    print(f"  Precision: {test_results['prec2']:.4f}")
    print(f"  Recall:    {test_results['rec2']:.4f}")
    print(f"  F1-Score:  {test_results['f1_2']:.4f}")
    print(f"  ROC-AUC:   {test_results['roc2'] if test_results['roc2'] is not None else 'N/A'}\n")

    print(f"Jersey Number Accuracy: {test_results['jersey_acc']:.4f}")

    # Sample predictions
    preds = predict(model, test_loader, device)
    print("\n[SAMPLE TEST PREDICTIONS]")
    for gt, pr in preds:
        print(f"GT: {gt} → Pred: {pr}")

    print("\n[INFO] Test set evaluation completed")

if __name__ == "__main__":
    main()
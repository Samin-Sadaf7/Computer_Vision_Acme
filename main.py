import torch
from dataloader import get_dataloaders
from model import TemporalJerseyNet
from train import train
from validate import validate
from inference import predict

def main():
    # Device setup
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

    # Train the model and plot curves
    history = train(
        model,
        train_loader,
        val_loader,
        device,
        epochs=10,
        plot_curves=True
    )

    # Evaluate on validation set
    results = validate(model, val_loader, device)
    print(f"Digit1 Acc: {results['acc1']:.3f}, Prec: {results['prec1']:.3f}, Rec: {results['rec1']:.3f}, F1: {results['f1_1']:.3f}, ROC-AUC: {results['roc1']:.3f}")
    print(f"Digit2 Acc: {results['acc2']:.3f}, Prec: {results['prec2']:.3f}, Rec: {results['rec2']:.3f}, F1: {results['f1_2']:.3f}, ROC-AUC: {results['roc2']:.3f}")
    print(f"Jersey Acc: {results['jersey_acc']:.3f}")

    # Sample predictions
    preds = predict(model, test_loader, device)
    print("[SAMPLE PREDICTIONS]")
    for gt, pr in preds:
        print(f"GT: {gt} → Pred: {pr}")

    # Evaluate on test set
    test_results = validate(model, test_loader, device)
    print(f"\n[Test Set] Digit1 Acc: {test_results['acc1']:.3f}, Prec: {test_results['prec1']:.3f}, Rec: {test_results['rec1']:.3f}, F1: {test_results['f1_1']:.3f}, ROC-AUC: {test_results['roc1']:.3f}")
    print(f"[Test Set] Digit2 Acc: {test_results['acc2']:.3f}, Prec: {test_results['prec2']:.3f}, Rec: {test_results['rec2']:.3f}, F1: {test_results['f1_2']:.3f}, ROC-AUC: {test_results['roc2']:.3f}")    
    print(f"[Test Set] Jersey Acc: {test_results['jersey_acc']:.3f}")
    print("\n[INFO] Inference completed")

if __name__ == "__main__":
    main()
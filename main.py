import torch
from dataloader import get_dataloaders
from model import TemporalJerseyModel
from train import train
from validate import validate
from inference import predict

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(
        root_dir="temporal_jersey_nr_recognition_dataset_subset",
        seq_len=8,
        batch_size=4
    )

    model = TemporalJerseyModel().to(device)
    print("[INFO] Model initialized")

    train(
        model,
        train_loader,
        val_loader,
        device,
        epochs=10
    )

    acc1, acc2, jersey_acc = validate(model, test_loader, device)
    print(f"[TEST] D1 Acc: {acc1:.3f}, D2 Acc: {acc2:.3f}, Jersey Acc: {jersey_acc:.3f}")

    preds = predict(model, test_loader, device)
    print("[SAMPLE PREDICTIONS]")
    for gt, pr in preds:
        print(f"GT: {gt} → Pred: {pr}")

if __name__ == "__main__":
    main()
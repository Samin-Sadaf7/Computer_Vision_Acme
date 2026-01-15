import torch
import torch.nn as nn
from validate import validate
import matplotlib.pyplot as plt
import os

def train(model, train_loader, val_loader, device, epochs=10, save_curves=True, curves_path="training_curves.png"):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    history = {
        "loss": [],
        "val_d1_acc": [],
        "val_d2_acc": [],
        "val_jersey_acc": []
    }

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for imgs, d1, d2 in train_loader:
            imgs = imgs.to(device)
            d1 = d1.to(device)
            d2 = d2.to(device)

            optimizer.zero_grad()
            o1, o2 = model(imgs)

            loss = criterion(o1, d1) + criterion(o2, d2)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        history["loss"].append(avg_loss)

        acc1, acc2, jersey_acc = validate(model, val_loader, device)
        history["val_d1_acc"].append(acc1)
        history["val_d2_acc"].append(acc2)
        history["val_jersey_acc"].append(jersey_acc)

        print(
            f"[Epoch {epoch+1}] "
            f"Loss: {avg_loss:.4f} | "
            f"D1 Acc: {acc1:.3f} | "
            f"D2 Acc: {acc2:.3f} | "
            f"Jersey Acc: {jersey_acc:.3f}"
        )

    # Plot and save curves
    if save_curves:
        epochs_range = range(1, epochs + 1)
        plt.figure(figsize=(12, 5))

        # Training Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, history["loss"], label="Training Loss", marker='o', color="red")
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()

        # Validation Accuracies
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, history["val_d1_acc"], label="D1 Accuracy", marker='o')
        plt.plot(epochs_range, history["val_d2_acc"], label="D2 Accuracy", marker='o')
        plt.plot(epochs_range, history["val_jersey_acc"], label="Jersey Accuracy", marker='o', linewidth=2)
        plt.title("Validation Accuracies")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig(curves_path)
        print(f"[INFO] Training curves saved to {curves_path}")
        plt.close()

    return history
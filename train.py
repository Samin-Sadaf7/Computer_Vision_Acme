import torch
import torch.nn as nn
from validate import validate
import matplotlib.pyplot as plt

def train(model, train_loader, val_loader, device, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_jersey_acc": [],
        "val_acc1": [],
        "val_acc2": []
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

        avg_train_loss = total_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)

        # Validation
        model.eval()
        val_loss_total = 0
        with torch.no_grad():
            for imgs, d1, d2 in val_loader:
                imgs = imgs.to(device)
                d1 = d1.to(device)
                d2 = d2.to(device)
                o1, o2 = model(imgs)
                val_loss_total += (criterion(o1, d1) + criterion(o2, d2)).item()

        avg_val_loss = val_loss_total / len(val_loader)
        history["val_loss"].append(avg_val_loss)

        # Get validation metrics (accuracy)
        acc1, acc2, jersey_acc = validate(model, val_loader, device)
        history["val_acc1"].append(acc1)
        history["val_acc2"].append(acc2)
        history["val_jersey_acc"].append(jersey_acc)

        print(
            f"[Epoch {epoch+1}/{epochs}] "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"D1 Acc: {acc1:.3f} | D2 Acc: {acc2:.3f} | Jersey Acc: {jersey_acc:.3f}"
        )

    # Plot training and validation curves
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history["val_acc1"], label="Digit 1 Acc")
    plt.plot(history["val_acc2"], label="Digit 2 Acc")
    plt.plot(history["val_jersey_acc"], label="Jersey Acc")
    plt.title("Validation Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return history
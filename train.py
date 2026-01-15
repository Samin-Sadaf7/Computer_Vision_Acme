import torch
import torch.nn as nn
from validate import validate

def train(model, train_loader, val_loader, device, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    history = {
        "loss": [],
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
        history["val_jersey_acc"].append(jersey_acc)

        print(
            f"[Epoch {epoch+1}] "
            f"Loss: {avg_loss:.4f} | "
            f"D1 Acc: {acc1:.3f} | "
            f"D2 Acc: {acc2:.3f} | "
            f"Jersey Acc: {jersey_acc:.3f}"
        )

    return history
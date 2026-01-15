import torch.optim as optim
import torch.nn.functional as F

def train(model, dataloader, epochs=20):
    device = next(model.parameters()).device
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for imgs, d1, d2 in dataloader:
            imgs = imgs.to(device)
            d1 = d1.to(device)
            d2 = d2.to(device)

            p1, p2 = model(imgs)
            loss = F.cross_entropy(p1, d1) + F.cross_entropy(p2, d2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss/len(dataloader):.4f}")
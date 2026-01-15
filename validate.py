import torch
from sklearn.metrics import accuracy_score

def validate(model, dataloader, device):
    model.eval()

    d1_true, d1_pred = [], []
    d2_true, d2_pred = [], []

    with torch.no_grad():
        for imgs, d1, d2 in dataloader:
            imgs = imgs.to(device)
            d1 = d1.to(device)
            d2 = d2.to(device)

            o1, o2 = model(imgs)

            p1 = torch.argmax(o1, dim=1)
            p2 = torch.argmax(o2, dim=1)

            d1_true.extend(d1.cpu().numpy())
            d1_pred.extend(p1.cpu().numpy())
            d2_true.extend(d2.cpu().numpy())
            d2_pred.extend(p2.cpu().numpy())

    acc1 = accuracy_score(d1_true, d1_pred)
    acc2 = accuracy_score(d2_true, d2_pred)

    jersey_acc = sum(
        (a == b and c == d)
        for a, b, c, d in zip(d1_true, d1_pred, d2_true, d2_pred)
    ) / len(d1_true)

    return acc1, acc2, jersey_acc
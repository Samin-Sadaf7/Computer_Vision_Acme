import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def validate(model, dataloader, device):
    model.eval()

    y1_true, y1_pred = [], []
    y2_true, y2_pred = [], []

    with torch.no_grad():
        for imgs, d1, d2 in dataloader:
            imgs = imgs.to(device)
            d1 = d1.to(device)
            d2 = d2.to(device)

            o1, o2 = model(imgs)

            p1 = torch.argmax(o1, dim=1)
            p2 = torch.argmax(o2, dim=1)

            y1_true.extend(d1.cpu().numpy())
            y1_pred.extend(p1.cpu().numpy())

            y2_true.extend(d2.cpu().numpy())
            y2_pred.extend(p2.cpu().numpy())

    acc1 = accuracy_score(y1_true, y1_pred)
    acc2 = accuracy_score(y2_true, y2_pred)

    f1_1 = precision_recall_fscore_support(
        y1_true, y1_pred, average="macro"
    )[2]
    f1_2 = precision_recall_fscore_support(
        y2_true, y2_pred, average="macro"
    )[2]

    jersey_acc = sum(
        (a == b and c == d)
        for a, b, c, d in zip(y1_true, y1_pred, y2_true, y2_pred)
    ) / len(y1_true)

    return {
        "Digit1 Acc": acc1,
        "Digit2 Acc": acc2,
        "Digit1 F1": f1_1,
        "Digit2 F1": f1_2,
        "Jersey Acc": jersey_acc
    }

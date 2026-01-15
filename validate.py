import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import numpy as np

def validate(model, dataloader, device):
    model.eval()

    d1_true, d1_pred, d1_probs = [], [], []
    d2_true, d2_pred, d2_probs = [], [], []

    with torch.no_grad():
        for imgs, d1, d2 in dataloader:
            imgs = imgs.to(device)
            d1 = d1.to(device)
            d2 = d2.to(device)

            o1, o2 = model(imgs)

            probs1 = torch.softmax(o1, dim=1)
            probs2 = torch.softmax(o2, dim=1)

            p1 = torch.argmax(probs1, dim=1)
            p2 = torch.argmax(probs2, dim=1)

            d1_true.extend(d1.cpu().numpy())
            d1_pred.extend(p1.cpu().numpy())
            d1_probs.extend(probs1.cpu().numpy())

            d2_true.extend(d2.cpu().numpy())
            d2_pred.extend(p2.cpu().numpy())
            d2_probs.extend(probs2.cpu().numpy())

    # Convert to numpy arrays
    d1_true = np.array(d1_true)
    d1_pred = np.array(d1_pred)
    d1_probs = np.array(d1_probs)

    d2_true = np.array(d2_true)
    d2_pred = np.array(d2_pred)
    d2_probs = np.array(d2_probs)

    # Accuracy
    acc1 = accuracy_score(d1_true, d1_pred)
    acc2 = accuracy_score(d2_true, d2_pred)

    # Precision, Recall, F1 (macro averaged)
    prec1 = precision_score(d1_true, d1_pred, average='macro', zero_division=0)
    prec2 = precision_score(d2_true, d2_pred, average='macro', zero_division=0)

    rec1 = recall_score(d1_true, d1_pred, average='macro', zero_division=0)
    rec2 = recall_score(d2_true, d2_pred, average='macro', zero_division=0)

    f1_1 = f1_score(d1_true, d1_pred, average='macro', zero_division=0)
    f1_2 = f1_score(d2_true, d2_pred, average='macro', zero_division=0)

    # Jersey-level accuracy
    jersey_acc = np.mean((d1_true == d1_pred) & (d2_true == d2_pred))

    # ROC-AUC (One-vs-Rest)
    try:
        roc1 = roc_auc_score(
            np.eye(10)[d1_true], d1_probs, multi_class='ovr'
        )
    except ValueError:
        roc1 = None  # If not enough classes in batch

    try:
        roc2 = roc_auc_score(
            np.eye(10)[d2_true], d2_probs, multi_class='ovr'
        )
    except ValueError:
        roc2 = None

    results = {
        "acc1": acc1,
        "acc2": acc2,
        "prec1": prec1,
        "prec2": prec2,
        "rec1": rec1,
        "rec2": rec2,
        "f1_1": f1_1,
        "f1_2": f1_2,
        "jersey_acc": jersey_acc,
        "roc1": roc1,
        "roc2": roc2
    }

    return results
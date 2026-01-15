import torch

def predict(model, imgs):
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        imgs = imgs.to(device)
        p1, p2 = model(imgs)
        d1 = p1.argmax(dim=1)
        d2 = p2.argmax(dim=1)
        return d1 * 10 + d2

import torch
def predict(model, imgs):
    model.eval()
    with torch.no_grad():
        p1, p2 = model(imgs.cuda())
        d1 = p1.argmax(dim=1)
        d2 = p2.argmax(dim=1)
        return d1 * 10 + d2

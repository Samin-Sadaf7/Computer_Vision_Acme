import torch

def predict(model, dataloader, device, num_samples=5):
    model.eval()
    results = []

    with torch.no_grad():
        for imgs, d1, d2 in dataloader:
            imgs = imgs.to(device)

            o1, o2 = model(imgs)
            p1 = torch.argmax(o1, dim=1)
            p2 = torch.argmax(o2, dim=1)

            for i in range(len(p1)):
                gt = int(d1[i]) * 10 + int(d2[i])
                pred = int(p1[i]) * 10 + int(p2[i])
                results.append((gt, pred))

                if len(results) >= num_samples:
                    return results
    return results
import torch
from torch.utils.data import DataLoader

from dataset import JerseySequenceDataset
from model import TemporalJerseyNet
from train import train
from inference import predict


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def main():
    # -------------------------
    # Config
    # -------------------------
    root_dir = "temporal_jersey_nr_recognition_dataset_subset"
    batch_size = 4
    seq_len = 8
    epochs = 5   # start small for sanity check

    device = get_device()
    print(f"[INFO] Using device: {device}")

    # -------------------------
    # Dataset & DataLoader
    # -------------------------
    dataset = JerseySequenceDataset(root_dir, seq_len=seq_len)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,   # IMPORTANT for macOS stability
        pin_memory=False
    )

    # -------------------------
    # Model
    # -------------------------
    model = TemporalJerseyNet().to(device)
    print("[INFO] Model initialized")

    # -------------------------
    # Train
    # -------------------------
    train(model, dataloader, epochs=epochs)

    # -------------------------
    # Quick inference test
    # -------------------------
    imgs, _, _ = next(iter(dataloader))
    imgs = imgs.to(device)

    preds = predict(model, imgs)
    print("[INFO] Sample predictions:", preds.cpu().tolist())


if __name__ == "__main__":
    main()

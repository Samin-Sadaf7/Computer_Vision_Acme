import torch
from torch.utils.data import DataLoader, random_split
from dataset import JerseySequenceDataset

def get_dataloaders(
    root_dir,
    seq_len=8,
    batch_size=4,
    num_workers=2,
    train_ratio=0.7,
    val_ratio=0.15
):
    dataset = JerseySequenceDataset(root_dir, seq_len)

    total = len(dataset)
    train_size = int(train_ratio * total)
    val_size = int(val_ratio * total)
    test_size = total - train_size - val_size

    train_set, val_set, test_set = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size,
        shuffle=True, num_workers=num_workers
    )

    val_loader = DataLoader(
        val_set, batch_size=batch_size,
        shuffle=False, num_workers=num_workers
    )

    test_loader = DataLoader(
        test_set, batch_size=batch_size,
        shuffle=False, num_workers=num_workers
    )

    print(f"[INFO] Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

    return train_loader, val_loader, test_loader

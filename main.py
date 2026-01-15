from torch.utils.data import DataLoader
from dataloader import JerseySequenceDataset

def main():
    root_dir = "temporal_jersey_nr_recognition_dataset_subset"

    dataset = JerseySequenceDataset(root_dir, seq_len=8)

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2
    )

    for imgs, d1, d2 in dataloader:
        print("Images:", imgs.shape)
        print("Digit1:", d1)
        print("Digit2:", d2)
        break

if __name__ == "__main__":
    main()

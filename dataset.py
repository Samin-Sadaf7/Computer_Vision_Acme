import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class JerseySequenceDataset(Dataset):
    def __init__(self, root_dir, seq_len=8):
        self.samples = []
        self.seq_len = seq_len
        self.root_dir = root_dir

        for label in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label)

            if not os.path.isdir(label_path):
                continue

            # level-1: track id folders
            for track in os.listdir(label_path):
                track_path = os.path.join(label_path, track)

                if not os.path.isdir(track_path):
                    continue

                # level-2: sequence folders (0, 1, 2, ...)
                for seq in os.listdir(track_path):
                    seq_path = os.path.join(track_path, seq)

                    if not os.path.isdir(seq_path):
                        continue

                    frames = sorted([
                        os.path.join(seq_path, f)
                        for f in os.listdir(seq_path)
                        if f.lower().endswith(".jpg") and f != "anchor.jpg"
                    ])

                    if len(frames) == 0:
                        continue

                    self.samples.append((frames, int(label)))

        print(f"[INFO] Loaded {len(self.samples)} sequences.")

        self.transform = transforms.Compose([
            transforms.Resize((96, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frames, label = self.samples[idx]

        # pad or trim sequence
        if len(frames) >= self.seq_len:
            frames = frames[:self.seq_len]
        else:
            frames = frames + [frames[-1]] * (self.seq_len - len(frames))

        imgs = []
        for f in frames:
            img = Image.open(f).convert("RGB")
            imgs.append(self.transform(img))

        imgs = torch.stack(imgs)  # (T, C, H, W)

        digit1 = label // 10
        digit2 = label % 10

        return imgs, torch.tensor(digit1), torch.tensor(digit2)

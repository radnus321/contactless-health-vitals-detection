import torch


class SignalDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # (N, T, 1)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        print("X length:", len(self.X))
        print("y length:", len(self.y))
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

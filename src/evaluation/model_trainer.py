import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

from .features.signal_dataset import SignalDataset


class ModelTrainer:
    def __init__(self, signals, targets, model, save_dir="trained_models"):
        self.signals = signals
        self.targets = targets
        self.model = model

        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    # ---------------------------------------------------------
    # Safe reshape function (prevents squeeze issues)
    # ---------------------------------------------------------
    @staticmethod
    def flatten_output(t):
        """
        Ensures output is always shape (N,)
        Accepts input shapes like:
            (N, 1), (N,), (1,), scalar
        """
        t = t.reshape(-1)
        return t

    # ---------------------------------------------------------
    # TRAINING LOOP
    # ---------------------------------------------------------
    def train_model(self, epochs=50, batch_size=16, lr=1e-3,
                    val_split=0.2, shuffle=True):

        full_ds = SignalDataset(self.signals, self.targets)
        total = len(full_ds)

        val_size = int(total * val_split)
        train_size = total - val_size

        train_ds, val_ds = random_split(full_ds, [train_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):

            self.model.train()
            train_losses = []

            for batch_x, batch_y in train_loader:

                optimizer.zero_grad()

                preds = self.model(batch_x)

                preds = self.flatten_output(preds)
                batch_y = self.flatten_output(batch_y)

                loss = loss_fn(preds, batch_y)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            train_loss = sum(train_losses) / len(train_losses)
            val_loss = self.evaluate(val_loader)

            print(f"Epoch {epoch}/{epochs} | Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model("best_model.pth")

        self.save_model("last_model.pth")
        return best_val_loss

    # ---------------------------------------------------------
    # VALIDATION LOOP
    # ---------------------------------------------------------
    def evaluate(self, loader):
        self.model.eval()
        loss_fn = nn.MSELoss()
        losses = []

        preds_all = []
        targets_all = []

        with torch.no_grad():
            for batch_x, batch_y in loader:

                pred = self.model(batch_x)

                pred = self.flatten_output(pred)
                batch_y = self.flatten_output(batch_y)

                loss = loss_fn(pred, batch_y)
                losses.append(loss.item())

                preds_all.extend(pred.cpu().numpy().tolist())
                targets_all.extend(batch_y.cpu().numpy().tolist())

        mse = mean_squared_error(targets_all, preds_all)
        mae = mean_absolute_error(targets_all, preds_all)

        # r2 only valid if >1 sample
        if len(preds_all) > 1:
            r2 = r2_score(targets_all, preds_all)
        else:
            r2 = float("nan")

        print(f"    Validation metrics → MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

        return mse

    # ---------------------------------------------------------
    # SAVE MODEL
    # ---------------------------------------------------------
    def save_model(self, filename):
        save_path = os.path.join(self.save_dir, filename)
        torch.save(self.model.state_dict(), save_path)
        print(f"Saved model → {save_path}")

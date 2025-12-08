import os
import math
import time
import torch
import shutil
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim


def pearson_loss(preds, target, eps=1e-8):
    if preds.ndim > 2:
        preds = preds.view(preds.size(0), -1)
    if target.ndim > 2:
        target = target.view(target.size(0), -1)

    preds_mean = preds - preds.mean(dim=1, keepdim=True)
    target_mean = target - target.mean(dim=1, keepdim=True)

    num = (preds_mean * target_mean).sum(dim=1)
    den = torch.sqrt((preds_mean ** 2).sum(dim=1) * (target_mean ** 2).sum(dim=1) + eps)

    corr = num / (den + eps)
    loss = 1.0 - corr
    return loss.mean()


class SupervisedModelTrainer:
    def __init__(self, model, cfg=None, optimizer=None, scheduler=None, criterion=None, device=None):
        self.cfg = cfg or {}
        self.device = device or self._get_device_from_cfg()
        self.model = model.to(self.device)

        # build optimizer/scheduler/loss from cfg or use provided ones
        self.optimizer = optimizer or self._build_optimizer(self.model, self.cfg.get("optimizer", {}))
        self.scheduler = scheduler or self._build_scheduler(self.optimizer, self.cfg.get("scheduler", {}))
        self.criterion_cfg = self.cfg.get("loss", {"type": "mse"})
        self.criterion = criterion  # may be ignored; we use criterion_cfg

        train_cfg = self.cfg.get("training", {})
        self.epochs = train_cfg.get("epochs", 30)
        self.grad_clip = train_cfg.get("grad_clip", None)
        self.use_amp = train_cfg.get("amp", False)
        self.checkpoint_dir = self.cfg.get("checkpoint", {}).get("dir", "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.best_val_loss = float("inf")
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

    def _get_device_from_cfg(self):
        if self.cfg and "training" in self.cfg and "device" in self.cfg["training"]:
            device = self.cfg["training"]["device"]
            if device == "cuda" and not torch.cuda.is_available():
                return torch.device("cpu")
            return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_optimizer(self, model, opt_cfg):
        if not opt_cfg:
            return optim.Adam(model.parameters(), lr=1e-4)
        typ = opt_cfg.get("type", "adam").lower()
        params = opt_cfg.get("params", {"lr": 1e-4, "weight_decay": 0})
        if typ == "adam":
            return optim.Adam(model.parameters(), **params)
        if typ == "adamw":
            return optim.AdamW(model.parameters(), **params)
        if typ == "sgd":
            return optim.SGD(model.parameters(), **params)
        raise ValueError(f"Unknown optimizer type: {typ}")

    def _build_scheduler(self, optimizer, sch_cfg):
        if not sch_cfg:
            return None
        typ = sch_cfg.get("type", "step").lower()
        params = sch_cfg.get("params", {})
        if typ == "step":
            return optim.lr_scheduler.StepLR(optimizer, **params)
        if typ == "multistep":
            return optim.lr_scheduler.MultiStepLR(optimizer, **params)
        if typ == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, **params)
        if typ == "reduce":
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **params)
        raise ValueError(f"Unknown scheduler type: {typ}")

    def _compute_loss(self, preds, targets):
        if isinstance(preds, (tuple, list)):
            preds = preds[0]

        if preds.ndim == 3 and preds.shape[1] == 1:
            preds = preds.squeeze(1)
        if targets.ndim == 3 and targets.shape[1] == 1:
            targets = targets.squeeze(1)

        loss_type = self.criterion_cfg.get("type", "mse").lower()

        if loss_type == "mse":
            return nn.functional.mse_loss(preds, targets)
        if loss_type == "l1" or loss_type == "mae":
            return nn.functional.l1_loss(preds, targets)
        if loss_type == "pearson":
            return pearson_loss(preds, targets)
        if loss_type == "mse+pearson":
            w_mse = self.criterion_cfg.get("weight_mse", 1.0)
            w_p = self.criterion_cfg.get("weight_pearson", 1.0)
            return w_mse * nn.functional.mse_loss(preds, targets) + w_p * pearson_loss(preds, targets)
        if self.criterion is not None:
            return self.criterion(preds, targets)

        raise ValueError(f"Unknown loss type: {loss_type}")

    def _maybe_unwrap_model_output(self, out):
        if isinstance(out, (tuple, list)):
            return out[0]
        return out

    def train_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0
        n_batches = 0

        pbar = tqdm(dataloader, desc="Train", leave=False)
        for batch in pbar:
            inputs, targets = self._unpack_batch(batch)
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    out = self.model(inputs)
                    loss = self._compute_loss(out, targets)
                self.scaler.scale(loss).backward()
                if self.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                out = self.model(inputs)
                loss = self._compute_loss(out, targets)
                loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            running_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{running_loss / n_batches:.6f}"})

        avg_loss = running_loss / max(1, n_batches)
        if self.scheduler is not None and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            try:
                self.scheduler.step()
            except Exception:
                pass
        return avg_loss

    def validate_epoch(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Val", leave=False)
            for batch in pbar:
                inputs, targets = self._unpack_batch(batch)
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                out = self.model(inputs)
                loss = self._compute_loss(out, targets)

                running_loss += loss.item()
                n_batches += 1
                pbar.set_postfix({"val_loss": f"{running_loss / n_batches:.6f}"})

        avg_loss = running_loss / max(1, n_batches)
        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            try:
                self.scheduler.step(avg_loss)
            except Exception:
                pass
        return avg_loss

    def fit(self, train_loader, val_loader=None, start_epoch=1, epochs=None, resume_from=None):
        epochs = epochs or self.epochs
        start_epoch = start_epoch or 1

        if resume_from:
            self._load_checkpoint(resume_from)

        for epoch in range(start_epoch, start_epoch + epochs):
            t0 = time.time()
            train_loss = self.train_epoch(train_loader)
            t1 = time.time()

            val_loss = None
            if val_loader is not None:
                val_loss = self.validate_epoch(val_loader)

            lr = self.optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch} | Train {train_loss:.6f} | Val {val_loss if val_loss is None else format(val_loss, '.6f')} | lr {lr:.6e} | time {(t1-t0):.1f}s")

            save_every = self.cfg.get("checkpoint", {}).get("save_every", 1)
            save_best = self.cfg.get("checkpoint", {}).get("save_best_only", True)

            do_save = (save_every and epoch % save_every == 0)
            if val_loss is not None and save_best:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(epoch, val_loss, is_best=True)
            if do_save:
                self._save_checkpoint(epoch, val_loss or train_loss, is_best=False)

        print("Training finished.")

    def _save_checkpoint(self, epoch, metric, is_best=False):
        fname = f"epoch_{epoch:03d}_metric_{metric:.6f}.pth"
        path = os.path.join(self.checkpoint_dir, fname)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "metric": metric
        }, path)
        if is_best:
            bestpath = os.path.join(self.checkpoint_dir, "best_model.pth")
            shutil.copyfile(path, bestpath)
            print(f"Saved best model -> {bestpath}")
        else:
            print(f"Saved checkpoint -> {path}")

    def _load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] is not None and self.scheduler is not None:
            try:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except Exception:
                pass
        print(f"Loaded checkpoint from {path}")

    def _unpack_batch(self, batch):
        if isinstance(batch, dict):
            if "inputs" in batch and "targets" in batch:
                return batch["inputs"], batch["targets"]
            keys = list(batch.keys())
            return batch[keys[0]], batch[keys[1]]

        if isinstance(batch, (list, tuple)):
            if len(batch) >= 2:
                return batch[0], batch[1]
            raise ValueError("Batch tuple/list must have at least two elements: (inputs, targets)")

        raise ValueError("Unsupported batch format")


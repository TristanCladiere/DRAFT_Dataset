import torch
import os
import numpy as np


def train_val_one_epoch(model, pairs, optimizer, early_stopper, criterion, epoch, logger, writer):
    logger.info(f"Epoch {epoch:>6}")
    for phase in ['Train', 'Val']:
        if phase == 'Train':
            model.train()
        else:
            model.eval()
        with torch.set_grad_enabled(phase == 'Train'):
            outputs = model(pairs[phase])
        loss = criterion(outputs, phase)
        if phase == 'Train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        writer[phase].add_scalar(f"loss", loss, epoch)
        logger.info(f"\t{phase:>5} > loss: {loss:7.4f}")

    early_stopper(loss, epoch, model, optimizer)
    return epoch + 1


class EarlyStopper:
    def __init__(self, patience, best_path='best_model.pth', snapshot_path='snapshot.pth'):
        self.patience = patience
        self.best_loss = np.inf
        self.best_epoch = 0
        self.counter = 0
        self.early_stop = False
        self.best_path = best_path
        self.snapshot_path = snapshot_path
        if os.path.exists(self.snapshot_path):
            snapshot = torch.load(snapshot_path)
            self.best_loss = snapshot['best_loss']
            self.best_epoch = snapshot['best_epoch']
            self.counter = snapshot['counter']

    def __call__(self, val_loss, epoch, model, optimizer):
        is_best = False
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            is_best = True
        else:
            self.counter += 1
            if self.counter >= self.patience > 0:
                self.early_stop = True

        self.save_checkpoint(model, optimizer, epoch, is_best)

    def save_checkpoint(self, model, optimizer, epoch, is_best):
        checkpoint = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
            "best_loss": self.best_loss, "best_epoch": self.best_epoch, "counter": self.counter
        }
        torch.save(checkpoint, self.snapshot_path)
        if is_best:
            torch.save(checkpoint, self.best_path)

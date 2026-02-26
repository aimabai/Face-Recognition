# scheduler/cosine.py
import torch.optim as optim

def Scheduler(optimizer, max_epoch, **kwargs):
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_epoch,     # total number of epochs
        eta_min=1e-6         # final LR
    )
    return scheduler, 'epoch'

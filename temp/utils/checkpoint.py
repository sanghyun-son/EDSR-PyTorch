import torch
from pathlib import Path
def save_checkpoint(model, optimizer, epoch, val_loss, path):
    check = {}
    path = Path(path)
    check['model_state'] = model.state_dict()
    check['optim_state'] = optimizer.state_dict()
    check['epoch'] = epoch
    check['val_loss'] = val_loss
    if not path.parent.exists():
        path.parent.mkdir()
    torch.save(obj=check, f=path)
    print(f"Saved checkpoint epoch {epoch} at {path}")


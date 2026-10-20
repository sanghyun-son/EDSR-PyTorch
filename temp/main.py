import torch
from torch.utils.data import DataLoader

import time 

from script.dataset import SRDataset
from script.models.edsr import EDSR  # Adjust if needed
from script.loss import L1Loss
from script.trainer import Trainer

def main():
    device = torch.device( "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device = {device}")

    train_dataset = SRDataset(
        data_dir="data/edsr_data",
        split="train",
        scale=2,
        max_samples=32
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
    )


    model = EDSR(scale=2)
    
    criterion = L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4,)
    trainer = Trainer(model=model, criterion=criterion, optimizer = optimizer, device = device)  

    epoch_loss = trainer.train_epoch(train_loader)
    print(f"Training loss: {epoch_loss:.6f}")

    val_dataset = SRDataset(
            data_dir="data/edsr_data",
            split="val",
            scale=2,
            max_samples=8
            )
    val_loader = DataLoader(
            val_dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            )
    # validation_loss = trainer.validate(val_loader)
    # print(f"Validation loss: {validation_loss:.6f}")
    num_epochs = 3

    for epoch in range(num_epochs):
        start = time.time()
        train_loss = trainer.train_epoch(train_loader)
        end = time.time() - start
        val_loss = trainer.validate(val_loader)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"| Train Loss: {train_loss:.6f} "
            f"| Training time: {end} "
            f"| Val Loss: {val_loss:.6f}"
        )

if __name__=="__main__":
    main()

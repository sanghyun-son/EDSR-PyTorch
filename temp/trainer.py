import torch 
from torch import nn
class Trainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.model.to(self.device)
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        total_samples = 0 

        for lr, hr in train_loader:
            # TODO 1: Move lr and hr to self.device
            lr = lr.to(self.device)
            hr = hr.to(self.device)
            # TODO 2: Clear gradients
            self.optimizer.zero_grad()
            # TODO 3: Forward pass
            sr = self.model(lr)
            # TODO 4: Calculate loss
            loss = self.criterion(sr, hr)
            # TODO 5: Backpropagate
            loss.backward()
            # TODO 6: Update model parameters
            self.optimizer.step()
            # TODO 7: Accumulate the loss
            total_loss += loss.item()*lr.size(0)
            total_samples += lr.size(0)

        # TODO 8: Return average loss across batches
        return total_loss/total_samples

    def validate(self, eval_loader):
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for lr, hr in eval_loader:
                lr = lr.to(self.device)
                hr = hr.to(self.device)
                sr = self.model(lr)
                loss = self.criterion(sr, hr)
                total_loss += loss.item()*lr.size(0)
                total_samples += lr.size(0)

        return total_loss/total_samples

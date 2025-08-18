import torch
from torch import nn
import numpy as np
from typing import Any
from lightning import LightningModule
from src.models.components.utils import plot_MSE_val

class AnomReconModule(LightningModule):

    def __init__(
            self,
            net: nn.Module,
            loss: nn.Module,
            monitor: str = 'val_loss',
            lr_factor: float = .8,
            lr_patience: int = 20,
            lr_eps: float = 1e-8
    ) -> None:
        
        super().__init__()
        self.save_hyperparameters()
        self.loss = loss
        self.net = net
        self.monitor = monitor
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.lr_eps = lr_eps
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=self.lr_factor, patience=self.lr_patience, eps=self.lr_eps)

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': self.monitor
        }

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X)

    def model_step(self, batch: Any):
        X, Y = batch
        
        return self.loss(self(X), Y)
    
    def training_step(self, batch: Any):
        loss = self.model_step(batch)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch: Any):
        X, Y = batch
        out = self(X)
        loss = self.loss(out, Y)

        self.log('val_loss', loss)

        # log ACC
        #mean_acc = np.mean([np.corrcoef(out[i].cpu(), Y[i].cpu()) for i in range(len(out))])  # wrong!
        #self.logger.experiment.add_scalar('val_ACC', mean_acc, self.global_step)
        
        # log pictures
        if self.current_epoch > 59 and self.current_epoch % 30 == 0:
            plot_MSE_val(self, X, Y, out)

        return loss

    def test_step(self, batch: Any):
        loss = self.model_step(batch)
        self.log('test_loss', loss)

        return loss
    
    def on_load_checkpoint(self, checkpoint):
        checkpoint["optimizer_states"] = []
        checkpoint['lr_schedulers'] = []
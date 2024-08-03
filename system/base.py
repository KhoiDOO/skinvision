from .utils import parse_optimizer, parse_scheduler, parse_loss
from typing import *

import pytorch_lightning as pl
import torch
import os


class BaseSystem(pl.LightningModule):
    def __init__(self, cfg: Dict, *args: torch.Any, **kwargs: torch.Any) -> None:
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.criterion = parse_loss(self.cfg.loss)
    
    def set_save_dir(self, path:str):
        self.trial_dir = path
        os.makedirs(self.trial_dir)
    
    def configure_optimizers(self):
        if self.model is None:
            raise ValueError(f'self.model is not initialized')
        optimizer = parse_optimizer(self.cfg.optimizer, self.model)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": parse_scheduler(self.cfg.scheduler, optimizer)}
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()
        print('[INFO]: Experiment Started')
    
    def on_fit_end(self) -> None:
        super().on_fit_end()
        print('[INFO]: Experiment Ended')
        with open(os.path.join(self.trial_dir, 'done.txt'), 'w') as file:
            file.close()
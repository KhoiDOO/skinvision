from .base import BaseSystem
from typing import *
from torch import nn, Tensor
from .utils import GlobalAvgPooling, score, acc

import torch.nn.functional as F
import timm

class Sup(BaseSystem):
    def __init__(self, cfg: Dict, *args: Any, **kwargs: Any) -> BaseSystem:
        super().__init__(cfg, *args, **kwargs)

        self.save_hyperparameters() 

        self.model = timm.create_model(
            model_name=self.cfg.model.name, 
            pretrained=self.cfg.model.pretrained,
            features_only=True, out_indices=[-1]
        )
        
        self.gap = GlobalAvgPooling()
        self.linear = nn.Linear(self.model.feature_info.channels()[0], 1)
    
    def forward(self, image: Tensor) -> Tensor:
        lat = self.gap(self.model(image)[0])
        logit = F.sigmoid(self.linear(lat).squeeze(-1))

        return logit

    def training_step(self, batch_dct:Dict[str, Tensor]) -> Tensor:
        img, lbl = batch_dct['image'], batch_dct['target'].float()
        logit = self(image=img)

        loss = self.criterion(logit, lbl)
        
        parauc = score(logit=logit, label=lbl)
        accuracy = acc(logit=logit, label=lbl)

        self.log("tr/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("tr/score", parauc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("tr/acc", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch_dct:Dict[str, Tensor]):
        img, lbl = batch_dct['image'], batch_dct['target'].float()
        logit = self(image=img)

        loss = self.criterion(logit, lbl)
        
        parauc = score(logit=logit, label=lbl)
        accuracy = acc(logit=logit, label=lbl)

        self.log("vl/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("vl/score", parauc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("vl/acc", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch_dct:Dict[str, Tensor]):
        pass
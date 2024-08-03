from lightning.pytorch import loggers as pl_loggers
from pytorch_lightning import callbacks
from pytorch_lightning import Trainer
from system import BaseSystem, Sup
from data import DataModule
from datetime import datetime
from dataclasses import dataclass, field
from typing import *

import os

devices = [0]
now = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
save_dir = os.path.join(os.getcwd(), 'runs', now)

@dataclass
class Data:
    root: str = '/media/mountHDD3/data_storage/biomedical_data/isic/2024/img_crop_0'
    trdf_path: str = '/media/mountHDD3/data_storage/biomedical_data/isic/2024/train-metadata.csv'
    tsdf_path: str = '/media/mountHDD3/data_storage/biomedical_data/isic/2024/test-metadata.csv'
    crop: bool = True
    mid: int = 40
    # num_workers: int = 0 if len(devices) > 1 else 24
    # shuffle: bool = False if len(devices) > 1 else True
    num_workers: int = 0
    shuffle: bool = False
    batch_size: int = 128
    aug: bool = False

@dataclass
class Model:
    name: str = 'tf_efficientnet_b0.ns_jft_in1k'
    pretrained: bool = True

@dataclass
class Opt:
    name: str = 'Adam'
    args: Dict = field(default_factory=dict)

@dataclass
class Sched:
    name: str = 'MultiStepLR'
    args: Dict = field(default_factory=dict)

@dataclass
class Loss:
    name: str = 'BCELoss'
    args: Dict = field(default_factory=dict)

@dataclass
class Logger:
    names: list = field(default_factory=list)
    args: dict = field(default_factory=dict)

@dataclass
class Callback:
    names: list = field(default_factory=list)
    args: dict = field(default_factory=dict)

@dataclass
class Train:
    trainer: dict = field(default_factory=dict)
    logger: Logger = Logger(
        names=['WandbLogger', 'CSVLogger'],
        args={
            'wandb' : {
                'name': now,
                'project': 'isic_2024',
                'save_dir' : os.path.join(save_dir, 'wandb'),
                'id' : now,
                'anonymous': True,
                'log_model': 'all'
            },
            'csv' : {
                'name': 'csv',
                'save_dir': save_dir
            } 
        }
    )
    callback: Callback = Callback(
        names=['ModelCheckpoint'],
        args={
            'modelcp' : {
                'monitor': 'vl/score',
                'mode': 'max'
            }
        }
    )

@dataclass
class Args:
    seed: int = 0
    trial_dir: str = save_dir
    data: Data = Data()
    model: Model = Model()
    optimizer: Opt = Opt(name='Adam', args={'lr': 0.001})
    scheduler: Sched = Sched(name='MultiStepLR', args={'milestones': [2, 4, 8], 'gamma': 0.5, 'last_epoch': -1})
    loss: Loss = Loss(name='BCELoss', args={})
    train: Train = Train(
        trainer={
            # 'accelerator' : 'cpu',
            'devices': devices,
            'max_epochs': 20,
            'check_val_every_n_epoch': 1,
            'enable_progress_bar': True,
            'accumulate_grad_batches': 1,
            'log_every_n_steps': 50,
            'default_root_dir': save_dir
        }
    )

def parse_logger(cfg:Logger):
    _loggers = []
    for name, (_, item) in zip(cfg.names, cfg.args.items()):
        logger = getattr(pl_loggers, name)(**item)
        _loggers.append(logger)
    return _loggers

def parse_callback(cfg:Callback):
    _callbacks = []
    for name, (_, item) in zip(cfg.names, cfg.args.items()):
        callback = getattr(callbacks, name)(**item)
        _callbacks.append(callback)
    return _callbacks

def traincfg_resolve(cfg: Train):

    logger_lst = parse_logger(cfg.logger)
    callback_lst = parse_callback(cfg.callback)

    print(f'[INFO]: Loggers: {logger_lst}')
    print(f'[INFO]: Callbacks: {callback_lst}')

    return logger_lst, callback_lst


if __name__ == '__main__':

    cfg = Args()

    dm = DataModule(cfg=cfg)
    print(f'[INFO]: DataModule: {type(dm)}')

    system: BaseSystem = Sup(cfg)
    system.set_save_dir(cfg.trial_dir)
    print(f'[INFO]: SystemModule: {type(system)}')

    logger_lst, callback_lst = traincfg_resolve(cfg=cfg.train)
    trainer = Trainer(**cfg.train.trainer, logger=logger_lst, callbacks=callback_lst)

    trainer.fit(model=system, datamodule=dm)
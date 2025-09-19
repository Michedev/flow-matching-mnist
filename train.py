import pytorch_lightning as pl
import torch
from torchvision import transforms, datasets 
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
# from models. import ImageClassifierFlowMatching
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf

import logging

from paths import FID_PATH
from utils import calculate_fid_score, save_real_features_

logger = logging.getLogger(__name__)

OmegaConf.register_new_resolver('file_name', lambda x: x.split('.')[-2])

@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg: DictConfig):
    logger.info("Starting training process")
    logger.info(f"Configuration: {OmegaConf.to_yaml(cfg)}")
    
    # Data setup 
    logger.info("Setting up data loaders")
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    logger.info(f"Loading MNIST dataset from {cfg.data.data_dir}")
    train_data = datasets.MNIST(cfg.data.data_dir, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(cfg.data.data_dir, train=False, transform=transform)
    
    logger.info(f"Train dataset size: {len(train_data)}, Test dataset size: {len(test_data)}")
    logger.info(f"Batch size: {cfg.data.batch_size}")
    
    
    train_loader = DataLoader(train_data, batch_size=cfg.data.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=cfg.data.batch_size)
    if not FID_PATH.exists():
        if not FID_PATH.parent.exists(): FID_PATH.parent.mkdir(parents=True, exist_ok=True)
        fid = calculate_fid_score(train_loader, max_samples=30_000)
        save_real_features_(fid)
    logger.info(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # Model setup
    logger.info("Instantiating model")
    model = hydra.utils.instantiate(cfg.model)
    logger.info(f"Model class: {cfg.model['_target_']}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Lightning setup
    logger.info("Setting up PyTorch Lightning trainer")
    callbacks: list[pl.Callback] = [
        ModelCheckpoint(
            monitor=cfg.checkpoint.monitor, 
            filename=cfg.checkpoint.filename,
            save_top_k=cfg.checkpoint.save_top_k,
            mode=cfg.checkpoint.mode
        )
    ]
    logger.info(f"Checkpoint callback configured - monitoring: {cfg.checkpoint.monitor}")

    tb_logger = TensorBoardLogger(save_dir='.', name='training_logs')
    logger.info("TensorBoard logger configured")

    if cfg.early_stopping.enabled:
        callbacks.append(
            EarlyStopping(monitor=cfg.early_stopping.monitor, patience=cfg.early_stopping.patience)
        )
        logger.info(f"Early stopping enabled - monitoring: {cfg.early_stopping.monitor}, patience: {cfg.early_stopping.patience}")
    else:
        logger.info("Early stopping disabled")

    trainer = pl.Trainer(
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        max_epochs=cfg.training.max_epochs,
        callbacks=callbacks,
        limit_val_batches=cfg.training.limit_val_batches,
        logger=tb_logger
    )
    
    logger.info(f"Trainer configured - max_epochs: {cfg.training.max_epochs}, devices: {cfg.training.devices}, accelerator: {cfg.training.accelerator}")

    # Train
    logger.info("Starting training")
    trainer.fit(model, train_loader, test_loader)
    logger.info("Training completed")

if __name__ == '__main__':
    main()
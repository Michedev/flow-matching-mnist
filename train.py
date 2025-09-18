from models.flow_matching_model import ImageFlowMatcher
import pytorch_lightning as pl
from torchvision import transforms, datasets 
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from flow_matching_classifier import ImageClassifierFlowMatching
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    
    # Data args
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory to store the datasets')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training and testing')
    
    # Training args
    parser.add_argument('--max_epochs', type=int, default=10,
                        help='Maximum number of training epochs')
    parser.add_argument('--devices', type=int, default=1,
                        help='Number of GPUs to use')
    
    # Checkpoint args
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--early_stopping', action='store_true',
                        help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=3,
                        help='Early stopping patience')
    
    ImageFlowMatcher.add_model_specific_args(parser)
    
    return parser.parse_args()

def main():
    args = get_args()
    
    # Data setup 
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_data = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(args.data_dir, train=False, transform=transform)
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)

    # Model setup
    # model = ImageFlowMatcher(lr=args.lr, c_unet=args.c_unet)
    model = ImageClassifierFlowMatching(lr=args.lr)

    # Lightning setup
    callbacks = [
        ModelCheckpoint(
            monitor='train_loss', 
            dirpath=args.ckpt_dir,
            filename='flow-{epoch:02d}-{train_loss:.2f}',
            save_top_k=3,
            mode='min'
        )
    ]

    if args.early_stopping:
        callbacks.append(
            EarlyStopping(monitor='train_loss', patience=args.patience)
        )

    trainer = pl.Trainer(
        accelerator='auto',
        devices=args.devices,
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        limit_val_batches=0.1,

    )

    # Train
    trainer.fit(model, train_loader, test_loader)

if __name__ == '__main__':
    main()
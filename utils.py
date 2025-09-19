import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm
import logging
from pathlib import Path
from paths import FID_PATH

logger = logging.getLogger(__name__)

def save_real_features_(fid):
    state = dict(real_features_sum=fid.real_features_sum, real_features_cov_sum=fid.real_features_cov_sum, real_features_num_samples=fid.real_features_num_samples)
    
    # Ensure the directory exists
    FID_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the state dict to FID_PATH
    torch.save(state, FID_PATH)
    logger.info(f"FID real features saved to {FID_PATH}")
    
    return state

def calculate_fid_score(dloader, max_samples: int | None = None):
    logger.info("Starting FID score calculation")
    logger.info(f"Max samples: {max_samples if max_samples is not None else 'unlimited'}")
    
    fid = FrechetInceptionDistance(normalize=True)
    tot_samples: int = 0
    pbar = tqdm(dloader, desc="Calculating FID", total=max_samples)
    
    logger.info("Processing batches for FID calculation")
    for batch in pbar:
        if max_samples is not None and tot_samples >= max_samples:
            logger.info(f"Reached max samples limit: {max_samples}")
            break
        X, y = batch
        X = X.repeat(1, 3, 1, 1)  # Convert to RGB
        fid.update(X, real=True)
        tot_samples += len(X) 
        pbar.update(len(X) if max_samples is not None else 1)
    
    logger.info(f"FID calculation completed. Total samples processed: {tot_samples}")
    return fid

def load_fid_with_real_features(fid_path: str = None, device: torch.device = None):
    """
    Load FID metric with pre-computed real features from saved state.
    
    Args:
        fid_path (str, optional): Path to the saved FID state dict. Defaults to FID_PATH.
        device (torch.device, optional): Device to load the FID metric on
        
    Returns:
        FrechetInceptionDistance: FID metric with loaded real features
    """
    if fid_path is None:
        fid_path = FID_PATH
        
    logger.info(f"Loading FID metric with real features from {fid_path}")
    
    fid = FrechetInceptionDistance(normalize=True, reset_real_features=False)
    if device is not None:
        fid = fid.to(device)
    
    # Load the saved state dict
    state_dict = torch.load(fid_path, map_location=device)
    
    # Assign features directly
    fid.real_features_sum = state_dict['real_features_sum']
    fid.real_features_cov_sum = state_dict['real_features_cov_sum'] 
    fid.real_features_num_samples = state_dict['real_features_num_samples']
    
    logger.info("FID metric loaded successfully with real features")
    return fid
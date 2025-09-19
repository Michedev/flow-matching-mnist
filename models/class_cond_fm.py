from math import sqrt
from models.fm import ImageFlowMatcher
from models.velocity_architectures.unet_class_cond import UNet
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch.nn import functional as F
# flow_matching library imports
from flow_matching.path import CondOTProbPath
from flow_matching.solver import ODESolver
from models.utils import optimal_assignment
from torchmetrics.image.fid import FrechetInceptionDistance

from paths import FID_PATH
import logging

from utils import load_fid_with_real_features

logger = logging.getLogger(__name__)

class ModelWrapper(nn.Module):

    def __init__(self, model, gamma):
        super().__init__()
        self.model = model
        self.gamma = gamma

    def forward(self, x, t, y):

        cond_signal = self.model(x, t, y)
        uncond_signal = self.model(x, t, y*0)

        final_signal = self.gamma * cond_signal + uncond_signal * (1 - self.gamma)

        return final_signal

class ImageFlowMatcherClassCond(ImageFlowMatcher):

    def __init__(
        self,
        lr: float = 1e-4,
        c_unet: int = 32,
        input_channels: int = 1,
        num_classes: int = 10,
        gamma: float = 7.0,
        p_cond: float = 0.8,
    ):
        """
        Initialize the ImageFlowMatcher module.

        Args:
            lr (float, optional): Learning rate for the optimizer. Defaults to 1e-4.
            c_unet (float, optional): Channel scaling factor for the UNet. Determines model capacity. Defaults to 32.
            gamma (float, optional): Classifier-free guidance strength. 0.0 means no guidance. Defaults to 7.0.
        """
        super().__init__()
        self.save_hyperparameters()

        # Learning rate for the optimizer
        self.lr = lr
        self.num_classes = num_classes
        self.gamma = gamma
        assert 0.0 <= p_cond <= 1.0, f'{p_cond=}'
        self.p_cond = p_cond

        # Flag indicating whether to normalize MNIST images from [0, 1] to [-1, 1]
        self.normalize_data = True

        # Velocity field model (UNet). This predicts v(x, t).
        self.model = UNet(
            in_channels=input_channels,   # MNIST is grayscale, single channel
            out_channels=input_channels,  # We produce the same shape as input
            c=c_unet, num_classes=num_classes,
        )

        # Probability path for Flow Matching. 
        # CondOTProbPath provides the optimal transport between x_0 and x_1 conditioned on x_1. The sample() returns x(t) with 0 < t < 1 and the velocity term dx(t)/dt.
        self.cond_ot_path = CondOTProbPath()

        # Mean-squared error loss. We compare predicted velocity to dx(t)/dt from the path.
        self.criterion = nn.MSELoss()

        # FID metric for evaluation
        self.fid = load_fid_with_real_features(device=self.device)


    def forward(self, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the UNet to predict the velocity v(x(t), t).

        Args:
            x_t (torch.Tensor): State of the sample at time t, shape (B, C, H, W).
            t (torch.Tensor): Times, shape (B,), each in [0, 1]; can be broadcast as needed.
            y (torch.Tensor): Class labels, shape (B, num_classes) one-hot encoded.

        Returns:
            torch.Tensor: The predicted velocity field, shape (B, C, H, W).
        """
        logger.debug(f"Forward pass - x_t shape: {x_t.shape}, t shape: {t.shape}, y shape: {y.shape}")
        result = self.model(x_t, t, y)
        logger.debug(f"Forward pass - output shape: {result.shape}")
        return result

    def generate(
        self,
        batch_size: int = 32,
        sample_image_size: Tuple[int, int, int] = (1, 28, 28),
        num_steps: int = 2,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Generate images by integrating the learned velocity field from t=0 to t=1 using an ODE solver.

        The ODE is dx(t)/dt = v(x(t), t). We solve from x(0) ~ Gaussian to x(1), giving images in the data space.

        Args:
            batch_size (int, optional): Number of images to generate. Defaults to 32.
            sample_image_size (Tuple[int, int, int], optional): Shape of the generated sample, (C, H, W). Defaults to (1, 28, 28).
            num_steps (int, optional): Number of subdivisions between t=0 and t=1 used by the solver. Defaults to 2.

        Returns:
            torch.Tensor: Generated images of shape [batch_size, C, H, W].
        """
        if y is None:
            y = torch.randint(0, self.num_classes, size=(batch_size,), device=self.device)
        if y.ndim == 1:
            y = F.one_hot(y, num_classes=self.num_classes)
            y = y.float()
        
        logger.debug(f"Generate - y shape after processing: {y.shape}")
        
        velocity_model: ModelWrapper = ModelWrapper(self.model, self.gamma)
            
        # ODESolver integrates the learned velocity field v(x, t).
        solver = ODESolver(velocity_model=velocity_model)

        # Sample from Gaussian noise as the initial state x(0)
        x_init = torch.randn(batch_size, *sample_image_size, device=self.device)
        logger.debug(f"Generate - x_init shape: {x_init.shape}")

        # Create a time grid from t=0 to t=1. The solver will integrate over these steps.
        time_grid = torch.linspace(0, 1, steps=num_steps, device=self.device)
        logger.debug(f"Generate - time_grid shape: {time_grid.shape}")

        # Solve the ODE dx/dt = v(x, t) from t=0 to t=1.
        generated_samples = solver.sample(
            time_grid=time_grid,
            x_init=x_init,
            method='dopri5',   # dopri5 is an adaptive Runge-Kutta method.
            step_size=None,    # None -> let dopri5 adapt step size internally.
            atol=1e-4,
            rtol=1e-4,
            y=y
        )
        assert isinstance(generated_samples, torch.Tensor)
        logger.debug(f"Generate - generated_samples shape before normalization: {generated_samples.shape}")

        # If data was normalized to [-1,1], invert that normalization back to [0,1].
        if self.normalize_data:
            generated_samples = (generated_samples + 1) / 2  # from [-1, 1] to [0, 1]
            generated_samples = torch.clamp(generated_samples, 0.0, 1.0)

        logger.debug(f"Generate - final generated_samples shape: {generated_samples.shape}")
        return generated_samples

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """
        Perform a single training step of the Flow Matching objective.

        The goal is to minimize MSE between the model's velocity v(x(t), t) and the
        'true' derivative dx(t)/dt of the path from x₀ (noise) to x₁ (data image).

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A batch of MNIST data (images, labels).
            batch_idx (int): Index of this batch.

        Returns:
            torch.Tensor: The MSE loss for this batch.
        """
        # Extract images from the batch. (labels are not used for unconditional generation)
        batch_images, y = batch
        logger.debug(f"Training step - batch_images shape: {batch_images.shape}, y shape: {y.shape}")
        
        if y.ndim == 1:
            y = F.one_hot(y, self.num_classes)
            y = y.float()
        y = y * (torch.rand(len(y), 1, device=self.device) < self.p_cond).float()
        logger.debug(f"Training step - y shape after processing: {y.shape}")
        
        with torch.no_grad():
            # Optionally normalize images to [-1,1] for stable training
            if self.normalize_data:
                batch_images = batch_images * 2.0 - 1.0

            # Sample x₀ from Gaussian noise
            x_0 = torch.randn_like(batch_images)
            x_0 = optimal_assignment(x_0, batch_images)
            logger.debug(f"Training step - x_0 shape: {x_0.shape}")

            # Sample times t from Uniform[0,1]
            t = torch.rand(batch_images.size(0), device=self.device)
            logger.debug(f"Training step - t shape: {t.shape}")

            # Obtain sample x(t) and the relative velocity dx(t)/dt that interpolates between x_0 (prior distribution) and x_1 (data distribution)
            path_batch = self.cond_ot_path.sample(x_0, batch_images, t)
            logger.debug(f"Training step - path_batch.x_t shape: {path_batch.x_t.shape}, path_batch.dx_t shape: {path_batch.dx_t.shape}")

        # Predict the velocity at x(t)
        predicted_velocity = self.forward(path_batch.x_t, path_batch.t, y)
        logger.debug(f"Training step - predicted_velocity shape: {predicted_velocity.shape}")

        # neural network prediction matches the true velocity with mse
        loss = self.criterion(predicted_velocity, path_batch.dx_t)
        logger.debug(f"Training step - loss: {loss.item()}")

        # Log training loss for monitoring
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        """
        Perform a validation step for the Flow Matching objective.

        Similar to training_step, but logs 'val_loss' instead, and can also generate sample images.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A batch of MNIST data (images, labels).
            batch_idx (int): Index of this validation batch.
        """
        # Extract images (labels are not used)
        batch_images, y = batch
        logger.debug(f"Validation step - batch_images shape: {batch_images.shape}, y shape: {y.shape}")
        
        if y.ndim == 1:
            y = F.one_hot(y, self.num_classes)
        # Normalize images for consistency
        if self.normalize_data:
            batch_images = batch_images * 2.0 - 1.0

        # (Optional) Generate sample images on the first validation batch of each epoch
        sample_images = self.generate(batch_size=len(batch_images), num_steps=10)
        logger.debug(f"Validation step - sample_images shape: {sample_images.shape}")
        
        grid = torchvision.utils.make_grid(sample_images, nrow=int(sqrt(len(sample_images))), normalize=False, value_range=(0, 1))
        logger.debug(f"Validation step - grid shape: {grid.shape}")
        
        if batch_idx == 0:
            self.logger.experiment.add_image('generated_images', grid, self.current_epoch)

        generated_images_rgb = sample_images.repeat(1, 3, 1, 1)  # Convert to RGB
        logger.debug(f"Validation step - generated_images_rgb shape: {generated_images_rgb.shape}")

        # Update FID with real and generated images
        # self.fid.update(real_images, real=True)
        self.fid.update(generated_images_rgb, real=False)
        
    def on_validation_epoch_end(self) -> None:
        fid_score = self.fid.compute()
        self.log('fid_score', fid_score, prog_bar=True, logger=True)
        self.fid.reset_real_features = False
        self.fid.reset()

    def configure_optimizers(self):
        """
        Configure and return the optimizer for the velocity field parameters.

        Returns:
            torch.optim.Optimizer: The optimizer used for training the UNet.
        """
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
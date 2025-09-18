from typing import Optional, Tuple
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import matplotlib.pyplot as plt

# flow_matching library imports
from flow_matching.path import CondOTProbPath
from flow_matching.solver import ODESolver

# Local imports
from unet import UNet


class ImageFlowMatcher(pl.LightningModule):
    """
    Unconditional Flow Matching model for MNIST image generation using the `flow_matching` library.

    """

    def __init__(
        self,
        lr: float = 1e-4,
        c_unet: float = 32
    ):
        """
        Initialize the ImageFlowMatcher module.

        Args:
            lr (float, optional): Learning rate for the optimizer. Defaults to 1e-4.
            c_unet (float, optional): Channel scaling factor for the UNet. Determines model capacity. Defaults to 32.
        """
        super().__init__()
        self.save_hyperparameters()

        # Learning rate for the optimizer
        self.lr = lr

        # Flag indicating whether to normalize MNIST images from [0, 1] to [-1, 1]
        self.normalize_data = True

        # Velocity field model (UNet). This predicts v(x, t).
        self.model = UNet(
            in_channels=1,   # MNIST is grayscale, single channel
            out_channels=1,  # We produce the same shape as input
            c=c_unet
        )

        # Probability path for Flow Matching. 
        # CondOTProbPath provides the optimal transport between x_0 and x_1 conditioned on x_1. The sample() returns x(t) with 0 < t < 1 and the velocity term dx(t)/dt.
        self.cond_ot_path = CondOTProbPath()

        # Mean-squared error loss. We compare predicted velocity to dx(t)/dt from the path.
        self.criterion = nn.MSELoss()


    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the UNet to predict the velocity v(x(t), t).

        Args:
            x_t (torch.Tensor): State of the sample at time t, shape (B, C, H, W).
            t (torch.Tensor): Times, shape (B,), each in [0, 1]; can be broadcast as needed.

        Returns:
            torch.Tensor: The predicted velocity field, shape (B, C, H, W).
        """
        return self.model(x_t, t)

    def generate(
        self,
        batch_size: int = 32,
        sample_image_size: Tuple[int, int, int] = (1, 28, 28),
        num_steps: int = 2
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
        # ODESolver integrates the learned velocity field v(x, t).
        solver = ODESolver(velocity_model=self.model)

        # Sample from Gaussian noise as the initial state x(0)
        x_init = torch.randn(batch_size, *sample_image_size, device=self.device)

        # Create a time grid from t=0 to t=1. The solver will integrate over these steps.
        time_grid = torch.linspace(0, 1, steps=num_steps, device=self.device)

        # Solve the ODE dx/dt = v(x, t) from t=0 to t=1.
        generated_samples = solver.sample(
            time_grid=time_grid,
            x_init=x_init,
            method='dopri5',   # dopri5 is an adaptive Runge-Kutta method.
            step_size=None,    # None -> let dopri5 adapt step size internally.
            atol=1e-4,
            rtol=1e-4
        )

        # If data was normalized to [-1,1], invert that normalization back to [0,1].
        if self.normalize_data:
            generated_samples = (generated_samples + 1) / 2  # from [-1, 1] to [0, 1]
            generated_samples = torch.clamp(generated_samples, 0.0, 1.0)

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
        batch_images, _ = batch

        # Optionally normalize images to [-1,1] for stable training
        if self.normalize_data:
            batch_images = batch_images * 2.0 - 1.0

        # Sample x₀ from Gaussian noise
        x_0 = torch.randn_like(batch_images)

        # Sample times t from Uniform[0,1]
        t = torch.rand(batch_images.size(0), device=self.device)

        # Obtain sample x(t) and the relative velocity dx(t)/dt that interpolates between x_0 (prior distribution) and x_1 (data distribution)
        path_batch = self.cond_ot_path.sample(x_0, batch_images, t)

        # Predict the velocity at x(t)
        predicted_velocity = self.forward(path_batch.x_t, path_batch.t)

        # neural network prediction matches the true velocity with mse
        loss = self.criterion(predicted_velocity, path_batch.dx_t)

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
        batch_images, _ = batch

        # Normalize images for consistency
        if self.normalize_data:
            batch_images = batch_images * 2.0 - 1.0

        # Sample initial noise x₀
        x_0 = torch.randn_like(batch_images)

        # Sample times t in [0,1]
        t = torch.rand(batch_images.size(0), device=self.device)

        # Get the path data: x(t), dx(t)/dt
        path_batch = self.cond_ot_path.sample(x_0, batch_images, t)

        # Predict velocity at x(t)
        predicted_velocity = self.forward(path_batch.x_t, path_batch.t)

        # Compute MSE loss
        loss = self.criterion(predicted_velocity, path_batch.dx_t)

        # Log validation loss
        self.log('val_loss', loss, prog_bar=True, logger=True)

        # (Optional) Generate sample images on the first validation batch of each epoch
        if batch_idx == 0:
            sample_images = self.generate(batch_size=16, num_steps=10)
            grid = torchvision.utils.make_grid(sample_images, nrow=4, normalize=False, value_range=(0, 1))
            self.logger.experiment.add_image('generated_images', grid, self.current_epoch)

            # Visualize in a notebook or interactive environment (optional)
            plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
            plt.show()
            plt.close()

    def configure_optimizers(self):
        """
        Configure and return the optimizer for the velocity field parameters.

        Returns:
            torch.optim.Optimizer: The optimizer used for training the UNet.
        """
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parser):
        """
        Add model-specific command-line arguments to an existing parser.

        Args:
            parser (argparse.ArgumentParser): The main or subparser in which to add these arguments.

        Returns:
            argparse.ArgumentParser: The parser updated with Flow Matcher arguments.
        """
        group = parser.add_argument_group('Flow Matcher Arguments')
        group.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the Flow Matcher model', dest='lr')
        group.add_argument('--c-unet', default=32, type=float, help='Channels of UNet model', dest='c_unet')
        return parser
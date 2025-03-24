from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from VAE import VAE
# Abstract base class for interpolation
class Interpolation(ABC):
    
    @abstractmethod
    def interpolate(self, img1, img2, alpha):
        """
        Abstract method to interpolate between two images.
        Parameters:
            img1: The first image.
            img2: The second image.
            alpha: Interpolation factor (0.0 <= alpha <= 1.0).
        """
        pass
    def name(self):
        return "Interpolation"

# Linear interpolation subclass
class LinearInterpolation(Interpolation):
    
    def interpolate(self, img1, img2, alpha):
        """
        Performs linear interpolation between two images.
        """
        return (1 - alpha) * img1 + alpha * img2
    def name(self):
        return "LinearInterpolation"

# VAE interpolation subclass (just a template, needs a VAE model)
class VAEInterpolation(Interpolation):
    
    def __init__(self, vae_model):
        """
        Initialize with a VAE model.
        """
        self.vae_model = vae_model
    
    def interpolate(self, img1, img2, alpha):
        """
        Interpolates between two images using VAE latent space.
        """
        # Encode images into latent space
        z1, _ = self.vae_model.encode(img1)
        z2, _ = self.vae_model.encode(img2)
        
        # Interpolate between the latent variables
        z_interp = (1 - alpha) * z1 + alpha * z2
        
        # Decode the interpolated latent representation back to an image
        return self.vae_model.decode(z_interp)
    def name(self):
        return "VAEInterpolation"

# Example usage:
# Initialize the models and images
if __name__=='__main__':
    vae_model = VAE()  # Your pre-trained VAE model
    linear_interp = LinearInterpolation()
    vae_interp = VAEInterpolation(vae_model)

    img1 = torch.randn(1, 3, 64, 64)  # Example image 1
    img2 = torch.randn(1, 3, 64, 64)  # Example image 2
    alpha = 0.5  # Interpolation factor

    # Linear interpolation
    linear_result = linear_interp.interpolate(img1, img2, alpha)

    # VAE interpolation
    vae_result = vae_interp.interpolate(img1, img2, alpha)

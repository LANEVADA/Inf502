import os
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np

image_size=128
batch_size = 32

class ImageFolderSplitter(Dataset):
    def __init__(self, base_dir, train_size=20, test_size=5, split='train', transform=None):
        """
        :param base_dir: Directory where all the image subfolders are located.
        :param train_size: Number of images to use for training from each folder.
        :param test_size: Number of images to use for testing from each folder.
        :param split: Either 'train' or 'test', decides which split to load.
        :param transform: Optional transform to be applied on a sample.
        """
        self.base_dir = base_dir
        self.train_size = train_size
        self.test_size = test_size
        self.split = split
        self.transform = transform

        # List all subfolders (each folder is a class)
        self.subfolders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

        self.image_paths = []
        for folder in self.subfolders:
            folder_path = os.path.join(base_dir, folder)

            # Get all image files in the current folder (assuming images are .jpg, .png, etc.)
            images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            random.shuffle(images)

            if self.split == 'train':
                selected_images = images[:self.train_size]
            elif self.split == 'test':
                selected_images = images[self.train_size:self.train_size+self.test_size]
            else:
                raise ValueError("split must be 'train' or 'test'")

            # Append the paths of selected images
            for image in selected_images:
                image_path = os.path.join(folder_path, image)
                self.image_paths.append((image_path, folder))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path, label = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

# Set up transformations
transform = transforms.Compose([
    transforms.Resize((image_size,image_size)),  # Resize to a smaller size for efficiency
    transforms.ToTensor(),
])

# Set up Dataset



class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 3x128x128 -> 32x64x64
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 32x64x64 -> 64x32x32
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 64x32x32 -> 128x16x16
            nn.ReLU(),
        )
        
        # Calculate the flattened size of the output from the encoder (feature map size)
        dummy_input = torch.zeros(1, 3, image_size,image_size)  # Batch size 1, 3 channels, 128x128 image
        with torch.no_grad():
            feature_map = self.encoder(dummy_input)  # Pass through the encoder
        flattened_size = feature_map.numel()  # Get the total number of elements in the feature map
        
        # Latent space (mu and logvar for variational distribution)
        self.fc_mu = nn.Linear(flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(flattened_size, latent_dim)
        
        # Decoder
        self.fc_dec = nn.Linear(latent_dim, flattened_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 128x8x8 -> 64x16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 64x16x16 -> 32x32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32x32x32 -> 3x64x64
            nn.Sigmoid()  # Sigmoid to get values between 0 and 1
        )
    
    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)  # Flatten the output
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h = self.fc_dec(z)
        
        # Dynamically reshape based on flattened size
        h = h.view(h.size(0), 128, (int)(image_size/8),(int)(image_size/8))  # Adjust based on the encoder output size
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL divergence between the learned distribution and a standard normal
    # KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # The factor of 0.5 comes from integrating over the continuous space
    # where the variance is expected to be 1.
    # -0.5 * sum(1 + log(var) - mean^2 - var)
    # 'sum' reduction computes the total for the batch
    # Make sure your input images are normalized to [0, 1] to use BCE
    # If using different loss formulations like MSE, this might need adjustments
    MSE_loss = nn.MSELoss(reduction='sum')
    recon_loss = MSE_loss(recon_x.view(-1, 3 * 64 * 64), x.view(-1, 3 * 64 * 64))
    
    # Calculate KL divergence loss
    KL_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + KL_divergence

def train(device,num_epochs,train_dataloader,test_dataloader):

    model = VAE(latent_dim=1024).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, (data, _) in enumerate(train_dataloader):
            data = data.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            # Forward pass
            recon_batch, mu, logvar = model(data)
            # Compute the loss
            #loss = loss_function(recon_batch, data, mu, logvar)
            
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            # Backward pass
        
            
            # Update weights
            
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 5 == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Batch [{batch_idx}/{len(train_dataloader)}], Loss: {loss.item()},Generator loss: {loss_G.item()}, Discriminator loss: {loss_D.item()}")

        print(f"Epoch [{epoch}/{num_epochs}] Training Loss: {train_loss / len(train_dataloader)}")

        # Test the model
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for data, _ in test_dataloader:
                data = data.to(device)
                recon_batch, mu, logvar = model(data)
                loss = loss_function(recon_batch, data, mu, logvar)
                test_loss += loss.item()

        print(f"Epoch [{epoch}/{num_epochs}] Test Loss: {test_loss / len(test_dataloader)}")
        if epoch % 10 == 0:
            # Save the model checkpoints
            torch.save(model.state_dict(), f"models/vae/{epoch}.pth")
            print("Model saved!")
        


# Function to perform the interpolation
def interpolate_and_display(model,img1, img2, num_steps=10):
    # Move the model and images to the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    img1 = img1.to(device).unsqueeze(0)  # Unsqueeze to add a batch dimension
    img2 = img2.to(device).unsqueeze(0)  # Unsqueeze to add a batch dimension

    # Encode the two images to get their latent vectors
    mu1, logvar1 = model.encode(img1)
    mu2, logvar2 = model.encode(img2)

    # Reparameterize to get the latent vectors (z1 and z2)
    z1 = model.reparameterize(mu1, logvar1)
    z2 = model.reparameterize(mu2, logvar2)
    
    # Interpolate between z1 and z2
    interpolated_images = []
    interpolated_images.append(img1.cpu().detach().numpy())
    for alpha in np.linspace(0, 1, num_steps):
        z_interpolated = (1 - alpha) * z1 + alpha * z2  # Linear interpolation
        recon_image = model.decode(z_interpolated)  # Decode the interpolated latent vector
        # recon_image=(recon_image+1)/2
        # recon_image = (recon_image - recon_image.min()) / (recon_image.max() - recon_image.min())

        interpolated_images.append(recon_image.cpu().detach().numpy())

    # Plot the interpolated images
    fig, axes = plt.subplots(1, num_steps, figsize=(15, 3))
    for i, ax in enumerate(axes):
        ax.imshow(np.transpose(interpolated_images[i][0], (1, 2, 0)))  # Convert from (C, H, W) to (H, W, C)
        ax.axis('off')
    plt.show()
    


# Example of how to use the function after training
# Assume you have your trained model and some sample images


if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs=100
    base_dir = 'data/images/Images'  # Path to the base folder containing subfolders for classes
    train_dataset = ImageFolderSplitter(base_dir=base_dir, split='train', transform=transform)
    test_dataset = ImageFolderSplitter(base_dir=base_dir, split='test', transform=transform)

    # Set up DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    train(device,num_epochs)
    model = VAE(latent_dim=1024).to(device)
    model.load_state_dict(torch.load("models/vae/100.pth"))
    img1, _ = next(iter(train_dataloader))  # Example: first image from the train dataset
    img2, _ = next(iter(train_dataloader)) 
    interpolate_and_display(model,img1, img2, num_steps=10)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
import cv2

from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg = models.vgg19(weights=True).features.eval().to(device)

def load_video_frames(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the frame from BGR (OpenCV default) to RGB (PyTorch format)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL image for better transformation compatibility
        frame_pil = Image.fromarray(frame_rgb)
        
        # Transform the frame to a tensor and normalize
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to the size used by VGG19
            transforms.ToTensor(),  # Convert to tensor with shape (C, H, W)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize like VGG
        ])
        
        frame_tensor = transform(frame_pil).unsqueeze(0)  # Add batch dimension (1, 3, 224, 224)
        frames.append(frame_tensor)
    
    cap.release()
    return frames

def get_vgg_features(layer_ids=[2, 7, 12]):
    """ Extracts certain layers of VGG19 to compute the perceptual loss. """
    layers = nn.ModuleList([vgg[i] for i in layer_ids])
    for param in layers.parameters():
        param.requires_grad = False 
    return layers

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # Load pre-trained VGG19 model (use the features only, no classifier)
        self.vgg19 = models.vgg19(pretrained=True).features
        self.vgg19.eval()  # Set the model to evaluation mode
        for param in self.vgg19.parameters():
            param.requires_grad = False  # No need to backpropagate through VGG19

    def forward(self, x, y):
        # Pass the images through VGG19 and compute the perceptual loss
        x = x.to(device)
        y = y.to(device)
        x_features = self.vgg19(x)
        y_features = self.vgg19(y)
        loss = F.mse_loss(x_features, y_features)
        return loss

def compute_perceptual_loss(video_frames, perceptual_loss_fn):
    """ Compute the loss of a video by appling the MSE loss between the features of the two frames at each layer. """
    loss = 0
    for t in range(len(video_frames) - 1):
        loss += perceptual_loss_fn(video_frames[t], video_frames[t + 1])
    return loss / (len(video_frames) - 1)  # Average over all the transitions between frames

if __name__=="__main__":
    perceptual_loss_fn = PerceptualLoss()

    # Load frames from a .avi file
    video_path = "../outputs_iter/output.avi"  # Path to your .avi file
    frames = load_video_frames(video_path)
    print("video loaded")

    # Compute the perceptual loss between consecutive frames in the video
    total_loss = compute_perceptual_loss(frames, perceptual_loss_fn)

    # Print the perceptual loss for the video sequence
    print(f"Perceptual Loss between consecutive frames: {total_loss.item()}")

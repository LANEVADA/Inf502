import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torchvision.models import VGG19_Weights
from transformers import CLIPProcessor, CLIPModel
from torchvision.transforms import ToTensor
import cv2
from PIL import Image

# weights for the total loss
alpha = 1/3
beta = 1/3
gamma = 1/3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vgg = models.vgg19(weights=VGG19_Weights.DEFAULT).features.eval().to(device)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def transform_pil_to_tensor(pil_img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Apply the transformations
    image_tensor = transform(pil_img).unsqueeze(0)
    image_tensor = image_tensor.to(device)
    return image_tensor

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform_pil_to_tensor(image)
    image_tensor = image_tensor.to(device)
    return image_tensor

def load_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frames_pil = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        frames_pil.append(frame_pil)
        frame_tensor = transform_pil_to_tensor(frame_pil)
        frames.append(frame_tensor)
    
    cap.release()
    return frames, frames_pil

def compute_perceptual_loss_img(img1, img2):
    """Compute the perceptual loss between two images"""
    with torch.no_grad():
        features_img1 = vgg(img1)
        features_img2 = vgg(img2)
    loss = F.mse_loss(features_img1, features_img2)
    return loss
    
def compute_clip_similarity(text, image):
    inputs = clip_processor(text=text, images=image, return_tensors="pt", padding=True).to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    outputs = clip_model(**inputs)
    image_features = outputs.image_embeds
    text_features = outputs.text_embeds
    similarity = F.cosine_similarity(image_features, text_features)
    return similarity.mean().item()

def evaluate_text_video_coherence(video_frames_pil, text_description):
    text_coherence = 0
    for frame in video_frames_pil:
        text_coherence += compute_clip_similarity(text_description, frame)
    text_coherence /= len(video_frames_pil) # Average text coherence
    return text_coherence

def evaluate_video_consistency(video_frames):
    """Compute the loss of a video by applying the compute_perceptual_loss_img between each frame of the video"""
    frame_consistency = 0
    for i in range(len(video_frames) - 1):
        frame_consistency += compute_perceptual_loss_img(video_frames[i], video_frames[i+1])
    frame_consistency /= (len(video_frames) - 1) # Average frame consistency
    return frame_consistency

def evaluate_coherence(video_frames, original_image, text_description, video_frames_pil):
    perceptual_loss = compute_perceptual_loss_img(original_image, video_frames[0])
    text_coherence = evaluate_text_video_coherence(video_frames_pil, text_description)
    frame_consistency = evaluate_video_consistency(video_frames)
    total_score = alpha * 1/(1 + perceptual_loss) + beta * text_coherence + gamma * 1/(1 + frame_consistency)
    return total_score, perceptual_loss, text_coherence, frame_consistency

# Example usage:
if __name__=="__main__":

    # Load frames from a .avi file
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    video_path = "outputs/output.avi"  # Path to your .avi file
    video_frames, video_frames_pil = load_video_frames(video_path)
    print(len(video_frames))
    original_image = load_image("images/test2.jpg")
    text_description = "Beautiful mountain landscape."
    print("video loaded")

    coherence_score, perceptual_loss_value, text_coherence_value, frame_consistency_value = evaluate_coherence(video_frames, original_image, text_description, video_frames_pil)

    print(f"Total Coherence Score: {coherence_score}")
    print(f"Perceptual Loss: {perceptual_loss_value}")
    print(f"Text Coherence: {text_coherence_value}")
    print(f"Frame Consistency: {frame_consistency_value}")

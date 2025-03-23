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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vgg = models.vgg19(weights=VGG19_Weights.DEFAULT).features.eval().to(device)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def transform_pil_to_tensor(pil_img):
    # Define the necessary transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to the size the model expects
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization for VGG or CLIP models
    ])
    
    # Apply the transformations
    image_tensor = transform(pil_img).unsqueeze(0)  # Add batch dimension (1, C, H, W)
    
    return image_tensor

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")  # Convert to RGB in case it's grayscale or has an alpha channel
    
    image_tensor = transform_pil_to_tensor(image)
    
    return image_tensor

def load_video_frames(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    frames = []
    frames_pil = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the frame from BGR (OpenCV default) to RGB (PyTorch format)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL image for better transformation compatibility
        frame_pil = Image.fromarray(frame_rgb)
        frames_pil.append(frame_pil)
        
        frame_tensor = transform_pil_to_tensor(frame_pil)
        frames.append(frame_tensor)
    
    cap.release()
    return frames, frames_pil

def compute_perceptual_loss_img(img1, img2):
    with torch.no_grad():
        features_img1 = vgg(img1)
        features_img2 = vgg(img2)
    
    # L2 Loss (Mean Squared Error) between feature maps
    loss = F.mse_loss(features_img1, features_img2)
    return loss
    
def compute_clip_similarity(text, image):
    inputs = clip_processor(text=text, images=image, return_tensors="pt", padding=True).to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    outputs = clip_model(**inputs)
    image_features = outputs.image_embeds
    text_features = outputs.text_embeds
    
    # Compute cosine similarity
    similarity = F.cosine_similarity(image_features, text_features)
    return similarity.mean().item()

def evaluate_text_video_coherence(video_frames_pil, text_description):
    text_coherence = 0
    for frame in video_frames_pil:
        text_coherence += compute_clip_similarity(text_description, frame)
    text_coherence /= len(video_frames_pil) # Average text coherence
    print("text coherence : ", text_coherence)
    return text_coherence

def evaluate_video_consistency(video_frames):
    """ Compute the loss of a video by appling the MSE loss between the features of the two frames at each layer. """
    frame_consistency = 0
    for i in range(len(video_frames) - 1):
        frame_consistency += compute_perceptual_loss_img(video_frames[i], video_frames[i+1])
    frame_consistency /= (len(video_frames) - 1) # Average frame consistency
    print("frame_consistency : ", frame_consistency)
    return frame_consistency


def evaluate_coherence(video_frames, original_image, text_description, video_frames_pil):
    # Step 1: Calculate the perceptual loss between the first frame and the original image
    perceptual_loss = compute_perceptual_loss_img(original_image, video_frames[0])
    print("first step done")
    
    # Step 2: Compute the coherence with the text for each frame
    text_coherence = evaluate_text_video_coherence(video_frames_pil, text_description)
    
    # Step 3: Compute the perceptual loss between consecutive frames for smoothness
    frame_consistency = evaluate_video_consistency(video_frames)
    
    # Combine all the scores into a final coherence score
    total_score = perceptual_loss + text_coherence + frame_consistency
    
    return total_score, perceptual_loss, text_coherence, frame_consistency

if __name__=="__main__":

    # Load frames from a .avi file
    video_path = "../outputs_iter/output.avi"  # Path to your .avi file
    video_frames, video_frames_pil = load_video_frames(video_path)
    original_image = load_image("../images/test.jpg")
    text_description = "A Summer night at the beach"
    print("video loaded")

    coherence_score, perceptual_loss_value, text_coherence_value, frame_consistency_value = evaluate_coherence(video_frames, original_image, text_description, video_frames_pil)

    # Print the perceptual loss for the video sequence
    print(f"Total Coherence Score: {coherence_score}")
    print(f"Perceptual Loss: {perceptual_loss_value}")
    print(f"Text Coherence: {text_coherence_value}")
    print(f"Frame Consistency: {frame_consistency_value}")

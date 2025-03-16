import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import transforms
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from diffusers import DDIMScheduler
from diffusers import StableDiffusionImg2ImgPipeline
import matplotlib.pyplot as plt
from PIL import Image
import os
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import os
import torch.nn.functional as F
import cv2
from Interpolation import Interpolation, LinearInterpolation, VAEInterpolation
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model for text encoding
# clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
# clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# def get_text_embedding(text):
#     inputs = clip_processor(text=text, return_tensors="pt").to(device)
#     with torch.no_grad():
#         text_features = clip_model.get_text_features(**inputs)
#     return text_features

# Load Stable Diffusion pipeline

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16,
    safety_checker=None  # Disables NSFW filter
).to("cuda")
# Set the scheduler
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.scheduler.set_timesteps(50)  # Reduce noise for smoother results
# Enable attention slicing with a smaller slice size
pipe.enable_attention_slicing(slice_size="auto")  # Reduce the slice size if needed
pipe.to(device)

def interpolate_images(image1, image2,interp_model=LinearInterpolation(), output_allframes="outputs/allframes",image_name="",num_frames=100):
    """ Interpolates between two images using Stable Diffusion and returns the generated images. """


    # Interpolate between the latent codes
    images_interpolated = []
    text_prompt="A natural image"
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        image_interpolated = interp_model.interpolate(image1, image2, alpha)
        interp_image=pipe(prompt=text_prompt, image=image_interpolated, strength=0.2, guidance_scale=5.0).images[0]
        image_path = os.path.join(output_allframes, f"frame_{image_name}_{i:03d}.png")
        interp_image.save(image_path)
        images_interpolated.append(interp_image)
    # Generate images from the interpolated latent codes
    
    return images_interpolated

def interpolate_images_iter(image1, image2,num=0,interp_model=LinearInterpolation(),depth=8):
    """ Interpolates between two images using Stable Diffusion and returns the generated images. """
    if depth ==0:
        return [image2]
    else:
        transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
    ])
        intermediate_image=interp_model.interpolate(image1,image2,0.5)
        text_prompt="A natural image"
        interp_image=pipe(prompt=text_prompt, image=intermediate_image, strength=0.05*depth, guidance_scale=5.0).images[0]
        path=os.path.join("outputs/allframes",f"indice_{num} at depth_{depth}.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        interp_image.save(path)
        interp_image=transform(interp_image).unsqueeze(0).to(device)
        return interpolate_images_iter(image1,interp_image,num,interp_model,depth-1)+interpolate_images_iter(interp_image,image2,num,interp_model,depth-1)


def preprocess_image(image_path, image_size=(256,256)):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0).to(device)  # Add batch dimension


def generate_key_frames(initial_image, text_prompt, num_frames=16, strength=0.8, guidance_scale=7.5, output_folder="outputs/keyframes", device="cuda"):
    """ Generates key frames using Stable Diffusion and saves them as images. """
    os.makedirs(output_folder, exist_ok=True)

    # Convert PIL image to tensor
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
    ])
    
    current_frame = initial_image
    for i in range(num_frames):
        # Generate the next frame
        image = pipe(prompt=text_prompt, image=current_frame, strength=strength, guidance_scale=guidance_scale).images[0]

        # Save the frame
        image_path = os.path.join(output_folder, f"frame_{i:03d}.png")
        image.save(image_path)

        # Display the frame
        # frame_np = np.array(image)
        # plt.imshow(frame_np)
        # plt.axis('off')
        # plt.show(block=False)
        # plt.pause(0.1)

        # Set the current frame to the last generated frame
        current_frame = transform(image).unsqueeze(0).to(device)

    print(f"Key frames saved in {output_folder}")
    
import torch
import numpy as np


def generate_interpolated_video(output_folder="outputs/keyframes", output_video="outputs/output.avi",step=100, device="cuda"):
    """ Loads key frames, generates interpolated frames using VAE, and creates a video. """
    frames = []
    frame_files = sorted(os.listdir(output_folder))  # Ensure correct order

    transform = transforms.Compose([
    transforms.Resize((256,256)),  # Resize to a smaller size for efficiency
    transforms.ToTensor(),
])
    cnt=0
    # Load key frames
    for i in range(len(frame_files) - 1):
        img1 = Image.open(os.path.join(output_folder, frame_files[i]))
        img2 = Image.open(os.path.join(output_folder, frame_files[i + 1]))

        # Convert to tensors
        current_frame = transform(img1).unsqueeze(0).to(device)
        next_frame = transform(img2).unsqueeze(0).to(device)

        # Interpolate between the frames
        #interpolated_frames = interpolate_images(current_frame, next_frame, image_name=f"key_{i}",num_frames=step)
        interpolated_frames=interpolate_images_iter(current_frame,next_frame,num=i)
        for frame in interpolated_frames:
            frame = transforms.ToPILImage()(frame.squeeze(0).cpu())  # Convert tensor to PIL
            frame=np.array(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frames.append(frame)  

    # Save the video
    # print(frames[0].shape)
    height, width,_= frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_video, fourcc, 10, (width, height))

    for frame in frames:
        frame=np.array(frame)
        out.write(frame)
    out.release()
    print(f"Video saved as {output_video}")
    
if __name__=="__main__":
    # Load the initial image
    image_path = "images/test.jpg"  # Change this to your image path
    text_prompt = "Sunrise and sunset at the beach."  # Change this to your text prompt
    initial_image = preprocess_image(image_path)

    generate_key_frames(initial_image, text_prompt, num_frames=3)
    generate_interpolated_video(output_folder="outputs/keyframes", output_video="outputs/output.avi", step=300)
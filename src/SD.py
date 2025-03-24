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
from VAE import VAE
from LLM import LLMClient
device = "cuda" if torch.cuda.is_available() else "cpu"

image_size=128

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16,
    safety_checker=None  # Disables NSFW filter
).to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.scheduler.set_timesteps(50)  # Reduce noise for smoother results
pipe.enable_attention_slicing(slice_size="auto")  # Reduce the slice size if needed
pipe.to(device)

def resize_image(image, size=(512, 512)):
    """Resizes a PIL Image or tensor to the given size."""
    if isinstance(image, torch.Tensor):
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToPILImage()
        ])
        return transform(image.squeeze(0).cpu())  # Convert to PIL for SD pipeline
    return image.resize(size, Image.LANCZOS)

def interpolate_images(image1, image2,interp_model=LinearInterpolation(), output_allframes="outputs/allframes",image_name="",num_frames=100):
    """ Interpolates between two images using Stable Diffusion and returns the generated images. """

    # Interpolate between the latent codes
    images_interpolated = []
    text_prompt="A natural image"
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        image_interpolated = interp_model.interpolate(image1, image2, alpha)
        image_interpolated=resize_image(image_interpolated)
        interp_image=pipe(prompt=text_prompt, image=image_interpolated, strength=0.2, guidance_scale=5.0).images[0]
        image_path = os.path.join(output_allframes, f"frame_{image_name}_{i:03d}.png")
        interp_image.save(image_path)
        images_interpolated.append(interp_image)
    # Generate images from the interpolated latent codes
    return images_interpolated

def interpolate_images_iter(image1, image2,prompts,num=0,interp_model=LinearInterpolation(),depth=8):
    """ Interpolates between two images using Stable Diffusion and returns the generated images. """
    if depth ==0:
        return [image2]
    else:
        if (depth<=4):
            interp_model=LinearInterpolation()
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    intermediate_image=interp_model.interpolate(image1,image2,0.5)
    text_prompt="A clear and natural image of "
    path=os.path.join("outputs/allframes",f"indice_{num} at depth_{depth}.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw = transforms.ToPILImage()(intermediate_image.squeeze(0).cpu())
    intermediate_image=resize_image(intermediate_image)
    interp_image=pipe(prompt=text_prompt+prompts[(int)(np.floor(len(prompts)/2))], image=intermediate_image, strength=0.5, guidance_scale=5.0).images[0]
    interp_image.save(path)
    raw.save(path.replace(".png","_raw.png"))
    interp_image=transform(interp_image).unsqueeze(0).to(device)
    return interpolate_images_iter(image1,interp_image,prompts[:(int)(np.floor((len(prompts)/2)))],num,interp_model,depth-1)+[interp_image]+interpolate_images_iter(interp_image,image2,prompts[(int)(np.floor((len(prompts)/2))):],num,interp_model,depth-1)


def preprocess_image(image_path, image_size=(image_size,image_size)):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((image_size)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0).to(device)  # Add batch dimension


def generate_key_frames(initial_image, text_prompt, num_frames=16, strength=0.8, guidance_scale=7.5, output_folder="outputs/keyframes", device="cuda"):
    """ Generates key frames using Stable Diffusion and saves them as images. """
    os.makedirs(output_folder, exist_ok=True)

    # Convert PIL image to tensor
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
    ])
    
    num_prompts = len(text_prompt)
    frames_per_prompt = num_frames // num_prompts
            
    current_frame = initial_image
    for i in range(num_prompts):
        for j in range(frames_per_prompt):
        # Generate the next frame
            current_frame = resize_image(current_frame)
            image = pipe(prompt=text_prompt[i], image=current_frame, strength=strength, guidance_scale=guidance_scale).images[0]
            image=resize_image(image,size=(image_size,image_size))
            # Save the frame
            image_path = os.path.join(output_folder, f"frame_{frames_per_prompt*i+j:03d}.png")
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


def generate_interpolated_video(prompts,interpolation=LinearInterpolation(),output_folder="outputs/keyframes", output_video="outputs/output.avi",depth=8, device="cuda"):
    """ Loads key frames, generates interpolated frames using VAE, and creates a video. """
    frames = []
    frame_files = sorted(os.listdir(output_folder))  # Ensure correct order

    transform = transforms.Compose([
    transforms.Resize((image_size,image_size)),  # Resize to a smaller size for efficiency
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
        interpolated_frames = interpolate_images_iter(current_frame,next_frame,prompts[i],num=i,interp_model=interpolation,depth=depth)
        for frame in interpolated_frames:
            frame = transforms.ToPILImage()(frame.squeeze(0).cpu())  # Convert tensor to PIL
            frame = np.array(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frames.append(frame)  

    # Save the video
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
    client = LLMClient()
    image_path = "images/test2.jpg"
    text_prompt = "Beautiful mountain landscape."

    text_prompt_list, text_subprompt_list = client.generate_or_load_subprompts(text_prompt, 32, filename_prompt="prompts/prompts.txt",filename_subprompt="prompts/subprompts.txt")
    print(len(text_prompt_list))
    print(len(text_subprompt_list))
    initial_image = preprocess_image(image_path)
    vae=VAE(latent_dim=128)
    vae.load_state_dict(torch.load("models/vae/100.pth"))
    vae.to(device)
    vae_interp=VAEInterpolation(vae)
    # Generate
    generate_key_frames(initial_image, text_prompt_list, num_frames=len(text_prompt_list))
    generate_interpolated_video(text_subprompt_list,interpolation=vae_interp,output_folder="outputs/keyframes", output_video="outputs/output.avi")

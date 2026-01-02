import os
os.environ['TORCH_HOME'] = os.path.join(os.path.dirname(__file__), '..', 'models', 'torch')

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import numpy as np
import glob

from tools import imagepaths_list_from_folder

# --- Config ---
IMAGE_SIZE = 512
STYLE_WEIGHT = 1e6
CONTENT_WEIGHT = 1
NUM_STEPS = 300  # Lower = faster but less refined
OUTPUT_DIR = 'outputs/pytorch_animation3'

# --- Setup ---
device = torch.device("mps" if torch.backends.mps.is_available() else 
                      "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# VGG19 for feature extraction
vgg = models.vgg19(weights='DEFAULT').features.to(device).eval()
for param in vgg.parameters():
    param.requires_grad_(False)

# Layers for style and content
CONTENT_LAYERS = ['conv_4']
STYLE_LAYERS = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_features(image, model):
    """Extract features from VGG layers."""
    features = {}
    x = image
    layer_num = 0
    for layer in model.children():
        x = layer(x)
        if isinstance(layer, nn.Conv2d):
            layer_num += 1
            name = f'conv_{layer_num}'
            features[name] = x
    return features

def gram_matrix(tensor):
    """Compute Gram matrix for style representation."""
    b, c, h, w = tensor.size()
    features = tensor.view(b, c, h * w)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram / (c * h * w)

# --- Image transforms ---
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225])
denormalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                    std=[1/0.229, 1/0.224, 1/0.225])

def load_image(path, size=IMAGE_SIZE):
    img = Image.open(path).convert('RGB')
    # Resize keeping aspect ratio
    ratio = size / max(img.size)
    new_size = tuple(int(dim * ratio) for dim in img.size)
    img = img.resize(new_size, Image.LANCZOS)
    tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
    return normalize(tensor)

def tensor_to_image(tensor):
    img = denormalize(tensor.squeeze(0)).clamp(0, 1)
    img = transforms.ToPILImage()(img.cpu())
    return img

def style_transfer(content_img, style_grams, num_steps=NUM_STEPS):
    """Run optimization-based style transfer."""
    # Start from content image
    target = content_img.clone().requires_grad_(True)
    optimizer = optim.Adam([target], lr=0.03)
    
    content_features = get_features(content_img, vgg)
    
    for step in range(num_steps):
        target_features = get_features(target, vgg)
        
        # Content loss
        content_loss = 0
        for layer in CONTENT_LAYERS:
            content_loss += torch.mean((target_features[layer] - content_features[layer]) ** 2)
        
        # Style loss
        style_loss = 0
        for layer in STYLE_LAYERS:
            target_gram = gram_matrix(target_features[layer])
            style_gram = style_grams[layer]
            style_loss += torch.mean((target_gram - style_gram) ** 2)
        
        total_loss = CONTENT_WEIGHT * content_loss + STYLE_WEIGHT * style_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            print(f"  Step {step}/{num_steps}, Loss: {total_loss.item():.2f}")
    
    return target.detach()

def get_style_grams(style_path):
    """Pre-compute Gram matrices for a style image."""
    style_img = load_image(style_path)
    style_features = get_features(style_img, vgg)
    return {layer: gram_matrix(style_features[layer]) for layer in STYLE_LAYERS}

def interpolate_grams(grams1, grams2, weight):
    """Interpolate between two sets of Gram matrices."""
    return {layer: (1 - weight) * grams1[layer] + weight * grams2[layer] 
            for layer in STYLE_LAYERS}

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load styles (supports jpg, png, jpeg, webp)
    style_files = imagepaths_list_from_folder("data/style2")
    content_frames = sorted(glob.glob('data/animation_paler/*.png'))
    
    print(f"Found {len(style_files)} style images")
    
    # Pre-compute style Gram matrices
    print("Pre-computing style features...")
    style_grams = [get_style_grams(s) for s in style_files]
    
    # Load animation frames
    
    print(f"Processing {len(content_frames)} frames...")
    
    for i, frame_path in enumerate(content_frames):
        print(f"\nFrame {i+1}/{len(content_frames)}: {os.path.basename(frame_path)}")
        
        # Calculate style blend for smooth transitions
        if len(style_grams) > 1:
            progress = i / max(1, len(content_frames) - 1)
            style_idx = progress * (len(style_grams) - 1)
            s1, s2 = int(np.floor(style_idx)), min(int(np.ceil(style_idx)), len(style_grams) - 1)
            weight = style_idx - s1
            blended_grams = interpolate_grams(style_grams[s1], style_grams[s2], weight)
            print(f"  Blending styles {style_files[s1]} ({1-weight:.1%}) + {style_files[s2]} ({weight:.1%})")
        else:
            blended_grams = style_grams[0]
        
        # Load and stylize
        content_img = load_image(frame_path)
        stylized = style_transfer(content_img, blended_grams)
        
        # Save
        output_path = os.path.join(OUTPUT_DIR, f"{i:04d}.png")
        tensor_to_image(stylized).save(output_path)
        print(f"  Saved: {output_path}")

if __name__ == "__main__":
    main()
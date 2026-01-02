import os
os.environ['TORCH_HOME'] = os.path.join(os.path.dirname(__file__), '..', 'models', 'torch')

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import glob

from tools import imagepaths_list_from_folder

# --- Config ---
IMAGE_SIZE = 512
OUTPUT_DIR = '../outputs/pytorch_adain'

# --- Setup Device ---
device = torch.device("mps" if torch.backends.mps.is_available() else 
                      "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Model paths ---
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'adain')
DECODER_PATH = os.path.join(MODELS_DIR, 'decoder.pth')
VGG_PATH = os.path.join(MODELS_DIR, 'vgg_normalised.pth')


# --- Network Definitions ---
class Decoder(nn.Module):
    """Mirrors the VGG encoder structure in reverse."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(512, 256, 3), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(256, 128, 3), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1), nn.Conv2d(128, 128, 3), nn.ReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(128, 64, 3), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1), nn.Conv2d(64, 64, 3), nn.ReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(64, 3, 3),
        )

    def forward(self, x):
        return self.net(x)


class VGGEncoder(nn.Module):
    """VGG19 encoder up to relu4_1."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 3, 1),  # Normalize layer
            nn.ReflectionPad2d(1), nn.Conv2d(3, 64, 3), nn.ReLU(),      # relu1_1
            nn.ReflectionPad2d(1), nn.Conv2d(64, 64, 3), nn.ReLU(),     # relu1_2
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.ReflectionPad2d(1), nn.Conv2d(64, 128, 3), nn.ReLU(),    # relu2_1
            nn.ReflectionPad2d(1), nn.Conv2d(128, 128, 3), nn.ReLU(),   # relu2_2
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.ReflectionPad2d(1), nn.Conv2d(128, 256, 3), nn.ReLU(),   # relu3_1
            nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU(),   # relu3_2
            nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU(),   # relu3_3
            nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU(),   # relu3_4
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.ReflectionPad2d(1), nn.Conv2d(256, 512, 3), nn.ReLU(),   # relu4_1
        )

    def forward(self, x):
        return self.net(x)


# --- Load models from local files ---
def load_models():
    if not os.path.exists(VGG_PATH) or not os.path.exists(DECODER_PATH):
        print("=" * 60)
        print("ERROR: AdaIN model weights not found!")
        print("Please download them manually:")
        print()
        print("1. decoder.pth:")
        print("   https://drive.google.com/file/d/1bMfhMMwPeXnYSQI6cDWElSZxOxc6aVyr/view")
        print(f"   Save to: {DECODER_PATH}")
        print()
        print("2. vgg_normalised.pth:")
        print("   https://drive.google.com/file/d/1EpkBA2K2eYILDSyPTt0fztz59UjAIpZU/view")
        print(f"   Save to: {VGG_PATH}")
        print("=" * 60)
        raise FileNotFoundError("Model weights not found. See instructions above.")
    
    print("Loading models...")
    encoder = VGGEncoder()
    encoder.net.load_state_dict(torch.load(VGG_PATH, map_location='cpu', weights_only=True))
    encoder = encoder.to(device).eval()

    decoder = Decoder()
    decoder.net.load_state_dict(torch.load(DECODER_PATH, map_location='cpu', weights_only=True))
    decoder = decoder.to(device).eval()
    print("Models loaded!")
    return encoder, decoder

encoder, decoder = load_models()


# --- AdaIN Functions ---
def calc_mean_std(feat, eps=1e-5):
    N, C = feat.size()[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized = (content_feat - content_mean) / content_std
    return normalized * style_std + style_mean


# --- Image Processing ---
def load_image(path, size=IMAGE_SIZE):
    img = Image.open(path).convert('RGB')
    ratio = size / max(img.size)
    new_size = tuple(int(dim * ratio) for dim in img.size)
    img = img.resize(new_size, Image.LANCZOS)
    return transforms.ToTensor()(img).unsqueeze(0).to(device)


def save_image(tensor, path):
    img = tensor.squeeze(0).clamp(0, 1)
    img = transforms.ToPILImage()(img.cpu())
    img.save(path)


def stylize(content_img, style_img, alpha=1.0):
    """Apply style transfer. alpha controls style strength (0-1)."""
    with torch.no_grad():
        content_feat = encoder(content_img)
        style_feat = encoder(style_img)
        
        stylized_feat = adaptive_instance_normalization(content_feat, style_feat)
        # Blend between original content features and stylized (alpha control)
        stylized_feat = alpha * stylized_feat + (1 - alpha) * content_feat
        
        return decoder(stylized_feat)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load styles
    style_files = imagepaths_list_from_folder("../data/style")
    if not style_files:
        print("No style images found!")
        return
    
    print(f"Found {len(style_files)} style images")
    
    # Pre-load style images
    style_imgs = [load_image(s) for s in style_files]
    
    # Load animation frames
    content_frames = sorted(glob.glob("../data/animation/*.png"))
    if not content_frames:
        print("No animation frames found!")
        return
    
    print(f"Processing {len(content_frames)} frames...")
    
    for i, frame_path in enumerate(content_frames):
        # Calculate style blend for smooth transitions
        if len(style_imgs) > 1:
            progress = i / max(1, len(content_frames) - 1)
            style_idx = progress * (len(style_imgs) - 1)
            s1 = int(np.floor(style_idx))
            s2 = min(int(np.ceil(style_idx)), len(style_imgs) - 1)
            weight = style_idx - s1
            
            # Interpolate style images
            blended_style = (1 - weight) * style_imgs[s1] + weight * style_imgs[s2]
        else:
            blended_style = style_imgs[0]
        
        # Load content and stylize
        content_img = load_image(frame_path)
        stylized = stylize(content_img, blended_style)
        
        # Save
        output_path = os.path.join(OUTPUT_DIR, f"{i:04d}.png")
        save_image(stylized, output_path)
        
        if i % 10 == 0:
            print(f"  Frame {i+1}/{len(content_frames)}")
    
    print(f"Done! Output saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
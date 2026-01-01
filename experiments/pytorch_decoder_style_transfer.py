import os
os.environ['TORCH_HOME'] = os.path.join(os.path.dirname(__file__), '..', 'models', 'torch')

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import glob

# --- Setup Device & Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Using a standard AdaIN-style encoder (VGG19)
# In a real setup, you'd load a pre-trained 'Decoder' as well.
vgg = models.vgg19(weights='DEFAULT').features.to(device).eval()

def calc_mean_std(feat, eps=1e-5):
    # This captures the 'essence' of the style (color/texture stats)
    size = feat.size()
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, style_feat):
    # This is the "Magic" that transfers style features to content
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean) / content_std
    return normalized_feat * style_std + style_mean

# --- Processing Logic ---
prep = transforms.Compose([
    transforms.Resize(512),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_style_features(path):
    img = prep(Image.open(path)).unsqueeze(0).to(device)
    with torch.no_grad():
        return vgg(img)

def main():
    # Load your folder of styles
    style_files = sorted(glob.glob('../data/style/*.png'))
    style_feats = [get_style_features(s) for s in style_files]
    content_frames = sorted(glob.glob('../data/animation/*.png'))

    for i, frame_path in enumerate(content_frames):
        # 1. Determine which two styles to blend
        progress = i / (len(content_frames) - 1)
        style_idx = progress * (len(style_feats) - 1)
        s1, s2 = int(np.floor(style_idx)), int(np.ceil(style_idx))
        weight = style_idx - s1
        
        # 2. Interpolate the style features (The smooth transition)
        # This prevents the "flicker" by evolving the style stats gradually
        mixed_style = (1 - weight) * style_feats[s1] + weight * style_feats[s2]
        
        # 3. Load content and apply style
        content_img = prep(Image.open(frame_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            content_feat = vgg(content_img)
            # Apply the blended style to the content features
            stylized_feat = adaptive_instance_normalization(content_feat, mixed_style)
            
            # Note: In a full script, you would pass 'stylized_feat' through 
            # a trained Decoder network to turn it back into an image.

if __name__ == "__main__":
    main()
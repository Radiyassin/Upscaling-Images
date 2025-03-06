import torch
from realesrgan import RealESRGANer
import numpy as np
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet

# Initialize device (GPU/CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load model
model_path = 'RealESRGAN_x4plus.pth'
state_dict = torch.load(model_path, map_location=device)['params_ema']

model = RRDBNet(
    num_in_ch=3, num_out_ch=3, num_feat=64, 
    num_block=23, num_grow_ch=32, scale=4
)
model.load_state_dict(state_dict, strict=True)
model.to(device)

# Initialize upsampler with tiling
upsampler = RealESRGANer(
    scale=4,
    model_path=model_path,
    model=model,
    tile=400,        # Split image into tiles for processing
    tile_pad=10,      # Padding for tiles
    pre_pad=0,
    half=True if torch.cuda.is_available() else False  # Disable FP16 on CPU
)

# Load and preprocess image
img = Image.open('input.jpg').convert('RGB')
img = np.array(img)

# Upscale with outscale=1 (4x total)
print("Starting upscaling...")
output, _ = upsampler.enhance(img, outscale=1)  # Use 1 for model's native 4x

# Save output
output_img = Image.fromarray(output)
output_img.save('output.png')
print("Done!")
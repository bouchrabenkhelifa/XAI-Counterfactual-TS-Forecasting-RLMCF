import torch
import sys

checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "assets/checkpoints/weather_chpts/forecaster/chpt_weather_96_48_patchtst_S.pth"

ckpt = torch.load(checkpoint_path, map_location='cpu')

print(f"\nCheckpoint: {checkpoint_path}")
print("="*60)

# Get key dimensions
for key in sorted(ckpt.keys()):
    if 'weight' in key or 'bias' in key:
        print(f"{key}: {ckpt[key].shape}")

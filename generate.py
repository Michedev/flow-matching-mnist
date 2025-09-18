from .models.flow_matching_model import ImageFlowMatcher
import torch
import argparse
from torchvision.utils import save_image
import os

def get_args():
    parser = argparse.ArgumentParser()
    
    # Model args
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    
    # Generation args
    parser.add_argument('--num_samples', type=int, default=16,
                        help='Number of images to generate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for generation')
    parser.add_argument('--output_dir', type=str, default='generated',
                        help='Directory to save generated images')
    parser.add_argument('--num_steps', type=int, default=2,
                        help='Number of steps for generation')
    parser.add_argument('--channels', type=int, default=1,
                        help='Number of channels in generated images')
    parser.add_argument('--height', type=int, default=28,
                        help='Height of generated images')
    parser.add_argument('--width', type=int, default=28,
                        help='Width of generated images')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()

def main():
    args = get_args()
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImageFlowMatcher()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    # Generate images
    for i in range(0, args.num_samples, args.batch_size):
        batch_size = min(args.batch_size, args.num_samples - i)
        
        # Generate samples
        samples = model.generate(
            batch_size=batch_size,
            sample_image_size=(args.channels, args.height, args.width),
            num_steps=args.num_steps
        )
        
        # Save individual images
        for j, sample in enumerate(samples):
            save_image(sample, os.path.join(args.output_dir, f'sample_{i+j}.png'))
                
    print(f"Generated {args.num_samples} images in {args.output_dir}")

if __name__ == '__main__':
    main()
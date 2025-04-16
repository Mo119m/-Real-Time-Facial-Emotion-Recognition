import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from pathlib import Path

# ------------------- Custom Transforms -------------------
class AddBackgroundNoise:
    def __init__(self, noise_std=0.1, p=0.5):
        self.noise_std = noise_std
        self.p = p

    def __call__(self, img):
        if torch.rand(1) > self.p:
            return img
        noise = torch.randn_like(img) * self.noise_std
        return torch.clamp(img + noise, 0.0, 1.0)

class ColorDistortion:
    def __init__(self, min_factor=0.5, max_factor=1.5, p=0.5):
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.p = p

    def __call__(self, img):
        if torch.rand(1) > self.p:
            return img
        factors = torch.empty(3).uniform_(self.min_factor, self.max_factor)
        distorted_img = img * factors[:, None, None]
        return torch.clamp(distorted_img, 0.0, 1.0)

class LightingOcclusion:
    def __init__(self, max_alpha=0.7, min_size=0.25, p=0.5):
        self.max_alpha = max_alpha
        self.min_size = min_size
        self.p = p

    def __call__(self, img):
        if torch.rand(1) > self.p:
            return img
        
        C, H, W = img.shape
        device = img.device
        
        # Random occlusion parameters
        alpha = torch.empty(1).uniform_(0.3, self.max_alpha).item()
        w = int(torch.empty(1).uniform_(self.min_size, 0.5).item() * W)
        h = int(torch.empty(1).uniform_(self.min_size, 0.5).item() * H)
        x0 = torch.randint(0, W - w, (1,)).item()
        y0 = torch.randint(0, H - h, (1,)).item()

        # Create elliptical occlusion mask
        mask = torch.ones((1, H, W), device=device)
        y = torch.arange(H, device=device).float()
        x = torch.arange(W, device=device).float()
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        cx, cy = x0 + w/2, y0 + h/2
        rx, ry = w/2 * 1.2, h/2 * 1.2
        ellipse_mask = ((xx - cx)/rx)**2 + ((yy - cy)/ry)**2 <= 1
        mask[0, ellipse_mask] = 1 - alpha
        
        return torch.clamp(img * mask, 0.0, 1.0)
    


def collate_fn_skip_errors(batch):
    """Handle corrupted images and empty batches"""
    # Filter out None values
    batch = [item for item in batch if item is not None]
    
    # Handle case where all items in batch are corrupted
    if len(batch) == 0:
        # Return empty tensor with expected shape
        return torch.empty(0, 3, 224, 224)  # Adjust size as needed
    
    return torch.utils.data.default_collate(batch)


# ------------------- Dataset Class -------------------
class AugmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        
        if not os.path.isdir(root_dir):
            raise ValueError(f"Directory {root_dir} does not exist")

        # Find all image files recursively
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        for dirpath, _, filenames in os.walk(root_dir):
            for fname in filenames:
                if fname.lower().endswith(valid_extensions) and not fname.startswith('.'):
                    full_path = os.path.join(dirpath, fname)
                    if os.path.isfile(full_path):
                        self.image_paths.append(full_path)

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No valid images found in {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error loading {img_path}: {str(e)}")
            return None

# ------------------- Full Pipeline -------------------
def create_data_pipeline(data_root, batch_size=32):
    augmentation = transforms.Compose([
        transforms.Resize((224, 224)),  # Add fixed size for empty batch handling
        transforms.ToTensor(),
        AddBackgroundNoise(noise_std=0.1, p=0.3),
        ColorDistortion(min_factor=0.5, max_factor=1.5, p=0.3),
        LightingOcclusion(max_alpha=0.7, min_size=0.25, p=0.3),
    ])

    try:
        dataset = AugmentationDataset(data_root, transform=augmentation)
        print(f"Loaded {len(dataset)} images")
    except Exception as e:
        print(f"Error: {str(e)}")
        return None, None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn_skip_errors
    )
    
    return dataset, dataloader


def visualize_augmentations(dataloader, num_images=5, save_dir="./augmented_samples"):
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Get a batch of images
    batch = next(iter(dataloader))
    
    # Convert tensor to numpy array and denormalize
    images = batch.cpu().numpy()
    
    # Plot configuration
    plt.figure(figsize=(15, 5))
    plt.title("Augmented Samples")
    
    # Display and save images
    for i in range(min(num_images, len(images))):
        # Convert CHW to HWC
        img = np.transpose(images[i], (1, 2, 0))
        
        # Create subplot
        plt.subplot(1, num_images, i+1)
        plt.imshow(img)
        plt.axis('off')
        
        # Save individual images
        img_path = save_path / f"augmented_{i}.png"
        vutils.save_image(torch.tensor(images[i]), str(img_path))
    
    # Save grid image
    grid = vutils.make_grid(batch[:num_images], nrow=num_images, padding=2)
    vutils.save_image(grid, save_path / "augmentation_grid.png")
    
    plt.tight_layout()
    plt.show()

def save_augmented_dataset(dataloader, output_dir="./augmented_dataset"):
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save all augmented images
    for batch_idx, batch in enumerate(dataloader):
        for img_idx, image in enumerate(batch):
            # Generate unique filename
            filename = output_path / f"img_{batch_idx}_{img_idx}.png"
            
            # Save image
            vutils.save_image(image, str(filename))
            
        print(f"Saved batch {batch_idx} with {len(batch)} images")
        
    print(f"Finished saving augmented dataset to {output_path}")

# Modified main section
if __name__ == "__main__":
    # Create data pipeline
    dataset, dataloader = create_data_pipeline("/Users/aj/Desktop/STAT453/Project/data", batch_size=8)
    
    if dataloader is not None:
        # Visualize first batch
        visualize_augmentations(dataloader, num_images=5, save_dir="./samples")
        
        # Save entire augmented dataset
        save_augmented_dataset(dataloader, output_dir="./augmented_data")
        
        # Interactive visualization
        batch = next(iter(dataloader))
        print("Batch shape:", batch.shape)
        
        # Show single image details
        sample_img = batch[0].numpy().transpose(1, 2, 0)
        plt.figure(figsize=(8, 8))
        plt.imshow(sample_img)
        plt.title("Sample Augmented Image")
        plt.axis('off')
        plt.show()


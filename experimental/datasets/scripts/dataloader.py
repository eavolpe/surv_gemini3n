from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

# Config
DATASET_ROOT = "/Users/rafael/Factored/surv_gemini3n/experimental/datasets/data/anomaly_frames"
IMG_SIZE = 256
BATCH_SIZE = 8

# Define transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

def main():
    dataset = ImageFolder(root=DATASET_ROOT, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    print("Class-to-index mapping:", dataset.class_to_idx)

    for images, labels in loader:
        print("Batch shape:", images.shape)
        print("Labels:", labels)
        break

if __name__ == '__main__':
    main()
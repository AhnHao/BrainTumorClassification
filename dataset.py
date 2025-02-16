import os
import numpy as np
import matplotlib.pyplot as plt
from analysis import plot_pie_chart
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image

class BrainTumorDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.image_paths = []
        self.labels = []

        for label, cls in enumerate(self.classes):
            cls_folder = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_folder):
                self.image_paths.append(os.path.join(cls_folder, img_name))
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Tạo transform
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Thay đổi đường dẫn dataset tương ứng
train_dataset = BrainTumorDataset(root_dir="dataset/Training", transform=train_transform)
test_dataset = BrainTumorDataset(root_dir="dataset/Testing", transform=test_transform)

# Chia tập test thành tập validation và tập test
val_dataset, test_dataset = train_test_split(test_dataset,test_size=0.5, random_state=42)

# Tạo DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Hiển thị tỷ lệ giữa 3 dataset
total_train = len(train_dataset)
total_val = len(val_dataset)
total_test = len(test_dataset)
plot_pie_chart(total_train, total_test, total_val)

# Hiển thị một số hình ảnh sau khi đã qua transform
def show_random_images_from_loader(data_loader, class_names, num_images=16):
    images, labels = next(iter(data_loader))

    images = images.numpy().transpose((0, 2, 3, 1))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    images = std * images + mean
    images = np.clip(images, 0, 1)

    plt.figure(figsize=(12, 6))
    for i in range(num_images):
        plt.subplot(4, 4, i + 1)
        plt.imshow(images[i])
        plt.title(f"{class_names[labels[i].item()]}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join("images", "images_tranform.png"))
    plt.show()

class_names = os.listdir("dataset/Training")
show_random_images_from_loader(train_loader, class_names)
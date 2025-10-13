import os
from PIL import Image
from torchvision import transforms

def load_images_path(path):
    images_path = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".png"):
                images_path.append(os.path.join(root, file))
    return images_path

def load_transform_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    image = Image.open(image_path).convert('L')
    return transform(image)

def load_images_dataset():
    categories = ["40_Percent", "70_Percent", "80_Percent", "100_Percent"]
    labels = {"40_Percent": 0, "70_Percent": 1, "80_Percent": 2, "100_Percent": 3}
    dataset = []
    for category in categories:
        path = f"data/{category}"
        for image_path in load_images_path(path):
            image_tensor = load_transform_image(image_path)
            dataset.append((image_tensor, labels[category]))
    return dataset

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from data_loader import load_data
from gait_dataset import GaitDataset
from cnn_model import CNN
from train_evaluate import train_and_evaluate, plot_roc_curve
from shap_explain import explain_with_shap

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256
EPOCHS = 5
CSV_PATH = 'ref.csv'
DIRECTORY_PATH = 'TreadmillDatasetD'
image_paths, labels = load_data(CSV_PATH, DIRECTORY_PATH)
trainval_paths, test_paths, trainval_labels, test_labels = train_test_split(image_paths, labels, test_size=0.1, random_state=42)
train_paths, val_paths, train_labels, val_labels = train_test_split(trainval_paths, trainval_labels, test_size=0.2, random_state=42)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
train_dataset = GaitDataset(train_paths, train_labels, transform=transform)
val_dataset = GaitDataset(val_paths, val_labels, transform=transform)
test_dataset = GaitDataset(test_paths, test_labels, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
model = CNN().to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_and_evaluate(model, train_loader, val_loader, test_loader, DEVICE, criterion, optimizer, EPOCHS)
explain_with_shap(model, test_loader, DEVICE)

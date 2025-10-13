import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from copy import deepcopy
from cnn_model import CNN
from torchvision import transforms

class AlgaePPO:
    def __init__(self, device, num_models, policy_class, input_shape, num_actions):
        self.device = device
        self.policies = [policy_class(input_shape, num_actions).to(device) for _ in range(num_models)]
        self.optimizers = [torch.optim.Adam(policy.parameters()) for policy in self.policies]
        self.scheduler = [lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) for optimizer in self.optimizers]
        self.loss_fn = nn.CrossEntropyLoss()

    def train(self, dataset, episodes):
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        accuracy_score = []
        episode_list = []
        for episode in range(episodes):
            for model_idx, policy in enumerate(self.policies):
                optimizer = self.optimizers[model_idx]
                total_loss = 0
                total = 0
                correct = 0
                for inputs, labels in loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = policy(inputs)
                    loss = self.loss_fn(outputs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                accuracy = 100 * correct / total
                accuracy_score.append(accuracy)
                episode_list.append(episode)
                print(f"Model {model_idx + 1} Episode {episode + 1} Loss: {total_loss} Accuracy: {accuracy}")

            if episode % 5 == 0:
                self.evolve()
        return episode_list, accuracy_score

    def evolve(self):
        parent1, parent2 = np.random.choice(self.policies, 2, replace=False)
        child_weights = {}
        for (name1, param1), (name2, param2) in zip(parent1.named_parameters(), parent2.named_parameters()):
            if np.random.rand() > 0.5:
                child_weights[name1] = param1.data.clone()
            else:
                child_weights[name1] = param2.data.clone()
        for name in child_weights:
            if np.random.rand() < 0.1:
                mutation = torch.randn_like(child_weights[name]) * 0.1
                child_weights[name] += mutation
        worst_model_idx = np.random.randint(len(self.policies))
        for name, param in self.policies[worst_model_idx].named_parameters():
            param.data.copy_(child_weights[name])
        print("Population evolved")

    def evaluate(self, test_loader):
        model_accuracies = []
        with torch.no_grad():
            for idx, model in enumerate(self.policies):
                correct = 0
                total = 0
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
                accuracy = 100 * correct / total
                model_accuracies.append(accuracy)
                print(f"Model {idx + 1} Accuracy on Test Set: {accuracy:.2f}%")
        return model_accuracies

    def predict(self, data_loader):
        model = self.model
        model.eval()
        predictions = []
        with torch.no_grad():
            for data in data_loader:
                processed_data = self._preprocess_data(data)
                output = model(processed_data.to(self.device))
                predicted = torch.max(output, 1)[1]
                predictions.extend(predicted.tolist())
        return predictions

    def _preprocess_data(self, batch):
        images, _ = batch
        return images

    def _preprocess_data(self, data):
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        return transform(data).unsqueeze(0)

    def save_model(self, path="best_model.pth"):
        self.policies.sort(key=lambda x: self.evaluate(x), reverse=True)
        model = self.policies[0]
        torch.save(model.state_dict(), path)
        print("Model saved successfully")

    def load_model(self, path="best_model.pth"):
        model = CNN((1, 128, 128), 4)
        model.load_state_dict(torch.load(path))
        print("Model loaded successfully")

import torch
from algae_ppo import AlgaePPO
from dataset import CustomDataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from cnn_model import CNN 

device = torch.device("cuda")
trainer = AlgaePPO(device=device, num_models=3, policy_class=CNN, input_shape=(1, 128, 128), num_actions=4)
train_dataset = CustomDataset()
episode_list, accuracy_scores = trainer.train(train_dataset, episodes=50)
plt.plot(episode_list, accuracy_scores)
plt.xlabel("Episodes")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Episodes")
plt.show()
trainer.save_model("model_classification.pth")
dataset = CustomDataset()
data = DataLoader(dataset, batch_size=1, shuffle=True)
trainer.evaluate(data)

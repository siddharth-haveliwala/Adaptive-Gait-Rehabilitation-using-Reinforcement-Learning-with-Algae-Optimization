import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        dummy_input = torch.zeros(1, *input_shape)
        output_size = self.conv_layers(dummy_input).nelement()
        self.fc_input_size = output_size // dummy_input.size(0)
        self.fc_layers = nn.Sequential(
            nn.Linear(self.fc_input_size, 256), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(256, num_actions)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# Third-party Imports
import torch.nn as nn

# Constants
INPUT_SIZE = 784
HIDDEN_SIZE = 128
NUM_CLASSES = 10

class MLP(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_classes=NUM_CLASSES):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First fully connected layer
        self.relu = nn.ReLU()  # Activation function
        self.fc2 = nn.Linear(hidden_size, num_classes)  # Second fully connected layer

    def forward(self, x):
        out = self.fc1(x)  # Pass input through the first layer
        out = self.relu(out)  # Apply activation function
        out = self.fc2(out)  # Pass through the second layer
        return out

# Instantiate the model
model = MLP()
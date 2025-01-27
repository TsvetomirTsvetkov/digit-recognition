# Third-party Imports
import torch.nn as nn

# Constants
INPUT_SIZE = 784 
NUM_CLASSES = 10


class LogisticRegression(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, num_classes=NUM_CLASSES):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.linear(x)

model = LogisticRegression()

import torch.nn as nn

INPUT_SIZE = 784
HIDDEN_SIZE = 128
NUM_CLASSES = 10

class RNN(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=NUM_CLASSES, num_layers=1):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x, _ = self.rnn(x)
        x = self.fc(x[:, -1, :])
        return x

# Instantiate the model
model = RNN()
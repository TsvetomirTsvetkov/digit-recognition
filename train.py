# Third-party Imports
import kagglehub
import torch
import torch.nn as nn
import torch.optim as optim

# Project Imports
from utils.dataloader import DataLoader

# Constants
EPOCHS = 100
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer():
    def __init__(self, model, name, epochs=EPOCHS, requires_flattening=True, learning_rate=LEARNING_RATE):
        # Set inputs
        self.model = model
        self.name = name
        self.epochs = epochs
        self.requires_flattening = requires_flattening
        self.learning_rate = learning_rate

        # Move the model to the correct device
        self.model = self.model.to(DEVICE)

        # Download latest dataset version
        path = kagglehub.dataset_download("hojjatk/mnist-dataset")

        # Create DataLoader object
        data = DataLoader(path)

        # Load Data
        (x_train, y_train), (x_test, y_test) = data.load_data()

        # Create Torch train loader
        self.train_loader = torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(
                torch.tensor(x_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.long)),
            batch_size=64,
            shuffle=True,
            num_workers=4
        )

        # Create Torch test loader
        self.test_loader = torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(
                torch.tensor(x_test, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.long)),
            batch_size=64,
            shuffle=False,
            num_workers=4
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        for epoch in range(self.epochs):
            print('Start Training...')

            # Set model to training mode
            self.model.train()

            for batch_idx, (images, labels) in enumerate(self.train_loader):
                # Move the data to the same device
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                # Flatten the images if needed
                if self.requires_flattening:
                    images = images.view(images.size(0), -1)  

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Show data to user
                if (batch_idx + 1) % 100 == 0:
                    print(f'Epoch [{epoch+1}/{self.epochs}], Step [{batch_idx+1}/{len(self.train_loader)}], Loss: {loss.item():.4f}')

        model_file = f'model_data/{self.name}.pth'

        print(f"Model will be saved in {model_file}")
        torch.save(self.model.state_dict(), model_file)

    def evaluate(self):
        # Initialize Counters
        correct = 0
        total = 0

        # Set model to evaluation mode
        self.model.eval()

        # Disable gradient computation
        with torch.no_grad():
            for images, labels in self.test_loader:
                # Move the data to the same device
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                # Flatten the images for LR
                if self.requires_flattening:
                    images = images.view(images.size(0), -1)

                # Forward pass
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)

                # Update counters
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calculate accuracy
        accuracy = 100 * correct / total

        # Show accuracy up to 2 digits after the point
        print(f"Accuracy: {accuracy:.2f} %")

        return accuracy

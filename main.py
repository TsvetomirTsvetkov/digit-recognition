# Third-party Imports
import kagglehub

# Project Imports
from dataloader import DataLoader


if __name__ == '__main__':
    # Download latest version
    path = kagglehub.dataset_download("hojjatk/mnist-dataset")

    # Create DataLoader object
    data = DataLoader(path)

    # Load Data
    (x_train, y_train), (x_test, y_test) = data.load_data()

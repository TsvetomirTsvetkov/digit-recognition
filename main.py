'''
# Project Imports
from models.logistic_regression import model as lr_model
from train import Trainer

# Constants
LR_FILENAME = 'logistic_regression'


if __name__ == '__main__':
    # Create trainer
    trainer = Trainer(model=lr_model, name=LR_FILENAME)

    # Train the model
    trainer.train()

    # Evaluate the model
    trainer.evaluate()

'''


from models.mlp import MLP  # Import the new MLP model
from train import Trainer

# Constants
MLP_FILENAME = 'mlp'  # Define a new filename for the MLP model

if __name__ == '__main__':
    # Instantiate the new MLP model
    mlp_model = MLP()

    # Create trainer with the new model
    trainer = Trainer(model=mlp_model, name=MLP_FILENAME)

    # Train the model
    trainer.train()

    # Evaluate the model
    trainer.evaluate()
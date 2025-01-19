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

    # Evaluate he model
    trainer.evaluate()

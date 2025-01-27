# 

from models.logistic_regression import model as lr_model
from models.mlp import model as mlp_model
from models.rnn_model import model as rnn_model
from train import Trainer
from gradio_ui.gradio_fe import create_interface


# Constants
MODEL_FILENAMES = {
    'lr': 'lr_data',
    'mlp': 'mlp_data',
    'rnn': 'rnn_data'
}

def train_and_evaluate(model, model_name):
    trainer = Trainer(model=model, name=model_name)
    trainer.train()
    trainer.evaluate()


def main():
    create_interface()

    # # Logistic Regression
    # train_and_evaluate(lr_model, MODEL_FILENAMES['lr'])

    # # MLP Model
    # train_and_evaluate(mlp_model, MODEL_FILENAMES['mlp'])

    # # RNN Model
    # train_and_evaluate(rnn_model, MODEL_FILENAMES['rnn'])

if __name__ == "__main__":
    main()
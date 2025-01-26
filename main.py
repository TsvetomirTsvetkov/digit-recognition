from models.logistic_regression import model as lr_model
from models.mlp import MLP
from models.rnn_model import RNN
from train import Trainer

# Constants
MODEL_FILENAMES = {
    'logistic_regression': 'logistic_regression',
    'mlp': 'mlp',
    'rnn': 'rnn_data'
}

def train_and_evaluate(model, model_name):
    trainer = Trainer(model=model, name=model_name)
    trainer.train()
    trainer.evaluate()

def main():
    pass
    # Logistic Regression
    # train_and_evaluate(lr_model, MODEL_FILENAMES['logistic_regression'])

    # # MLP Model
    # mlp_model = MLP()
    # train_and_evaluate(mlp_model, MODEL_FILENAMES['mlp'])

    # # RNN Model
    # rnn_model = RNN()
    # train_and_evaluate(rnn_model, MODEL_FILENAMES['rnn'])

if __name__ == "__main__":
    main()
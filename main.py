import gradio as gr

from models.logistic_regression import model as lr_model
from models.mlp import MLP
from models.rnn_model import RNN
from train import Trainer
from models.gradio_fe import generate_and_predict




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

def start_fe():
    interface = gr.Interface(fn=generate_and_predict, 
                         inputs=gr.inputs.Slider(minimum=0, maximum=9, step=1, default=0, label="Input Digit (0-9)"), 
                         outputs=gr.outputs.Textbox(label="Predictions"),
                         live=True)

    interface.launch()


def main():
    start_fe()
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
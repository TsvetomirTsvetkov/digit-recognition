
import sys
import os
import torch
import torch.nn as nn
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from models.rnn_model import RNN
from models.mlp import MLP
from models.logistic_regression import model as LR
# Assuming models are saved as .pt files

# Add the root folder to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set models to evaluation mode (important for PyTorch)
rnn_model = RNN()
mlp_model = MLP()
lr_model= LR()

rnn_state = torch.load("model_data/rnn_data.pth", map_location=torch.device('cpu'))
mlp_state = torch.load("model_data/mlp_data.pth", map_location=torch.device('cpu'))
lr_state = torch.load("model_data/lr_data.pth", map_location=torch.device('cpu'))

rnn_model.load_state_dict(rnn_state)
mlp_model.load_state_dict(mlp_model)
lr_state.load_state_dict(lr_state)

rnn_model.eval()
mlp_model.eval()
lr_model.eval()

def generate_digit_image(digit: int):
    # Here you would load or generate an image for the digit
    # For simplicity, assume we're using a matplotlib plot to display a random image
    fig, ax = plt.subplots()
    ax.imshow(np.random.random((28, 28)), cmap='gray')  # Random noise as placeholder
    ax.set_title(f"Generated Image: {digit}")
    plt.axis('off')
    plt.show()

def preprocess_image(image: np.ndarray):
    image = image.astype(np.float32) / 255  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension (for models expecting batches)
    image = np.expand_dims(image, axis=1)  # Add channel dimension (for grayscale images)
    return torch.tensor(image)

def predict_with_rnn(image):
    image = preprocess_image(image)
    with torch.no_grad():
        output = rnn_model(image)
        prediction = output.argmax(dim=1).item()
    return prediction

def predict_with_mlp(image):
    image = preprocess_image(image)
    with torch.no_grad():
        output = mlp_model(image)
        prediction = output.argmax(dim=1).item()
    return prediction

def predict_with_lr(image):
    # Assuming the Logistic Regression model is trained with flattened images
    image = image.flatten().reshape(1, -1)  # Flatten the image
    prediction = lr_model.predict(image)
    return prediction[0]


def generate_and_predict(digit: int):
    # Generate a random image for the digit
    generate_digit_image(digit)

    # Simulate a random image for prediction (this should be replaced with actual image generation logic)
    random_image = np.random.random((28, 28))  # Replace with your actual image generation logic
    
    # Get predictions from the models
    rnn_pred = predict_with_rnn(random_image)
    mlp_pred = predict_with_mlp(random_image)
    lr_pred = predict_with_lr(random_image)
    
    return f"RNN Prediction: {rnn_pred}\nMLP Prediction: {mlp_pred}\nLogistic Regression Prediction: {lr_pred}"

# Create the Gradio interface
interface = gr.Interface(fn=generate_and_predict, 
                         inputs=gr.inputs.Slider(minimum=0, maximum=9, step=1, default=0, label="Input Digit (0-9)"), 
                         outputs=gr.outputs.Textbox(label="Predictions"),
                         live=True)

interface.launch()
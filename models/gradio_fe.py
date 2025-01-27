
import sys
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
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


rnn_state = torch.load("model_data/rnn_data.pth", map_location=torch.device('cpu'))
mlp_state = torch.load("model_data/mlp_data.pth", map_location=torch.device('cpu'))


rnn_model.load_state_dict(rnn_state)
mlp_model.load_state_dict(mlp_state)

rnn_model.eval()
mlp_model.eval()


# Preprcess image for RNN
def preprocess_image_rnn(image):
    if isinstance(image, torch.Tensor):
        image = transforms.ToPILImage()(image)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    image_tensor = transform(image)

    print(image_tensor.dim())
    if image_tensor.dim() == 4:
        image_tensor = image_tensor.squeeze(0)
    image_tensor = image_tensor.view(1, 28, 28)
    return image_tensor


def predict_with_rnn(image):
    image = preprocess_image_rnn(image)
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


def load_mnist_dataset():
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    return dataset

def get_random_mnist_image(index):
    dataset = load_mnist_dataset()
    img, label = dataset[index]
    # img = img.squeeze(0)  # Remove the channel dimension if it's 1
    img = transforms.ToPILImage()(img)
    return img, label


def predict_image(index, model_choice):
    img, true_label = get_random_mnist_image(index)
    img_tensor = preprocess_image_rnn(img).unsqueeze(0)
    
    prediction = None
    
    try:
        print(model_choice)
        if model_choice == "RNN":
            prediction = predict_with_rnn(img_tensor)
        elif model_choice == "MLP":
            prediction = predict_with_mlp(img_tensor)
        elif model_choice == "Logistic Regression":
            prediction = predict_with_lr(img_tensor)
        else:
            prediction = "Invalid model choice."
    except Exception as e:
        prediction = f"Error during prediction: {str(e)}"
    
    return img, f"Prediction: {prediction}, Actual: {true_label}"


def create_interface():
    gr.Interface(
        fn=predict_image,
        inputs=[
            gr.Slider(minimum=0, maximum=1000, step=1, label="Select Random Index (0-1000)"),
            gr.Radio(choices=["RNN", "MLP", "Logistic Regression"], label="Choose Model")
        ],
        outputs=[
            gr.Image(label="Random MNIST Image"),
            gr.Textbox(label="Prediction vs Actual")
        ],
        title="MNIST Random Image Prediction",
        description="Select a random number (0-1000) and choose a model to predict the digit."
    ).launch()


create_interface()
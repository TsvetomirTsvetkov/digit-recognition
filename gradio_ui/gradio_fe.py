# System Imports
import sys
import os

# External Imports
import gradio as gr
import torch
import torch.nn as nn
from torchvision import datasets, transforms

# Project Imports
from models.rnn_model import model as rnn_model
from models.mlp import model as mlp_model
from models.logistic_regression import model as lr_model

# Add the root folder to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load the pth files
rnn_state = torch.load("model_data/rnn_data.pth", map_location=torch.device('cpu'))
mlp_state = torch.load("model_data/mlp_data.pth", map_location=torch.device('cpu'))
lr_state  = torch.load("model_data/lr_data.pth",  map_location=torch.device('cpu'))

rnn_model.load_state_dict(rnn_state)
mlp_model.load_state_dict(mlp_state)
lr_model.load_state_dict(lr_state)

# Set models to evaluation mode (important for PyTorch)
rnn_model.eval()
mlp_model.eval()
lr_model.eval()

# Preprcess image for RNN
def preprocess_image_rnn(image):
    if isinstance(image, torch.Tensor):
        image = transforms.ToPILImage()(image)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure single channel (grayscale)
        transforms.ToTensor(),                        # Convert to tensor [1, H, W]
        transforms.Normalize(mean=[0.5], std=[0.5])   # Normalize pixel values
    ])
    image_tensor = transform(image)  # Shape: [1, 28, 28]

    # Flatten the image for RNN
    image_tensor = image_tensor.view(1, -1)  # Shape: [1, 784]

    return image_tensor

# Predict image with RNN
def predict_with_rnn(image):
    image = preprocess_image_rnn(image)
    image = image.to(torch.float32)

    with torch.no_grad():
        output = rnn_model(image)
        prediction = output.argmax(dim=1).item()

    return prediction

# Preprcess image for MLP
def preprocess_image_mlp(image):
    if isinstance(image, torch.Tensor):
        image = transforms.ToPILImage()(image)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    image_tensor = transform(image)
    image_tensor = image_tensor.view(-1, 28 * 28)

    return image_tensor

# Predict image with MLP
def predict_with_mlp(image):
    image = preprocess_image_mlp(image)
    with torch.no_grad():
        output = mlp_model(image)
        prediction = output.argmax(dim=1).item()
    return prediction

# Preprcess image for LR
def preprocess_image_lr(image):
    if isinstance(image, torch.Tensor):
        image = transforms.ToPILImage()(image)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure single channel
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to match training
    ])
    image_tensor = transform(image)  # Shape: [1, 28, 28]

    # Flatten the image
    image_tensor = image_tensor.view(-1)  # Shape: [784]

    return image_tensor

# Predict image with LR
def predict_with_lr(image):
    image = image.flatten().reshape(1, -1)  # Flatten the image
    prediction = lr_model(image)
    _, predicted_classes = torch.max(prediction, dim=1)
    return predicted_classes[0]


def load_mnist_dataset():
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    return dataset

def get_random_mnist_image(index):
    dataset = load_mnist_dataset()
    
    img, label = dataset[index]
    img = transforms.ToPILImage()(img)

    return img, label


def predict_image(index, model_choice):
    img, true_label = get_random_mnist_image(index)

    prediction = None
    print('DEBUGGGG')
    img_tensor = preprocess_image_rnn(img)
    print(f"Preprocessed RNN image shape: {img_tensor.shape}")
    print('DEBUGGGG')
    try:
        print(model_choice)
        if model_choice == "RNN":
            img_tensor = preprocess_image_rnn(img)
            prediction = predict_with_rnn(img_tensor)
        elif model_choice == "MLP":
            img_tensor = preprocess_image_mlp(img).unsqueeze(0)
            prediction = predict_with_mlp(img_tensor)
        elif model_choice == "Logistic Regression":
            img_tensor = preprocess_image_lr(img).unsqueeze(0)
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

import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from model import SimpleCNN  # model class must match your training script

# Load model
device = torch.device("cpu")
model = SimpleCNN()
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
model.eval()

# Load MNIST data
transform = transforms.Compose([transforms.ToTensor()])
mnist = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

# Title
st.title("Handwritten Digit Generator (0–9)")
digit = st.selectbox("Select a digit to generate", list(range(10)))

# Filter MNIST dataset by digit
images = [img for img, label in mnist if label == digit]

# Randomly sample 5 images
selected_images = [images[i] for i in np.random.choice(len(images), 5, replace=False)]

# Show images
fig, axes = plt.subplots(1, 5, figsize=(10, 2))
for ax, img in zip(axes, selected_images):
    ax.imshow(img.squeeze(), cmap="gray")
    ax.axis("off")
st.pyplot(fig)

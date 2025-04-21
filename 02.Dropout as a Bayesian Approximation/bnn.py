# -*- coding: utf-8 -*-
"""
Implementation of Bayesian Dropout (MC Dropout) based on Gal & Ghahramani (2016)
on the MNIST Fashion dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration ---
CONFIG = {
    "epochs": 10,           # Number of training epochs
    "batch_size": 128,      # Batch size for training and testing
    "learning_rate": 0.001, # Optimizer learning rate
    "dropout_prob": 0.25,   # Dropout probability
    "hidden_units_1": 128,  # Number of units in the first hidden layer
    "hidden_units_2": 64,   # Number of units in the second hidden layer
    "num_classes": 10,      # Number of classes in MNIST Fashion (0-9)
    "input_dim": 28*28,     # Input dimension (flattened 28x28 image)
    "mc_samples": 50,       # Number of Monte Carlo samples for uncertainty estimation
    "seed": 42,             # Random seed for reproducibility
    "data_dir": "./data",   # Directory to save/load the dataset
    "model_save_path": "./bayesian_dropout_mnist_fashion.pth", # Path to save the model
    "device": "cuda" if torch.cuda.is_available() else "cpu", # Use GPU if available
}

# --- Reproducibility ---
torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
if CONFIG["device"] == "cuda":
    torch.cuda.manual_seed(CONFIG["seed"])
    # Might make things slower, but ensures reproducibility on GPU
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

print(f"Using device: {CONFIG['device']}")

# --- 1. Data Loading and Preprocessing ---
print("\nStep 1: Loading and Preprocessing Data...")

# Define transformations:
# ToTensor() converts PIL image or numpy array (H x W x C) in range [0, 255]
# to a torch.FloatTensor of shape (C x H x W) in range [0.0, 1.0]
# Normalize() subtracts the mean and divides by the standard deviation.
# For MNIST Fashion, mean and std are approximately 0.5 each across many datasets.
# Using (0.5,) for single channel grayscale.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download and load the training data
# Explanation:
# - datasets.FashionMNIST: PyTorch's built-in dataset handler for MNIST Fashion.
# - root=CONFIG["data_dir"]: Specifies where to download/find the data.
# - train=True: Indicates we want the training split.
# - download=True: Downloads the data if not found locally.
# - transform=transform: Applies the defined preprocessing steps.
train_dataset = datasets.FashionMNIST(
    root=CONFIG["data_dir"],
    train=True,
    download=True,
    transform=transform
)

# Download and load the test data
test_dataset = datasets.FashionMNIST(
    root=CONFIG["data_dir"],
    train=False,
    download=True,
    transform=transform
)

# Create DataLoaders
# Explanation:
# - DataLoader: Wraps an iterable around the dataset for easy batching, shuffling,
#   and parallel data loading.
# - batch_size: Number of samples per batch.
# - shuffle=True: Shuffles the training data at every epoch to prevent the model
#   from learning the order of samples and improve generalization. Test set
#   shuffling is not necessary but doesn't harm standard evaluation.
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=True
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=False # No need to shuffle test data for evaluation
)

print(f"Dataset downloaded/loaded from: {os.path.abspath(CONFIG['data_dir'])}")
print(f"Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
print(f"DataLoaders created with batch size: {CONFIG['batch_size']}")


# --- 2. Model Definition ---
print("\nStep 2: Defining the Neural Network Model...")

class BayesianMLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) with Dropout layers.
    Crucially, Dropout is applied after activation functions, which is a common practice.
    The paper's theory suggests dropout before weight layers, but applying it after
    activation before the next linear layer achieves a similar regularization effect
    and is standard in many frameworks.
    """
    def __init__(self, input_dim, hidden1, hidden2, output_dim, dropout_prob):
        super(BayesianMLP, self).__init__()
        self.input_dim = input_dim
        self.dropout_prob = dropout_prob

        # Define layers
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_dim)

        # Dropout layer - will be applied after activations
        # Explanation:
        # nn.Dropout(p=dropout_prob) randomly zeros out elements of the input tensor
        # with probability 'dropout_prob' during training.
        # According to Gal & Ghahramani, keeping this active during inference
        # allows us to sample from the approximate posterior distribution.
        self.dropout = nn.Dropout(p=self.dropout_prob)

    def forward(self, x):
        """
        Defines the forward pass of the network.
        Args:
            x (torch.Tensor): Input tensor (batch_size, input_dim)
        Returns:
            torch.Tensor: Output tensor (batch_size, output_dim) - logits
        """
        # Flatten the image (if needed, assuming input is already flat here)
        x = x.view(-1, self.input_dim) # Flatten image

        # Layer 1: Linear -> ReLU -> Dropout
        x = self.fc1(x)
        x = F.relu(x)
        # Explanation: Applying dropout AFTER activation is standard practice.
        # The dropout layer is explicitly called here.
        x = self.dropout(x)

        # Layer 2: Linear -> ReLU -> Dropout
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Output Layer: Linear (Logits)
        # No activation/dropout here, as CrossEntropyLoss applies LogSoftmax internally.
        x = self.fc3(x)
        return x

# Instantiate the model
model = BayesianMLP(
    input_dim=CONFIG["input_dim"],
    hidden1=CONFIG["hidden_units_1"],
    hidden2=CONFIG["hidden_units_2"],
    output_dim=CONFIG["num_classes"],
    dropout_prob=CONFIG["dropout_prob"]
).to(CONFIG["device"])

print("Model architecture:")
print(model)

# --- 3. Loss Function and Optimizer ---
print("\nStep 3: Defining Loss Function and Optimizer...")

# Explanation:
# - nn.CrossEntropyLoss: Combines nn.LogSoftmax and nn.NLLLoss in one class.
#   It's suitable for multi-class classification problems. It expects raw logits
#   as input and class indices as targets.
criterion = nn.CrossEntropyLoss()

# Explanation:
# - optim.Adam: An adaptive learning rate optimization algorithm that's widely
#   used and often performs well with default settings.
# - model.parameters(): Passes all trainable parameters (weights and biases)
#   of the model to the optimizer.
# - lr=CONFIG["learning_rate"]: Sets the learning rate.
optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

# --- 4. Training Loop ---
print("\nStep 4: Training the Model...")

train_losses = []
train_accuracies = []

for epoch in range(CONFIG["epochs"]):
    model.train() # Set the model to training mode (enables dropout)
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for i, (images, labels) in enumerate(train_loader):
        # Move data to the configured device (GPU or CPU)
        images = images.to(CONFIG["device"])
        labels = labels.to(CONFIG["device"])

        # --- Forward Pass ---
        # Explanation: Calculate the model's predictions (logits) for the input batch.
        outputs = model(images)

        # --- Calculate Loss ---
        # Explanation: Compare the model's predictions (outputs) with the true labels
        # using the chosen criterion (CrossEntropyLoss).
        loss = criterion(outputs, labels)

        # --- Backward Pass and Optimization ---
        # Explanation:
        # 1. optimizer.zero_grad(): Clears old gradients from the previous iteration.
        #    Crucial, otherwise gradients accumulate.
        # 2. loss.backward(): Computes the gradients of the loss with respect to
        #    all model parameters (backpropagation).
        # 3. optimizer.step(): Updates the model parameters based on the computed
        #    gradients and the optimizer's logic (e.g., Adam update rule).
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # --- Statistics ---
        running_loss += loss.item() * images.size(0) # loss.item() gives avg loss per item in batch
        _, predicted_classes = torch.max(outputs.data, 1) # Get the index of the max logit
        total_samples += labels.size(0)
        correct_predictions += (predicted_classes == labels).sum().item()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{CONFIG["epochs"]}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)
    print(f'--- End of Epoch {epoch+1} ---')
    print(f'Average Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.4f}')

print("Training finished.")

# --- 5. Standard Evaluation (Dropout OFF) ---
print("\nStep 5: Evaluating Model Performance (Standard - Dropout OFF)...")

model.eval() # Set the model to evaluation mode (disables dropout)
correct_predictions = 0
total_samples = 0

# Explanation:
# torch.no_grad(): Disables gradient calculation during inference. This reduces memory
# consumption and speeds up computation, as we don't need gradients for evaluation.
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(CONFIG["device"])
        labels = labels.to(CONFIG["device"])

        outputs = model(images)
        _, predicted_classes = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted_classes == labels).sum().item()

accuracy = correct_predictions / total_samples
print(f'Standard Test Accuracy (Dropout OFF): {accuracy:.4f}')

# --- 6. Bayesian Inference with MC Dropout (Dropout ON) ---
print("\nStep 6: Estimating Uncertainty using Monte Carlo Dropout (Dropout ON)...")

# Let's take one sample image from the test set
test_image, test_label = next(iter(test_loader))
sample_image = test_image[0].unsqueeze(0).to(CONFIG["device"]) # Take the first image, keep batch dim
sample_label = test_label[0].item()

# Explanation: Crucially, set the model to TRAIN mode here.
# This ensures that the nn.Dropout layers are ACTIVE during the forward passes,
# which is the core idea of MC Dropout for approximating Bayesian inference.
# Each forward pass will use a different dropout mask, simulating sampling
# from the approximate posterior distribution of the weights.
model.train()

mc_predictions = []
print(f"Running {CONFIG['mc_samples']} Monte Carlo samples for the first test image...")

# Explanation:
# Perform T forward passes (Monte Carlo sampling) with dropout active.
# Store the softmax probabilities for each pass.
with torch.no_grad(): # Still no gradients needed for inference itself
    for _ in range(CONFIG["mc_samples"]):
        output = model(sample_image)
        # Apply Softmax to get probabilities
        probabilities = F.softmax(output, dim=1)
        mc_predictions.append(probabilities.cpu().numpy())

# Stack the predictions into a single numpy array (T, num_classes)
mc_predictions = np.vstack(mc_predictions) # Shape: (mc_samples, num_classes)

# --- Calculate Mean Prediction and Uncertainty ---
# Explanation:
# - Predictive Mean: The average of the softmax outputs across the T samples.
#   This is the final prediction, analogous to the predictive mean in a BNN.
# - Predictive Variance/StdDev: The variance or standard deviation across the T samples
#   for EACH class. High variance for a class indicates high model uncertainty
#   about predicting that class. This represents EPISTEMIC uncertainty.
predictive_mean = np.mean(mc_predictions, axis=0)
predictive_std = np.std(mc_predictions, axis=0)
predicted_class = np.argmax(predictive_mean)

# --- Class Labels for MNIST Fashion ---
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


print(f"\n--- Results for Sample Image (True Label: {class_names[sample_label]}) ---")
print(f"Predicted Class (based on mean): {class_names[predicted_class]} (Index: {predicted_class})")
print(f"Predictive Mean Probabilities:\n{predictive_mean}")
print(f"Predictive Standard Deviations (Uncertainty):\n{predictive_std}")
print(f"Max uncertainty (std dev): {np.max(predictive_std):.4f} for class '{class_names[np.argmax(predictive_std)]}'")
print(f"Uncertainty (std dev) for predicted class '{class_names[predicted_class]}': {predictive_std[predicted_class]:.4f}")

# --- 7. Visualization ---
print("\nStep 7: Visualizing Results...")

# Plot training curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, CONFIG["epochs"] + 1), train_losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, CONFIG["epochs"] + 1), train_accuracies, label='Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training Accuracy Curve')
plt.legend()
plt.tight_layout()
plt.show()


# Visualize the sample image and the uncertainty estimate
plt.figure(figsize=(12, 6))

# Plot the sample image
plt.subplot(1, 2, 1)
# Need to un-normalize and reshape the image for display
img_display = sample_image.squeeze().cpu().numpy() * 0.5 + 0.5 # Reverse normalization
plt.imshow(img_display, cmap='gray')
plt.title(f"Sample Test Image\nTrue Label: {class_names[sample_label]}")
plt.axis('off')

# Plot predictive distribution and uncertainty
plt.subplot(1, 2, 2)
bar_positions = np.arange(CONFIG["num_classes"])
plt.bar(bar_positions, predictive_mean, yerr=predictive_std, capsize=5, alpha=0.7, label='Predictive Mean +/- Std Dev')
plt.xticks(bar_positions, class_names, rotation=45, ha='right')
plt.xlabel("Class")
plt.ylabel("Probability")
plt.title(f"MC Dropout Prediction & Uncertainty\nPredicted: {class_names[predicted_class]}")
plt.ylim(0, 1) # Probabilities are between 0 and 1
plt.legend()
plt.tight_layout()
plt.show()

# --- 8. Saving the Model ---
print("\nStep 8: Saving the Model...")

# Explanation:
# We save the model's 'state dictionary', which contains all the learned parameters
# (weights and biases). This is the recommended way to save models in PyTorch,
# as it's more flexible than saving the entire model object.
# To load it later, you would first define the model architecture again and then
# load the state dict into it.
try:
    torch.save(model.state_dict(), CONFIG["model_save_path"])
    print(f"Model state dictionary saved successfully to: {os.path.abspath(CONFIG['model_save_path'])}")
except Exception as e:
    print(f"Error saving model: {e}")

# Note on saving the dataset:
# The dataset itself is managed by torchvision and saved in the `root` directory
# specified during `datasets.FashionMNIST` initialization (CONFIG["data_dir"]).
# There's typically no need to save it separately in a different format unless
# you have specific requirements. The code assumes the data exists or can be
# downloaded to that directory.
print(f"Dataset files are stored/managed by torchvision in: {os.path.abspath(CONFIG['data_dir'])}")

print("\n--- Script Finished ---")

# --- How to Load the Saved Model (Example) ---
# print("\n--- Example: Loading the Saved Model ---")
# # 1. Re-define the model architecture EXACTLY as before
# loaded_model = BayesianMLP(
#     input_dim=CONFIG["input_dim"],
#     hidden1=CONFIG["hidden_units_1"],
#     hidden2=CONFIG["hidden_units_2"],
#     output_dim=CONFIG["num_classes"],
#     dropout_prob=CONFIG["dropout_prob"]
# )
# # 2. Load the state dictionary
# try:
#     loaded_model.load_state_dict(torch.load(CONFIG["model_save_path"]))
#     loaded_model.to(CONFIG["device"]) # Move to appropriate device
#     loaded_model.eval() # Set to eval mode for standard inference initially
#     print("Model loaded successfully.")
#     # Now 'loaded_model' can be used for inference. Remember to call
#     # loaded_model.train() before doing MC Dropout inference if needed.
# except Exception as e:
#     print(f"Error loading model: {e}")
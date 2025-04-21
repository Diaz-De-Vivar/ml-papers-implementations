# -*- coding: utf-8 -*-
"""
Implementation of Bayesian Dropout (MC Dropout) using NumPy
based on Gal & Ghahramani (2016) on the MNIST Fashion dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import torch
from torchvision import datasets # Still use torchvision for easy data access
from torchvision import transforms # For initial loading/transform
from collections import defaultdict # For saving parameters easier

# --- Configuration ---
# Keeping most settings the same as the PyTorch version for comparison
CONFIG = {
    "epochs": 10,           # Number of training epochs
    "batch_size": 128,      # Batch size for training and testing
    "learning_rate": 0.001, # Optimizer learning rate (Adam)
    "beta1": 0.9,           # Adam parameter beta1
    "beta2": 0.999,         # Adam parameter beta2
    "epsilon": 1e-8,        # Adam parameter epsilon (for numerical stability)
    "dropout_prob": 0.25,   # Dropout probability
    "hidden_units_1": 128,  # Number of units in the first hidden layer
    "hidden_units_2": 64,   # Number of units in the second hidden layer
    "num_classes": 10,      # Number of classes in MNIST Fashion (0-9)
    "input_dim": 28*28,     # Input dimension (flattened 28x28 image)
    "mc_samples": 50,       # Number of Monte Carlo samples for uncertainty estimation
    "seed": 42,             # Random seed for reproducibility
    "data_dir": "./data_numpy", # Directory to save/load the dataset
    "model_save_path": "./bayesian_dropout_numpy_mnist_fashion.npz", # Path to save model params
}

# --- Reproducibility ---
np.random.seed(CONFIG["seed"])

print(f"Using device: CPU (NumPy implementation)")

# --- 1. Data Loading and Preprocessing (using torchvision initially) ---
print("\nStep 1: Loading and Preprocessing Data...")

# Basic transform to get data as tensors, then convert to numpy
transform_to_tensor = transforms.Compose([transforms.ToTensor()])

# Download and load the training data
train_dataset_torch = datasets.FashionMNIST(
    root=CONFIG["data_dir"], train=True, download=True, transform=transform_to_tensor
)
test_dataset_torch = datasets.FashionMNIST(
    root=CONFIG["data_dir"], train=False, download=True, transform=transform_to_tensor
)

# Convert to NumPy arrays and normalize manually
def preprocess_data(dataset_torch):
    data_list = []
    label_list = []
    # Use DataLoader just for efficient iteration, batch_size=1 is fine
    loader = torch.utils.data.DataLoader(dataset_torch, batch_size=1)
    for img, label in loader:
        img_np = img.numpy().astype(np.float32) # Convert to numpy float32
        img_np = img_np.reshape(1, -1) # Flatten (1, 1, 28, 28) -> (1, 784)
        # Normalize: (pixel - mean) / std_dev. Using mean=0.5, std=0.5 simplifies
        # to (pixel - 0.5) / 0.5 = 2*pixel - 1. Input range [0, 1] -> [-1, 1]
        # Note: PyTorch Normalize((0.5,), (0.5,)) does (x-0.5)/0.5 = 2x-1
        img_np = (img_np / 255.0 - 0.5) / 0.5 # Normalize to [-1, 1] (alternative common way)
        # Or match PyTorch exactly: 2 * (img_np / 255.0) - 1
        # Let's match PyTorch's ToTensor() [0,1] and Normalize((0.5,),(0.5,)) [-1,1]
        img_np = 2.0 * (img.numpy().astype(np.float32) / 1.0) - 1.0 # ToTensor [0,1], then Normalize
        img_np = img_np.reshape(1, -1) # Flatten (1, 1, 28, 28) -> (1, 784)

        data_list.append(img_np)
        label_list.append(label.item()) # Get scalar label

    # Stack into single large arrays
    X = np.vstack(data_list)
    Y = np.array(label_list, dtype=np.int32) # Use int32 for labels
    return X, Y

X_train, Y_train = preprocess_data(train_dataset_torch)
X_test, Y_test = preprocess_data(test_dataset_torch)

# --- Save original data indices for later (optional, but good practice) ---
# This step is just to explicitly save the dataset version we used
# In a real scenario, you might save X_train, Y_train etc. directly if large
print(f"Saving dataset reference info (used torchvision download)...")
os.makedirs(os.path.dirname(CONFIG["data_dir"]), exist_ok=True)
# We don't save the numpy arrays here as they can be large,
# just noting that torchvision handles the raw data storage.
print(f"Raw data downloaded/managed by torchvision in: {os.path.abspath(CONFIG['data_dir'])}")
print(f"Data preprocessed into NumPy arrays.")
print(f"Training data shape: X={X_train.shape}, Y={Y_train.shape}")
print(f"Test data shape: X={X_test.shape}, Y={Y_test.shape}")

# --- Helper: One-Hot Encode Labels ---
def one_hot(y, num_classes):
    """ Convert integer labels y to one-hot vectors """
    y_one_hot = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    y_one_hot[np.arange(y.shape[0]), y] = 1.0
    return y_one_hot

Y_train_one_hot = one_hot(Y_train, CONFIG["num_classes"])
# Y_test_one_hot = one_hot(Y_test, CONFIG["num_classes"]) # Not strictly needed for accuracy calc

# --- 2. Model Definition (NumPy functions) ---
print("\nStep 2: Defining the Neural Network Components (NumPy)...")

def initialize_parameters(input_dim, h1, h2, output_dim):
    """ Initializes weights and biases """
    # Use He initialization scaling factor (good for ReLU)
    params = {
        'W1': np.random.randn(input_dim, h1).astype(np.float32) * np.sqrt(2. / input_dim),
        'b1': np.zeros((1, h1), dtype=np.float32),
        'W2': np.random.randn(h1, h2).astype(np.float32) * np.sqrt(2. / h1),
        'b2': np.zeros((1, h2), dtype=np.float32),
        'W3': np.random.randn(h2, output_dim).astype(np.float32) * np.sqrt(2. / h2), # Or Xavier for output
        'b3': np.zeros((1, output_dim), dtype=np.float32)
    }
    return params

def relu(Z):
    """ ReLU activation function """
    return np.maximum(0, Z)

def relu_derivative(Z):
    """ Derivative of ReLU """
    return (Z > 0).astype(np.float32)

def dropout(A, p, mode='train'):
    """
    Applies dropout (inverted dropout).
    Args:
        A: Activation input (output of previous layer or activation)
        p: Dropout probability (probability of *keeping* a unit is 1-p, but p is probability of *zeroing*)
        mode: 'train' (apply dropout and scale) or 'eval' (do nothing)
    Returns:
        A_dropout: Activation after dropout
        mask: The dropout mask used (needed for backprop)
    """
    if mode == 'eval' or p == 0.0:
        return A, np.ones_like(A, dtype=np.float32) # Return input and dummy mask

    keep_prob = 1.0 - p
    # Create mask: 1s where we keep, 0s where we drop
    mask = (np.random.rand(*A.shape) < keep_prob).astype(np.float32)
    # Apply mask and scale by keep_prob (inverted dropout)
    A_dropout = (A * mask) / keep_prob
    return A_dropout, mask

def softmax(Z):
    """ Softmax activation (numerically stable) """
    # Shift Z by max value for numerical stability (prevents overflow)
    shifted_Z = Z - np.max(Z, axis=1, keepdims=True)
    exps = np.exp(shifted_Z)
    sum_exps = np.sum(exps, axis=1, keepdims=True)
    return exps / sum_exps

def forward_pass(X, parameters, dropout_prob, mode='train'):
    """
    Performs a forward pass through the MLP.
    Args:
        X: Input data (batch_size, input_dim)
        parameters: Dictionary containing W1, b1, W2, b2, W3, b3
        dropout_prob: Dropout probability
        mode: 'train' or 'eval' or 'mc_dropout' (train and mc_dropout apply dropout)
    Returns:
        output (Z3): Logits (output before softmax)
        cache: Dictionary containing intermediate values for backprop
               (Z1, A1, D1, mask1, Z2, A2, D2, mask2, X_input)
               A = after activation, D = after dropout
    """
    W1, b1 = parameters['W1'], parameters['b1']
    W2, b2 = parameters['W2'], parameters['b2']
    W3, b3 = parameters['W3'], parameters['b3']
    apply_dropout = (mode == 'train' or mode == 'mc_dropout')

    # Layer 1: Linear -> ReLU -> Dropout
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    D1, mask1 = dropout(A1, dropout_prob, mode=mode) # Pass mode here

    # Layer 2: Linear -> ReLU -> Dropout
    Z2 = np.dot(D1, W2) + b2
    A2 = relu(Z2)
    D2, mask2 = dropout(A2, dropout_prob, mode=mode) # Pass mode here

    # Layer 3: Linear (Output logits)
    Z3 = np.dot(D2, W3) + b3

    cache = {
        'Z1': Z1, 'A1': A1, 'D1': D1, 'mask1': mask1,
        'Z2': Z2, 'A2': A2, 'D2': D2, 'mask2': mask2,
        'X': X # Store input X for dW1 calculation
    }
    return Z3, cache

# --- 3. Loss Function and Optimizer (NumPy) ---
print("\nStep 3: Defining Loss and Optimizer (NumPy)...")

def cross_entropy_loss(Y_pred_softmax, Y_true_one_hot):
    """ Calculates cross-entropy loss """
    m = Y_true_one_hot.shape[0] # Number of samples in batch
    # Add small epsilon for numerical stability (avoid log(0))
    epsilon = 1e-9
    loss = -np.sum(Y_true_one_hot * np.log(Y_pred_softmax + epsilon)) / m
    return loss

# Derivative of cross-entropy loss w.r.t Z (logits) is simply (softmax_output - y_true)
# No separate function needed, will use this directly in backprop.

def initialize_adam(parameters):
    """ Initializes momentum and RMSprop variables for Adam """
    adam_state = {}
    for key, param in parameters.items():
        adam_state['m_' + key] = np.zeros_like(param)
        adam_state['v_' + key] = np.zeros_like(param)
    adam_state['t'] = 0 # Timestep counter
    return adam_state

def update_parameters_adam(parameters, grads, adam_state, learning_rate, beta1, beta2, epsilon):
    """ Updates parameters using Adam optimizer """
    adam_state['t'] += 1
    t = adam_state['t']

    for key in parameters.keys():
        # Get gradients
        grad_key = 'd' + key # e.g., 'dW1'
        grad = grads[grad_key]

        # Update momentum (m)
        adam_state['m_' + key] = beta1 * adam_state['m_' + key] + (1 - beta1) * grad
        # Update RMSprop (v)
        adam_state['v_' + key] = beta2 * adam_state['v_' + key] + (1 - beta2) * (grad**2)

        # Bias correction
        m_hat = adam_state['m_' + key] / (1 - beta1**t)
        v_hat = adam_state['v_' + key] / (1 - beta2**t)

        # Update parameters
        parameters[key] -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        # print(f"Updated {key}, grad norm: {np.linalg.norm(grad)}, param norm: {np.linalg.norm(parameters[key])}") # Debug

    # No return needed, parameters dict is updated in-place

# --- 4. Backpropagation (NumPy) ---
print("\nStep 4: Defining Backpropagation (NumPy)...")

def backward_pass(Y_pred_softmax, Y_true_one_hot, parameters, cache, dropout_prob):
    """
    Performs backpropagation to calculate gradients.
    Args:
        Y_pred_softmax: Output of softmax (batch_size, num_classes)
        Y_true_one_hot: Ground truth labels (batch_size, num_classes)
        parameters: Dictionary of weights and biases
        cache: Dictionary from forward pass containing intermediate values
        dropout_prob: Dropout probability (needed for scaling backprop)
    Returns:
        grads: Dictionary containing gradients (dW1, db1, dW2, db2, dW3, db3)
    """
    m = Y_true_one_hot.shape[0] # Batch size
    keep_prob = 1.0 - dropout_prob

    # Retrieve from cache and parameters
    W1, W2, W3 = parameters['W1'], parameters['W2'], parameters['W3']
    Z1, A1, D1, mask1 = cache['Z1'], cache['A1'], cache['D1'], cache['mask1']
    Z2, A2, D2, mask2 = cache['Z2'], cache['A2'], cache['D2'], cache['mask2']
    X = cache['X']

    # ---- Gradients for Output Layer (Layer 3) ----
    # Derivative of Loss w.r.t Z3 (logits)
    dZ3 = (Y_pred_softmax - Y_true_one_hot) / m # Average over batch size

    # Derivative of Loss w.r.t W3 and b3
    # Input to Layer 3 was D2
    dW3 = np.dot(D2.T, dZ3)
    db3 = np.sum(dZ3, axis=0, keepdims=True)

    # ---- Gradients for Hidden Layer 2 ----
    # Derivative of Loss w.r.t D2 (input to Layer 3)
    dD2 = np.dot(dZ3, W3.T)

    # Derivative of Loss w.r.t A2 (before Dropout 2)
    # Backprop through dropout: apply mask and scale
    dA2 = (dD2 * cache['mask2']) / keep_prob if keep_prob > 0 else dD2 # Apply mask used in forward pass

    # Derivative of Loss w.r.t Z2 (before ReLU 2)
    dZ2 = dA2 * relu_derivative(Z2) # Element-wise product

    # Derivative of Loss w.r.t W2 and b2
    # Input to Layer 2 was D1
    dW2 = np.dot(D1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    # ---- Gradients for Hidden Layer 1 ----
    # Derivative of Loss w.r.t D1 (input to Layer 2)
    dD1 = np.dot(dZ2, W2.T)

    # Derivative of Loss w.r.t A1 (before Dropout 1)
    dA1 = (dD1 * cache['mask1']) / keep_prob if keep_prob > 0 else dD1 # Apply mask used in forward pass

    # Derivative of Loss w.r.t Z1 (before ReLU 1)
    dZ1 = dA1 * relu_derivative(Z1) # Element-wise product

    # Derivative of Loss w.r.t W1 and b1
    # Input to Layer 1 was X
    dW1 = np.dot(X.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    grads = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2, 'dW3': dW3, 'db3': db3}
    return grads

# --- 5. Training Loop (NumPy) ---
print("\nStep 5: Training the Model (NumPy)...")

parameters = initialize_parameters(
    CONFIG["input_dim"], CONFIG["hidden_units_1"], CONFIG["hidden_units_2"], CONFIG["num_classes"]
)
adam_state = initialize_adam(parameters)

num_samples = X_train.shape[0]
num_batches = num_samples // CONFIG["batch_size"]

train_losses = []
train_accuracies = []

start_time = time.time()
for epoch in range(CONFIG["epochs"]):
    epoch_loss = 0.0
    epoch_correct = 0
    epoch_total = 0

    # Shuffle training data each epoch
    permutation = np.random.permutation(num_samples)
    X_train_shuffled = X_train[permutation]
    Y_train_one_hot_shuffled = Y_train_one_hot[permutation]
    Y_train_shuffled = Y_train[permutation] # Keep original labels for accuracy calc

    for i in range(num_batches):
        # Get batch
        start_idx = i * CONFIG["batch_size"]
        end_idx = start_idx + CONFIG["batch_size"]
        X_batch = X_train_shuffled[start_idx:end_idx]
        Y_batch_one_hot = Y_train_one_hot_shuffled[start_idx:end_idx]
        Y_batch_labels = Y_train_shuffled[start_idx:end_idx] # For accuracy

        # --- Forward Pass (Training Mode) ---
        Z3, cache = forward_pass(X_batch, parameters, CONFIG["dropout_prob"], mode='train')
        Y_pred_softmax = softmax(Z3)

        # --- Calculate Loss ---
        loss = cross_entropy_loss(Y_pred_softmax, Y_batch_one_hot)
        epoch_loss += loss * X_batch.shape[0] # Accumulate total loss

        # --- Backward Pass ---
        grads = backward_pass(Y_pred_softmax, Y_batch_one_hot, parameters, cache, CONFIG["dropout_prob"])

        # --- Update Parameters (Adam) ---
        update_parameters_adam(parameters, grads, adam_state, CONFIG["learning_rate"],
                               CONFIG["beta1"], CONFIG["beta2"], CONFIG["epsilon"])

        # --- Calculate Batch Accuracy ---
        predicted_classes = np.argmax(Y_pred_softmax, axis=1)
        epoch_correct += np.sum(predicted_classes == Y_batch_labels)
        epoch_total += X_batch.shape[0]

        if (i + 1) % 100 == 0:
             print(f'Epoch [{epoch+1}/{CONFIG["epochs"]}], Step [{i+1}/{num_batches}], Loss: {loss:.4f}')

    # End of epoch statistics
    avg_epoch_loss = epoch_loss / epoch_total
    avg_epoch_acc = epoch_correct / epoch_total
    train_losses.append(avg_epoch_loss)
    train_accuracies.append(avg_epoch_acc)
    print(f'--- End of Epoch {epoch+1} ---')
    print(f'Average Training Loss: {avg_epoch_loss:.4f}, Training Accuracy: {avg_epoch_acc:.4f}')
    print(f"Time elapsed: {time.time() - start_time:.2f} seconds")


print("Training finished.")
training_duration = time.time() - start_time
print(f"Total Training Time: {training_duration:.2f} seconds")

# --- 6. Standard Evaluation (Dropout OFF - NumPy) ---
print("\nStep 6: Evaluating Model Performance (Standard - Dropout OFF)...")

num_test_samples = X_test.shape[0]
test_batches = (num_test_samples + CONFIG["batch_size"] - 1) // CONFIG["batch_size"] # Handle last batch
correct_predictions = 0
total_samples = 0

for i in range(test_batches):
    start_idx = i * CONFIG["batch_size"]
    end_idx = min(start_idx + CONFIG["batch_size"], num_test_samples)
    X_batch = X_test[start_idx:end_idx]
    Y_batch_labels = Y_test[start_idx:end_idx]

    # --- Forward Pass (Evaluation Mode - Dropout OFF) ---
    # **Crucially, set mode='eval' here**
    Z3, _ = forward_pass(X_batch, parameters, CONFIG["dropout_prob"], mode='eval')
    Y_pred_softmax = softmax(Z3)

    predicted_classes = np.argmax(Y_pred_softmax, axis=1)
    correct_predictions += np.sum(predicted_classes == Y_batch_labels)
    total_samples += X_batch.shape[0]

accuracy = correct_predictions / total_samples
print(f'Standard Test Accuracy (Dropout OFF): {accuracy:.4f}')


# --- 7. Bayesian Inference with MC Dropout (Dropout ON - NumPy) ---
print("\nStep 7: Estimating Uncertainty using Monte Carlo Dropout (Dropout ON)...")

# Let's take one sample image from the test set
sample_image_np = X_test[0:1] # Keep batch dimension (1, 784)
sample_label = Y_test[0]

mc_predictions = []
print(f"Running {CONFIG['mc_samples']} Monte Carlo samples for the first test image...")

# Explanation: Perform T forward passes WITH dropout active.
# The forward_pass function needs the 'mc_dropout' or 'train' mode.
# We don't need gradients here, just the predictions.
for _ in range(CONFIG["mc_samples"]):
    # **Crucially, set mode='mc_dropout' or 'train' here**
    Z3, _ = forward_pass(sample_image_np, parameters, CONFIG["dropout_prob"], mode='mc_dropout')
    probabilities = softmax(Z3) # Z3 has shape (1, num_classes)
    mc_predictions.append(probabilities[0]) # Append the (num_classes,) array

# Stack the predictions into a single numpy array (T, num_classes)
mc_predictions = np.array(mc_predictions) # Shape: (mc_samples, num_classes)

# --- Calculate Mean Prediction and Uncertainty ---
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


# --- 8. Visualization ---
print("\nStep 8: Visualizing Results...")

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
# Un-normalize: Original range was [0, 255], ToTensor made it [0, 1], Normalize made it [-1, 1]
# Reverse: (X + 1) / 2 to get back to [0, 1]
img_display = (sample_image_np.reshape(28, 28) + 1.0) / 2.0
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


# --- 9. Saving the Model (NumPy) ---
print("\nStep 9: Saving the Model (NumPy)...")

# Explanation:
# We save the 'parameters' dictionary (containing W1, b1, ...) and potentially
# the 'adam_state' dictionary if we want to resume training later.
# np.savez allows saving multiple numpy arrays into a single .npz file.
# We use defaultdict to easily handle adding parameters to the save dictionary.
save_dict = defaultdict(dict)
save_dict['parameters'] = parameters
save_dict['adam_state'] = adam_state # Optional: save optimizer state
save_dict['config'] = CONFIG         # Save config used for training

try:
    np.savez(CONFIG["model_save_path"], **save_dict)
    print(f"Model parameters and state saved successfully to: {os.path.abspath(CONFIG['model_save_path'])}")
except Exception as e:
    print(f"Error saving model: {e}")

# Note on saving the dataset:
# As before, the dataset is managed by torchvision in the specified directory.
print(f"Dataset files are stored/managed by torchvision in: {os.path.abspath(CONFIG['data_dir'])}")

print("\n--- Script Finished ---")


# --- How to Load the Saved Model (NumPy Example) ---
# print("\n--- Example: Loading the Saved Model (NumPy) ---")
# try:
#     loaded_data = np.load(CONFIG["model_save_path"], allow_pickle=True) # Allow pickle for dictionaries
#
#     # Extract parameters (need .item() because it's stored as a 0-dim array containing the dict)
#     loaded_parameters = loaded_data['parameters'].item()
#     # loaded_adam_state = loaded_data['adam_state'].item() # Optional
#     # loaded_config = loaded_data['config'].item()       # Optional
#
#     print("Model parameters loaded successfully.")
#     # Now 'loaded_parameters' can be used with the forward_pass function.
#
#     # Example: Evaluate loaded model (dropout OFF)
#     Z3_loaded, _ = forward_pass(X_test[0:1], loaded_parameters, CONFIG["dropout_prob"], mode='eval')
#     probs_loaded = softmax(Z3_loaded)
#     print(f"Prediction from loaded model for first test image: {np.argmax(probs_loaded)}")
#
# except Exception as e:
#     print(f"Error loading model: {e}")
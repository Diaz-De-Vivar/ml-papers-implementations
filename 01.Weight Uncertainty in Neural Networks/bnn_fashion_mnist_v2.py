# -*- coding: utf-8 -*-
"""
Implementation of Bayes by Backprop (Weight Uncertainty in Neural Networks)
Paper: https://arxiv.org/abs/1505.05424
Dataset: Fashion MNIST
Framework: PyTorch
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import os
from tqdm import tqdm

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

BATCH_SIZE = 128
EPOCHS = 10 # Keep low for demonstration, might need more for good results
LEARNING_RATE = 1e-3
NUM_SAMPLES_TRAIN = 1 # Number of samples from q(w|theta) per input during training
NUM_SAMPLES_TEST = 10 # Number of samples from q(w|theta) per input during testing for uncertainty
PI = 0.5 # Mixture weight for prior (optional, see paper Sec 3.3) - simplified here
PRIOR_SIGMA1 = 1.0 # Prior std dev 1 (e.g., N(0, 1^2))
PRIOR_SIGMA2 = 0.1 # Prior std dev 2 (e.g., N(0, 0.1^2)) - for scale mixture prior
# For simplicity, we'll initially use a single Gaussian prior N(0, PRIOR_SIGMA1^2)
# Set PI = 1.0 to use only PRIOR_SIGMA1

DATA_DIR = './data'
MODEL_SAVE_PATH = './bayes_by_backprop_fashionmnist.pth'

# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

# --- Data Loading ---
print("Loading Fashion MNIST dataset...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # Fashion MNIST specific normalization
])

train_dataset = datasets.FashionMNIST(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=transform
)
test_dataset = datasets.FashionMNIST(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2, # Adjust based on your system
    pin_memory=True if DEVICE == 'cuda' else False
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True if DEVICE == 'cuda' else False
)

# Get number of batches for KL scaling
NUM_BATCHES = len(train_loader)
print(f"Number of training batches: {NUM_BATCHES}")

# --- Bayesian Layer ---
class BayesianLinear(nn.Module):
    """
    Implementation of a Linear layer with Bayesian inference on weights (and biases).
    Weights and biases are represented by distributions (Gaussian) parameterized by
    mean (mu) and standard deviation (sigma, derived from rho via softplus).
    """
    def __init__(self, in_features, out_features, use_bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

        # --- Variational Parameters ---
        # These are the parameters we learn during training (theta in the paper)
        # Weight parameters (mu and rho)
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))

        # Bias parameters (mu and rho) - optional
        if self.use_bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_features))
            self.bias_rho = nn.Parameter(torch.Tensor(out_features))
        else:
            # Register as None if not used - important for state_dict saving/loading
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        # --- Initialization ---
        # Sensible initialization is important.
        # Initialize mus using Kaiming uniform (good for ReLU activations)
        # Initialize rhos to a small negative value, leading to small initial sigma
        # Small initial sigma means the initial weights are close to their means
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        # Initial rho corresponding to sigma = approx 0.01 (log(exp(sigma)-1))
        # Example: rho = log(exp(0.01) - 1) approx -4.6
        nn.init.constant_(self.weight_rho, -4.6) # Small initial uncertainty
        if self.use_bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_mu, -bound, bound)
            nn.init.constant_(self.bias_rho, -4.6)

    def forward(self, x):
        """
        Performs the forward pass using the reparameterization trick and calculates
        the KL divergence component of the loss for this layer.
        """
        # --- Reparameterization Trick ---
        # 1. Calculate sigma from rho using softplus: sigma = log(1 + exp(rho))
        # Softplus ensures sigma is always positive.
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        if self.use_bias:
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))

        # 2. Sample epsilon from standard Gaussian N(0, 1)
        # Epsilon should have the same shape as the parameters (mu, rho, sigma)
        # We use *.data.new_empty(...).normal_(0, 1) for device compatibility
        epsilon_weight = self.weight_mu.data.new_empty(self.weight_mu.size()).normal_(0, 1)
        if self.use_bias:
            epsilon_bias = self.bias_mu.data.new_empty(self.bias_mu.size()).normal_(0, 1)

        # 3. Calculate sampled weights and biases: w = mu + sigma * epsilon
        weight = self.weight_mu + weight_sigma * epsilon_weight
        if self.use_bias:
            bias = self.bias_mu + bias_sigma * epsilon_bias
        else:
            bias = None

        # --- Calculate KL Divergence Component ---
        # This is log q(w|theta) - log P(w)
        # log q(w|theta): Log probability of the sampled w under the variational posterior N(mu, sigma^2)
        # log P(w): Log probability of the sampled w under the prior distribution
        # We use the analytical KL divergence between two Gaussians for simplicity here.
        # KL[q(w|theta) || P(w)] = KL[N(mu, sigma^2) || N(0, prior_sigma^2)]

        # Prior: Simple Gaussian N(0, PRIOR_SIGMA1^2)
        # Note: Could implement scale mixture prior from paper here if desired.
        prior_log_sigma = math.log(PRIOR_SIGMA1)

        # KL divergence for weights q(w|mu, rho) || P(w) = N(0, prior_sigma^2)
        kl_weights = self.kl_divergence(self.weight_mu, weight_sigma, 0.0, PRIOR_SIGMA1)

        # KL divergence for biases q(b|mu, rho) || P(b) = N(0, prior_sigma^2)
        if self.use_bias:
            kl_bias = self.kl_divergence(self.bias_mu, bias_sigma, 0.0, PRIOR_SIGMA1)
            kl_div = kl_weights + kl_bias
        else:
            kl_div = kl_weights

        # --- Linear Transformation ---
        # Perform the standard linear operation using the *sampled* weights and biases
        # F.linear computes X @ W.T + b
        output = F.linear(x, weight, bias)

        return output, kl_div

    @staticmethod
    def kl_divergence(mu_q, sigma_q, mu_p, sigma_p):
        """
        Calculate KL divergence KL[ N(mu_q, sigma_q^2) || N(mu_p, sigma_p^2) ]
        Assumes diagonal covariance matrices.
        Formula: log(sigma_p/sigma_q) + (sigma_q^2 + (mu_q-mu_p)^2)/(2*sigma_p^2) - 0.5
        Summed over all elements.
        """
        log_sigma_p = math.log(sigma_p)
        log_sigma_q = torch.log(sigma_q + 1e-10) # Add epsilon for numerical stability
        sigma_q_sq = sigma_q**2
        sigma_p_sq = sigma_p**2
        mu_diff_sq = (mu_q - mu_p)**2

        kl = (log_sigma_p - log_sigma_q +
              (sigma_q_sq + mu_diff_sq) / (2 * sigma_p_sq) - 0.5)

        return kl.sum() # Sum KL over all parameters in the layer

# --- Bayesian Network ---
class BayesianMLP(nn.Module):
    """
    A simple Multi-Layer Perceptron using BayesianLinear layers.
    """
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super().__init__()
        self.fc1 = BayesianLinear(input_dim, hidden_dim1)
        self.fc2 = BayesianLinear(hidden_dim1, hidden_dim2)
        self.fc3 = BayesianLinear(hidden_dim2, output_dim)

    def forward(self, x):
        """
        Forward pass through the network, accumulating KL divergence.
        """
        # Flatten input image
        x = x.view(x.size(0), -1) # Shape: [batch_size, 784]

        total_kl = 0.0

        # Layer 1
        x, kl1 = self.fc1(x)
        x = F.relu(x)
        total_kl += kl1

        # Layer 2
        x, kl2 = self.fc2(x)
        x = F.relu(x)
        total_kl += kl2

        # Layer 3 (Output)
        x, kl3 = self.fc3(x)
        # No activation here, as CrossEntropyLoss expects logits
        total_kl += kl3

        return x, total_kl # Return logits and total KL divergence

# --- Loss Function (ELBO) ---
def calculate_loss(model_output, target, total_kl, num_batches):
    """
    Calculates the loss based on the Evidence Lower Bound (ELBO).
    Loss = Complexity Cost (KL Divergence) + Data Fit Cost (Negative Log Likelihood)
    """
    # 1. Data Fit Cost (Negative Log Likelihood - NLL)
    # We use CrossEntropyLoss which combines LogSoftmax and NLLLoss.
    # It measures how well the predictions match the actual labels.
    # We use reduction='mean' to average over the batch.
    nll = F.cross_entropy(model_output, target, reduction='mean')

    # 2. Complexity Cost (KL Divergence)
    # This is the accumulated KL divergence from all Bayesian layers.
    # The paper suggests scaling this term (Eq. 8). A common way is to divide
    # by the number of mini-batches in the dataset. This effectively amortizes
    # the KL cost over the entire dataset throughout the epoch.
    # total_kl is the sum for the current mini-batch.
    kl_cost = total_kl / num_batches

    # 3. Total Loss (Negative ELBO approximation)
    loss = kl_cost + nll

    return loss, kl_cost, nll

# --- Training Function ---
def train_epoch(model, optimizer, train_loader, num_batches, epoch):
    model.train() # Set model to training mode
    total_loss = 0.0
    total_kl_cost = 0.0
    total_nll_cost = 0.0
    correct_predictions = 0
    total_samples = 0

    # Use tqdm for a progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]")

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(DEVICE), target.to(DEVICE)

        optimizer.zero_grad() # Clear previous gradients

        # --- Forward Pass with Multiple Samples (Optional but better estimate) ---
        # Average predictions and KL divergences over NUM_SAMPLES_TRAIN
        batch_loss = 0.0
        batch_kl = 0.0
        batch_nll = 0.0
        batch_outputs = [] # Store outputs for accuracy calculation

        for _ in range(NUM_SAMPLES_TRAIN):
            model_output, kl_div = model(data)
            loss, kl_cost, nll = calculate_loss(model_output, target, kl_div, num_batches)

            # Divide loss by number of samples before backprop
            loss_sampled = loss / NUM_SAMPLES_TRAIN
            batch_loss += loss_sampled.item() # Accumulate loss value
            batch_kl += kl_cost.item() / NUM_SAMPLES_TRAIN
            batch_nll += nll.item() / NUM_SAMPLES_TRAIN
            batch_outputs.append(F.log_softmax(model_output, dim=1)) # Store log-probabilities

            # --- Backward Pass and Optimization ---
            loss_sampled.backward() # Calculate gradients for this sample

        optimizer.step() # Update model parameters (mu, rho) based on accumulated gradients

        # --- Calculate Accuracy for the Batch ---
        # Average the log-probabilities across samples
        avg_log_probs = torch.logsumexp(torch.stack(batch_outputs, dim=0), dim=0) - math.log(NUM_SAMPLES_TRAIN)
        pred = avg_log_probs.argmax(dim=1, keepdim=True) # Get the index of the max log-probability
        correct_predictions += pred.eq(target.view_as(pred)).sum().item()
        total_samples += target.size(0)

        # --- Logging ---
        total_loss += batch_loss
        total_kl_cost += batch_kl
        total_nll_cost += batch_nll

        # Update progress bar description
        pbar.set_postfix({
            'Loss': f'{batch_loss:.4f}',
            'KL': f'{batch_kl:.4f}',
            'NLL': f'{batch_nll:.4f}',
            'Acc': f'{100. * correct_predictions / total_samples:.2f}%'
        })

    avg_loss = total_loss / len(train_loader)
    avg_kl = total_kl_cost / len(train_loader)
    avg_nll = total_nll_cost / len(train_loader)
    accuracy = 100. * correct_predictions / len(train_loader.dataset)

    print(f"Epoch {epoch+1} Train Summary: Avg Loss: {avg_loss:.4f}, Avg KL: {avg_kl:.4f}, Avg NLL: {avg_nll:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy

# --- Evaluation Function ---
def evaluate(model, test_loader, num_batches, num_samples=NUM_SAMPLES_TEST):
    model.eval() # Set model to evaluation mode
    total_loss = 0.0
    total_kl_cost = 0.0
    total_nll_cost = 0.0
    correct_predictions = 0
    total_samples = 0

    # Use tqdm for a progress bar
    pbar = tqdm(test_loader, desc="[Evaluating]")

    with torch.no_grad(): # Disable gradient calculations for efficiency
        for data, target in pbar:
            data, target = data.to(DEVICE), target.to(DEVICE)

            # --- Prediction with Multiple Samples ---
            # To get a robust prediction and estimate uncertainty, we run the
            # forward pass multiple times with different sampled weights.
            batch_outputs_mc = [] # Store outputs for each Monte Carlo sample
            batch_kl_mc = 0.0

            for _ in range(num_samples):
                model_output, kl_div = model(data)
                batch_outputs_mc.append(F.log_softmax(model_output, dim=1))
                # Note: KL calculation might still be useful for monitoring during eval
                batch_kl_mc += kl_div.item() # Accumulate KL for average

            # Average KL over samples
            avg_kl = batch_kl_mc / num_samples

            # Average the log-probabilities across samples
            # LogSumExp trick for numerical stability: log(1/N * sum(exp(logits)))
            avg_log_probs = torch.logsumexp(torch.stack(batch_outputs_mc, dim=0), dim=0) - math.log(num_samples)

            # Calculate NLL using the averaged prediction
            # Note: Loss calculation during evaluation is mainly for monitoring.
            # The primary metric is accuracy.
            nll = F.nll_loss(avg_log_probs, target, reduction='mean') # Use nll_loss as input is log-softmax
            kl_cost = avg_kl / num_batches # Use the same scaling as training
            loss = kl_cost + nll.item() # Be careful: nll is already averaged over batch

            total_loss += loss * data.size(0) # Accumulate total loss (not avg)
            total_kl_cost += kl_cost * data.size(0)
            total_nll_cost += nll.item() * data.size(0)

            # --- Calculate Accuracy ---
            pred = avg_log_probs.argmax(dim=1, keepdim=True) # Get the index of the max log-probability
            correct_predictions += pred.eq(target.view_as(pred)).sum().item()
            total_samples += target.size(0)

            # Update progress bar description
            pbar.set_postfix({
                'AvgLoss': f'{loss:.4f}', # Loss for this batch
                'Accuracy': f'{100. * correct_predictions / total_samples:.2f}%'
            })

    avg_loss = total_loss / len(test_loader.dataset)
    avg_kl = total_kl_cost / len(test_loader.dataset)
    avg_nll = total_nll_cost / len(test_loader.dataset)
    accuracy = 100. * correct_predictions / len(test_loader.dataset)

    print(f"Test Summary: Avg Loss: {avg_loss:.4f}, Avg KL: {avg_kl:.4f}, Avg NLL: {avg_nll:.4f}, Accuracy: {accuracy:.2f}% ({correct_predictions}/{len(test_loader.dataset)})")
    return avg_loss, accuracy


# --- Main Execution ---
if __name__ == "__main__":
    INPUT_DIM = 28 * 28 # Fashion MNIST image size
    HIDDEN_DIM1 = 400
    HIDDEN_DIM2 = 200
    OUTPUT_DIM = 10 # 10 classes

    print("Initializing model and optimizer...")
    model = BayesianMLP(INPUT_DIM, HIDDEN_DIM1, HIDDEN_DIM2, OUTPUT_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Starting training...")
    best_test_accuracy = 0.0
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, optimizer, train_loader, NUM_BATCHES, epoch)
        test_loss, test_acc = evaluate(model, test_loader, NUM_BATCHES)

        # Save model if it has the best test accuracy so far
        if test_acc > best_test_accuracy:
            print(f"New best test accuracy: {test_acc:.2f}%. Saving model...")
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            best_test_accuracy = test_acc

    print("Training finished.")
    print(f"Best test accuracy achieved: {best_test_accuracy:.2f}%")

    # --- Example: Prediction with Uncertainty ---
    # Load the best model
    print("\nLoading best model for prediction example...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()

    # Get a single image from the test set
    data_iterator = iter(test_loader)
    sample_data, sample_target = next(data_iterator)
    image = sample_data[0].to(DEVICE) # Take the first image
    true_label = sample_target[0].item()

    # Fashion MNIST Class Labels
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    print(f"\nPredicting for a sample image (True Label: {classes[true_label]})")

    # Predict multiple times using different weight samples
    num_predict_samples = 100
    predictions_mc = []
    with torch.no_grad():
        for _ in range(num_predict_samples):
            # Need to add batch dimension for the model
            image_batch = image.unsqueeze(0)
            output_logits, _ = model(image_batch)
            # Convert logits to probabilities
            probabilities = F.softmax(output_logits, dim=1)
            predictions_mc.append(probabilities.squeeze().cpu().numpy())

    # Analyze the Monte Carlo predictions
    predictions_mc = torch.tensor(predictions_mc) # Shape: [num_predict_samples, num_classes]

    # Calculate mean prediction
    mean_prediction = predictions_mc.mean(dim=0)
    predicted_class_index = mean_prediction.argmax().item()
    predicted_class_prob = mean_prediction.max().item()

    # Calculate predictive entropy (a measure of uncertainty)
    # Entropy = - sum(p * log(p))
    predictive_entropy = -torch.sum(mean_prediction * torch.log(mean_prediction + 1e-10)).item()

    # Calculate variance of predictions (another measure of uncertainty)
    variance_prediction = predictions_mc.var(dim=0).mean().item() # Average variance across classes


    print(f"  Mean Predicted Class: {classes[predicted_class_index]} (Probability: {predicted_class_prob:.4f})")
    print(f"  Predictive Entropy: {predictive_entropy:.4f}")
    print(f"  Mean Prediction Variance: {variance_prediction:.6f}")

    # You could also plot the distribution of probabilities for the predicted class
    # across the Monte Carlo samples to visualize uncertainty.
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.hist(predictions_mc[:, predicted_class_index].numpy(), bins=20, alpha=0.7)
    plt.title(f"Distribution of Probabilities for Predicted Class '{classes[predicted_class_index]}'")
    plt.xlabel("Probability")
    plt.ylabel("Frequency (out of 100 samples)")
    plt.show()
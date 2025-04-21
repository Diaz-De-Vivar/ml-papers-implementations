# -*- coding: utf-8 -*-
"""
Implementation of Bayes by Backprop (Weight Uncertainty in Neural Networks)
Paper: https://arxiv.org/abs/1505.05424
Dataset: Fashion MNIST
Framework: PyTorch
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# -----------------------------
# --- 1. Configuration --------
# -----------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Hyperparameters (can be tuned)
BATCH_SIZE = 128
EPOCHS = 10 # Keep low for demonstration, increase for better results
LEARNING_RATE = 0.001
NUM_HIDDEN_UNITS = 400
NUM_SAMPLES_TRAIN = 1 # Number of samples from posterior during training perdatapoint
NUM_SAMPLES_TEST = 10 # Number of samples from posterior during testing for averaging
PRIOR_SIGMA_1 = 0.1 # Sigma for the prior distribution (scale mixture prior component 1)
PRIOR_SIGMA_2 = 0.4 # Sigma for the prior distribution (scale mixture prior component 2)
PRIOR_PI = 0.5      # Mixture weight for the prior (optional, simpler prior below)
SIMPLE_PRIOR_SIGMA = 0.1 # Sigma for a simple Gaussian prior N(0, sigma^2)
USE_SIMPLE_PRIOR = True # Use a simple N(0, sigma^2) prior instead of scale mixture

# Data loading parameters
DATA_DIR = './data'

# -----------------------------
# --- 2. Data Loading ---------
# -----------------------------

def load_fashion_mnist():
    """Loads and prepares Fashion MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # MNIST mean/std
        # Note: Fashion MNIST stats are slightly different (approx. 0.286, 0.353)
        # but MNIST stats are often used and work reasonably well.
        # For optimal results, recalculate for Fashion MNIST.
        # transforms.Normalize((0.2860,), (0.3530,))
    ])

    train_dataset = datasets.FashionMNIST(
        DATA_DIR, train=True, download=True, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        DATA_DIR, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    num_batches = len(train_loader)
    return train_loader, test_loader, num_batches

# -----------------------------
# --- 3. Bayesian Layer -------
# -----------------------------

class BayesianLinear(nn.Module):
    """
    Linear layer with weights and biases sampled from Gaussian distributions.
    Implements the local reparameterization trick and KL divergence calculation.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Parameters for weight distribution q(W|theta)
        # Initialized similar to Kaiming uniform for mu, and small negative for rho
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features)) # log(sigma^2) is not used, we use log(1+exp(rho)) for sigma

        # Parameters for bias distribution q(b|theta)
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))

        self.reset_parameters()

        # Prior distributions p(W), p(b) - Assuming standard Gaussian for simplicity
        # Constants, not trainable parameters
        self.prior_weight_mu = 0.0
        self.prior_weight_sigma = SIMPLE_PRIOR_SIGMA if USE_SIMPLE_PRIOR else None
        self.prior_bias_mu = 0.0
        self.prior_bias_sigma = SIMPLE_PRIOR_SIGMA if USE_SIMPLE_PRIOR else None

        # Placeholders for sampled weights and biases and KL divergence
        self.sampled_weight = None
        self.sampled_bias = None
        self.kl_divergence_weight = None
        self.kl_divergence_bias = None

    def reset_parameters(self):
        # Initialization similar to Kaiming Uniform
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        # Initialize rho to a small negative value for small initial variance
        # e.g., sigma = log(1 + exp(-6)) approx 0.0025
        nn.init.constant_(self.weight_rho, -6.0)
        # Initialize bias parameters
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.bias_rho, -6.0)

    def forward(self, x):
        """
        Forward pass with sampling using the reparameterization trick.
        x: input tensor [batch_size, in_features]
        """
        # 1. Calculate sigma from rho: sigma = log(1 + exp(rho)) -> Softplus
        # This ensures sigma is always positive.
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))

        # 2. Sample epsilon from standard normal N(0, I)
        # Size should match the parameters mu and sigma
        epsilon_weight = torch.randn_like(weight_sigma)
        epsilon_bias = torch.randn_like(bias_sigma)

        # 3. Sample weights and biases using reparameterization trick: w = mu + sigma * epsilon
        self.sampled_weight = self.weight_mu + weight_sigma * epsilon_weight
        self.sampled_bias = self.bias_mu + bias_sigma * epsilon_bias

        # 4. Calculate the KL divergence between posterior q(w|theta) and prior p(w)
        self.kl_divergence_weight = self._calculate_kl_gaussian(
            self.weight_mu, weight_sigma, self.prior_weight_mu, self.prior_weight_sigma
        )
        self.kl_divergence_bias = self._calculate_kl_gaussian(
            self.bias_mu, bias_sigma, self.prior_bias_mu, self.prior_bias_sigma
        )

        # 5. Perform the linear operation using the *sampled* weights and biases
        output = F.linear(x, self.sampled_weight, self.sampled_bias)
        return output

    def _calculate_kl_gaussian(self, mu_q, sigma_q, mu_p, sigma_p):
        """
        Calculate KL divergence between two Gaussian distributions KL(q || p).
        If sigma_p is None, assumes a scale mixture prior (more complex).
        For simplicity, we'll implement the standard Gaussian prior KL divergence.
        KL( N(mu_q, sigma_q^2) || N(mu_p, sigma_p^2) ) =
            log(sigma_p / sigma_q) + (sigma_q^2 + (mu_q - mu_p)^2) / (2 * sigma_p^2) - 0.5
        """
        if USE_SIMPLE_PRIOR and sigma_p is not None:
            # Ensure sigma_q is not exactly zero for numerical stability
            sigma_q_safe = sigma_q + 1e-8
            sigma_p_safe = sigma_p + 1e-8

            kl = (torch.log(sigma_p_safe / sigma_q_safe) +
                  (sigma_q_safe**2 + (mu_q - mu_p)**2) / (2 * sigma_p_safe**2) - 0.5)
            return kl.sum() # Sum KL over all parameters in the layer
        else:
            # Implementation for Scale Mixture Prior (optional, from paper)
            # Prior p(w) = pi * N(w|0, sigma1^2) + (1-pi) * N(w|0, sigma2^2)
            # KL(q || p) = sum over weights [ KL(q(w_i) || p(w_i)) ]
            # KL(q(w_i) || p(w_i)) = E_q [log q(w_i) - log p(w_i)]
            # log p(w) = log [ pi*N(w|0,s1^2) + (1-pi)*N(w|0,s2^2) ]
            # This integral is tricky. The paper approximates KL(q||p) instead.
            # Let's stick to the simpler Gaussian prior for this implementation.
            # If you want the scale mixture, you need to approximate the KL term,
            # often done using Monte Carlo sampling from q during KL calculation or
            # using the closed-form approximation mentioned in the paper's appendix.
            # For now, raise an error if complex prior is selected without implementation.
            raise NotImplementedError("Scale Mixture Prior KL calculation not implemented. Set USE_SIMPLE_PRIOR=True.")


    def kl_divergence(self):
        """Returns the total KL divergence for the layer."""
        return self.kl_divergence_weight + self.kl_divergence_bias

# -----------------------------
# --- 4. Bayesian MLP Model ---
# -----------------------------

class BayesianMLP(nn.Module):
    """A simple Bayesian Multi-Layer Perceptron."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = BayesianLinear(input_dim, hidden_dim)
        self.fc2 = BayesianLinear(hidden_dim, hidden_dim)
        self.fc3 = BayesianLinear(hidden_dim, output_dim)

    def forward(self, x):
        # Flatten the input image
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # Output layer (logits)
        return x

    def kl_divergence(self):
        """Calculate the total KL divergence for all Bayesian layers in the model."""
        total_kl = 0
        for module in self.modules():
            if isinstance(module, BayesianLinear):
                total_kl += module.kl_divergence()
        return total_kl

# -----------------------------
# --- 5. Loss Function (ELBO) ---
# -----------------------------

def calculate_elbo_loss(model, inputs, targets, num_batches, num_samples=1):
    """
    Calculates the Evidence Lower Bound (ELBO) loss, which is:
    ELBO = E_q[log P(D|w)] - KL(q(w|theta) || p(w))
    We want to *maximize* ELBO, which is equivalent to *minimizing* -ELBO.
    -ELBO = - E_q[log P(D|w)] + KL(q(w|theta) || p(w))
          = NLL + KL_divergence

    Args:
        model: The Bayesian neural network.
        inputs: Input data tensor.
        targets: Target labels tensor.
        num_batches: Total number of batches in the training dataset (for scaling KL).
        num_samples: Number of Monte Carlo samples to approximate the expectation E_q.

    Returns:
        Negative ELBO loss, Negative Log Likelihood (NLL), KL divergence.
    """
    total_negative_log_likelihood = 0.0
    total_kl_divergence = 0.0

    for _ in range(num_samples):
        # Forward pass uses sampled weights automatically due to BayesianLinear.forward()
        outputs = model(inputs)

        # Calculate Negative Log Likelihood (Cross-Entropy Loss)
        # Use reduction='sum' initially, then average over samples and batch
        nll = F.cross_entropy(outputs, targets, reduction='sum')
        total_negative_log_likelihood += nll

        # KL divergence is calculated once per forward pass within the layers
        # but needs to be retrieved after the forward pass.
        # For multiple samples, the KL term remains the same for the *distribution*,
        # so we calculate it based on the *last* sample's forward pass.
        # (Technically, KL is about the distributions q and p, not the specific samples w)
        total_kl_divergence = model.kl_divergence() # Retrieves sum from all layers

    # Average NLL over the number of samples
    avg_negative_log_likelihood = total_negative_log_likelihood / num_samples

    # Calculate the loss: -ELBO = NLL_batch + (KL_total / num_batches)
    # We scale the KL term as suggested in the paper (Section 3.4)
    # This balances the contribution of the likelihood and the prior/complexity term.
    # The KL divergence term is summed over *all* weights in the network.
    # The NLL term is summed over the *batch*. To make them comparable,
    # the KL term (representing the whole network prior mismatch) is often scaled
    # down by the number of batches (or dataset size).
    kl_weight = 1.0 / num_batches
    loss = (avg_negative_log_likelihood + kl_weight * total_kl_divergence) / inputs.size(0) # Average loss per data point in batch

    # Return average NLL and KL per data point as well for monitoring
    nll_per_datapoint = avg_negative_log_likelihood / inputs.size(0)
    kl_per_datapoint = (kl_weight * total_kl_divergence) / inputs.size(0)


    return loss, nll_per_datapoint, kl_per_datapoint


# -----------------------------
# --- 6. Training Loop --------
# -----------------------------

def train_model(model, optimizer, train_loader, epoch, num_batches):
    """Trains the model for one epoch."""
    model.train() # Set model to training mode
    total_loss = 0
    total_nll = 0
    total_kl = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)

        optimizer.zero_grad()

        # Calculate loss using ELBO
        loss, nll, kl_div = calculate_elbo_loss(
            model, data, target, num_batches, num_samples=NUM_SAMPLES_TRAIN
        )

        loss.backward() # Backpropagate the gradients
        optimizer.step() # Update model parameters (mu and rho)

        total_loss += loss.item()
        total_nll += nll.item()
        total_kl += kl_div.item()

        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item():.4f} (NLL: {nll.item():.4f}, KL: {kl_div.item():.4f})')

    avg_loss = total_loss / len(train_loader)
    avg_nll = total_nll / len(train_loader)
    avg_kl = total_kl / len(train_loader)
    print(f'\nEpoch {epoch} Average Training Loss: {avg_loss:.4f} (NLL: {avg_nll:.4f}, KL: {avg_kl:.4f})')

# -----------------------------
# --- 7. Evaluation Function --
# -----------------------------

def evaluate_model(model, test_loader, num_samples=10):
    """Evaluates the model on the test set using Monte Carlo sampling."""
    model.eval() # Set model to evaluation mode
    correct = 0
    total = 0
    all_probs = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            batch_probs = []
            for _ in range(num_samples):
                # Get logits from one sample of weights
                outputs = model(data)
                # Convert logits to probabilities using softmax
                probs = F.softmax(outputs, dim=1)
                batch_probs.append(probs.cpu().numpy())

            # Average probabilities across samples for each data point in the batch
            # Shape: (num_samples, batch_size, num_classes) -> (batch_size, num_classes)
            avg_probs = np.mean(batch_probs, axis=0)

            # Get predicted class from averaged probabilities
            predicted = np.argmax(avg_probs, axis=1)
            predicted = torch.from_numpy(predicted).to(DEVICE) # Move back to device if needed for comparison

            total += target.size(0)
            correct += (predicted == target).sum().item()
            all_probs.extend(avg_probs) # Store probabilities if needed for uncertainty analysis

    accuracy = 100. * correct / total
    print(f'\nTest Set Accuracy ({num_samples} samples): {correct}/{total} ({accuracy:.2f}%)\n')
    return accuracy, all_probs

# -----------------------------
# --- 8. Main Execution -------
# -----------------------------

if __name__ == '__main__':
    # 1. Load Data
    train_loader, test_loader, num_batches = load_fashion_mnist()
    input_dim = 28 * 28 # Fashion MNIST image size
    output_dim = 10   # Number of classes

    # 2. Initialize Model
    bnn_model = BayesianMLP(input_dim, NUM_HIDDEN_UNITS, output_dim).to(DEVICE)
    print("Model Architecture:")
    print(bnn_model)

    # 3. Initialize Optimizer
    optimizer = optim.Adam(bnn_model.parameters(), lr=LEARNING_RATE)

    # 4. Training Loop
    print("\n--- Starting Training ---")
    for epoch in range(1, EPOCHS + 1):
        train_model(bnn_model, optimizer, train_loader, epoch, num_batches)
        # Evaluate after each epoch (optional, can slow down training)
        evaluate_model(bnn_model, test_loader, num_samples=NUM_SAMPLES_TEST)
    print("--- Training Finished ---")

    # 5. Final Evaluation
    print("\n--- Final Evaluation ---")
    final_accuracy, final_probs = evaluate_model(bnn_model, test_loader, num_samples=NUM_SAMPLES_TEST)

    # Example: Inspect uncertainty for the first test image
    # print("\nExample: Prediction probabilities for the first test image:")
    # print(final_probs[0])
    # print(f"Predicted class: {np.argmax(final_probs[0])}")
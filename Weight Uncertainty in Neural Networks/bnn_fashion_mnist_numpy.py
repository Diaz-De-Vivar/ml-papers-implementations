import numpy as np
import gzip
import os
import time
from tqdm import tqdm # Using tqdm for progress bars is still okay

# --- Configuration ---
# Using fewer epochs/smaller network due to NumPy slowness
BATCH_SIZE = 128
EPOCHS = 3 # Reduced for NumPy speed
LEARNING_RATE = 1e-4 # May need adjustment
NUM_SAMPLES_TRAIN = 1
NUM_SAMPLES_TEST = 10
# PRIOR_SIGMA1 = 1.0 # Using a simple Gaussian prior N(0, 1)
PRIOR_VAR = 1.0 # Variance of the prior N(0, PRIOR_VAR)
# Rho initialization corresponding to small initial sigma (e.g., sigma=0.01)
# sigma = log(1+exp(rho)) -> rho = log(exp(sigma)-1)
INITIAL_RHO = np.log(np.expm1(0.01)) # Approx -4.6

DATA_DIR = './data/FashionMNIST/raw/' # Assumes data is here
MODEL_SAVE_PATH = './bayes_by_backprop_numpy_fashionmnist.npz'

# Numerical stability constant
EPSILON = 1e-10

# --- Data Loading (Manual MNIST Parser) ---
def load_mnist_images(filename):
    """Loads MNIST images from a .gz file."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}. Please download the Fashion MNIST dataset (e.g., by running the PyTorch version once).")
    with gzip.open(filename, 'rb') as f:
        # Read metadata (magic number, number of images, rows, columns)
        _magic_number = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        num_rows = int.from_bytes(f.read(4), 'big')
        num_cols = int.from_bytes(f.read(4), 'big')
        # Read image data
        img_data = np.frombuffer(f.read(), dtype=np.uint8)
        # Reshape into [num_images, num_rows * num_cols]
        images = img_data.reshape(num_images, num_rows * num_cols)
    return images

def load_mnist_labels(filename):
    """Loads MNIST labels from a .gz file."""
    if not os.path.exists(filename):
         raise FileNotFoundError(f"File not found: {filename}. Please download the Fashion MNIST dataset.")
    with gzip.open(filename, 'rb') as f:
        # Read metadata (magic number, number of items)
        _magic_number = int.from_bytes(f.read(4), 'big')
        num_items = int.from_bytes(f.read(4), 'big')
        # Read label data
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

print("Loading Fashion MNIST dataset using NumPy...")
try:
    train_images = load_mnist_images(os.path.join(DATA_DIR, 'train-images-idx3-ubyte.gz'))
    train_labels = load_mnist_labels(os.path.join(DATA_DIR, 'train-labels-idx1-ubyte.gz'))
    test_images = load_mnist_images(os.path.join(DATA_DIR, 't10k-images-idx3-ubyte.gz'))
    test_labels = load_mnist_labels(os.path.join(DATA_DIR, 't10k-labels-idx1-ubyte.gz'))

    # --- Preprocessing ---
    # Convert to float32
    train_images = train_images.astype(np.float32)
    test_images = test_images.astype(np.float32)

    # Normalize (simple scaling to [0, 1] and then standardization)
    train_images /= 255.0
    test_images /= 255.0
    # Use pre-calculated Fashion MNIST mean/std
    mean = 0.1307
    std = 0.3081
    train_images = (train_images - mean) / std
    test_images = (test_images - mean) / std

    print(f"Training data shape: {train_images.shape}, Labels shape: {train_labels.shape}")
    print(f"Test data shape: {test_images.shape}, Labels shape: {test_labels.shape}")

    NUM_TRAIN_SAMPLES = train_images.shape[0]
    NUM_TEST_SAMPLES = test_images.shape[0]
    INPUT_DIM = train_images.shape[1]
    OUTPUT_DIM = int(train_labels.max() + 1) # Should be 10

    NUM_BATCHES = NUM_TRAIN_SAMPLES // BATCH_SIZE

except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    print("Please ensure the Fashion MNIST raw files (.gz) are in the ./data/FashionMNIST/raw/ directory.")
    exit()

# --- Activation Functions ---
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(x.dtype)

def softmax(x):
    """Numerically stable softmax."""
    # Subtract max for numerical stability (avoids large exponents)
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def log_softmax(x):
     """Numerically stable log_softmax."""
     max_x = np.max(x, axis=-1, keepdims=True)
     log_sum_exp = max_x + np.log(np.sum(np.exp(x - max_x), axis=-1, keepdims=True))
     return x - log_sum_exp

# --- Helper Functions ---
def softplus(x):
    # np.log1p(np.exp(x)) is log(1+exp(x))
    # More stable version: max(x, 0) + log(1 + exp(-abs(x)))
    # Simpler numpy version:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x,0) # More stable
    # return np.log1p(np.exp(x)) # Less stable for large negative x

def softplus_derivative(x):
    # Derivative of log(1+exp(x)) is exp(x) / (1+exp(x)) = 1 / (1 + exp(-x)) (sigmoid)
    return 1.0 / (1.0 + np.exp(-x))

def kaiming_uniform_init(shape, fan_in):
    """Kaiming Uniform initialization for weights."""
    bound = np.sqrt(6.0 / fan_in)
    return np.random.uniform(-bound, bound, size=shape).astype(np.float32)

def uniform_init(shape, fan_in):
    """Standard uniform initialization for biases."""
    bound = 1.0 / np.sqrt(fan_in) if fan_in > 0 else 0
    return np.random.uniform(-bound, bound, size=shape).astype(np.float32)

# --- Bayesian Layer (NumPy) ---
class BayesianLinearNP:
    def __init__(self, in_features, out_features, use_bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

        # Variational Parameters (NumPy arrays)
        self.weight_mu = kaiming_uniform_init((out_features, in_features), in_features)
        self.weight_rho = np.full((out_features, in_features), INITIAL_RHO, dtype=np.float32)

        if self.use_bias:
            self.bias_mu = uniform_init((out_features,), in_features)
            self.bias_rho = np.full((out_features,), INITIAL_RHO, dtype=np.float32)
        else:
            self.bias_mu = None
            self.bias_rho = None

        # Placeholders for forward pass intermediates needed for backward pass
        self.inputs = None
        self.epsilon_w = None
        self.epsilon_b = None
        self.sampled_weight = None
        self.sampled_bias = None
        self.weight_sigma = None
        self.bias_sigma = None

        # Gradients for parameters (accumulated during backprop)
        self.grad_weight_mu = np.zeros_like(self.weight_mu)
        self.grad_weight_rho = np.zeros_like(self.weight_rho)
        if self.use_bias:
            self.grad_bias_mu = np.zeros_like(self.bias_mu)
            self.grad_bias_rho = np.zeros_like(self.bias_rho)
        else:
            self.grad_bias_mu = None
            self.grad_bias_rho = None

    def forward(self, x):
        """Forward pass with reparameterization and KL calculation."""
        self.inputs = x # Store input for backprop

        # Calculate sigma from rho
        self.weight_sigma = softplus(self.weight_rho)
        if self.use_bias:
            self.bias_sigma = softplus(self.bias_rho)

        # Sample epsilon
        self.epsilon_w = np.random.randn(*self.weight_mu.shape).astype(np.float32)
        if self.use_bias:
            self.epsilon_b = np.random.randn(*self.bias_mu.shape).astype(np.float32)

        # Sample weights and biases (Reparameterization Trick)
        self.sampled_weight = self.weight_mu + self.weight_sigma * self.epsilon_w
        if self.use_bias:
            self.sampled_bias = self.bias_mu + self.bias_sigma * self.epsilon_b
        else:
            self.sampled_bias = None

        # --- Linear Transformation ---
        # Output = Input @ Weight.T + Bias
        output = np.dot(x, self.sampled_weight.T)
        if self.use_bias:
            output += self.sampled_bias

        # --- Calculate KL Divergence Component ---
        # KL[ N(mu, sigma^2) || N(0, prior_var) ]
        # Formula: log(sqrt(prior_var)/sigma) + (sigma^2 + mu^2)/(2*prior_var) - 0.5
        prior_log_sigma = 0.5 * np.log(PRIOR_VAR) # log(sqrt(prior_var))

        # Weight KL
        log_sigma_q_w = np.log(self.weight_sigma + EPSILON)
        kl_weights = (prior_log_sigma - log_sigma_q_w +
                      (self.weight_sigma**2 + self.weight_mu**2) / (2 * PRIOR_VAR) - 0.5)
        kl_div = np.sum(kl_weights)

        # Bias KL
        if self.use_bias:
            log_sigma_q_b = np.log(self.bias_sigma + EPSILON)
            kl_bias = (prior_log_sigma - log_sigma_q_b +
                       (self.bias_sigma**2 + self.bias_mu**2) / (2 * PRIOR_VAR) - 0.5)
            kl_div += np.sum(kl_bias)

        return output, kl_div

    def backward(self, dL_dy):
        """
        Backward pass to compute gradients w.r.t inputs and variational parameters.
        dL_dy is the gradient of the final loss w.r.t the output of this layer.
        """
        batch_size = dL_dy.shape[0]

        # --- Gradients w.r.t sampled weights and biases ---
        # dL_dw = dL_dy^T @ inputs
        dL_dw_sampled = np.dot(dL_dy.T, self.inputs)
        # dL_db = sum(dL_dy, axis=0)
        if self.use_bias:
            dL_db_sampled = np.sum(dL_dy, axis=0)
        else:
             dL_db_sampled = None

        # --- Gradients w.r.t inputs ---
        # dL_dx = dL_dy @ sampled_weight
        dL_dx = np.dot(dL_dy, self.sampled_weight)

        # --- Gradients w.r.t variational parameters (mu, rho) using chain rule ---
        # Weight gradients:
        # dL/dmu = dL/dw * dw/dmu = dL/dw * 1
        self.grad_weight_mu = dL_dw_sampled
        # dL/drho = dL/dw * dw/dsigma * dsigma/drho
        # dw/dsigma = epsilon
        # dsigma/drho = derivative of softplus(rho)
        dsigma_drho_w = softplus_derivative(self.weight_rho)
        self.grad_weight_rho = dL_dw_sampled * self.epsilon_w * dsigma_drho_w

        # Bias gradients (if used):
        if self.use_bias:
            # dL/dmu = dL/db * db/dmu = dL/db * 1
            self.grad_bias_mu = dL_db_sampled
            # dL/drho = dL/db * db/dsigma * dsigma/drho
            dsigma_drho_b = softplus_derivative(self.bias_rho)
            self.grad_bias_rho = dL_db_sampled * self.epsilon_b * dsigma_drho_b

        # --- Gradients from the KL divergence term ---
        # Need derivatives of KL[q||p] w.r.t mu and rho
        # KL = log(sp/sq) + (sq^2 + (mu-mp)^2)/(2*sp^2) - 0.5
        # Here mp = 0, sp^2 = PRIOR_VAR
        # KL = log(sqrt(PRIOR_VAR)) - log(sigma) + (sigma^2 + mu^2)/(2*PRIOR_VAR) - 0.5
        # dKL/dmu = mu / PRIOR_VAR
        grad_kl_weight_mu = self.weight_mu / PRIOR_VAR
        # dKL/dsigma = -1/sigma + sigma/PRIOR_VAR
        # dKL/drho = dKL/dsigma * dsigma/drho
        grad_kl_dsigma_w = -1.0/(self.weight_sigma + EPSILON) + self.weight_sigma / PRIOR_VAR
        grad_kl_weight_rho = grad_kl_dsigma_w * dsigma_drho_w

        self.grad_weight_mu += grad_kl_weight_mu / NUM_BATCHES # Scale KL gradient
        self.grad_weight_rho += grad_kl_weight_rho / NUM_BATCHES

        if self.use_bias:
            grad_kl_bias_mu = self.bias_mu / PRIOR_VAR
            grad_kl_dsigma_b = -1.0/(self.bias_sigma + EPSILON) + self.bias_sigma / PRIOR_VAR
            grad_kl_bias_rho = grad_kl_dsigma_b * dsigma_drho_b

            self.grad_bias_mu += grad_kl_bias_mu / NUM_BATCHES
            self.grad_bias_rho += grad_kl_bias_rho / NUM_BATCHES

        return dL_dx

    def get_params_and_grads(self):
        """Return parameters and their corresponding gradients."""
        params = {'w_mu': self.weight_mu, 'w_rho': self.weight_rho}
        grads = {'w_mu': self.grad_weight_mu, 'w_rho': self.grad_weight_rho}
        if self.use_bias:
            params['b_mu'] = self.bias_mu
            params['b_rho'] = self.bias_rho
            grads['b_mu'] = self.grad_bias_mu
            grads['b_rho'] = self.grad_bias_rho
        return params, grads

    def zero_grad(self):
        """Reset gradients to zero."""
        self.grad_weight_mu.fill(0)
        self.grad_weight_rho.fill(0)
        if self.use_bias:
            self.grad_bias_mu.fill(0)
            self.grad_bias_rho.fill(0)

# --- Bayesian Network (NumPy) ---
class BayesianMLPNP:
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        self.fc1 = BayesianLinearNP(input_dim, hidden_dim1)
        self.fc2 = BayesianLinearNP(hidden_dim1, hidden_dim2)
        self.fc3 = BayesianLinearNP(hidden_dim2, output_dim)
        self.layers = [self.fc1, self.fc2, self.fc3]

        # Store activations during forward pass for backprop
        self.a1 = None # Output of fc1 (before ReLU)
        self.z1 = None # Output of ReLU(a1)
        self.a2 = None # Output of fc2 (before ReLU)
        self.z2 = None # Output of ReLU(a2)
        self.a3 = None # Output of fc3 (logits)

    def forward(self, x):
        total_kl = 0.0

        # Layer 1
        self.a1, kl1 = self.fc1.forward(x)
        self.z1 = relu(self.a1)
        total_kl += kl1

        # Layer 2
        self.a2, kl2 = self.fc2.forward(self.z1)
        self.z2 = relu(self.a2)
        total_kl += kl2

        # Layer 3 (Output)
        self.a3, kl3 = self.fc3.forward(self.z2)
        total_kl += kl3

        # a3 contains the final logits
        return self.a3, total_kl

    def backward(self, dL_dy):
        """Performs backpropagation through the network."""
        # dL_dy is the initial gradient (e.g., from softmax cross-entropy loss)

        # --- Layer 3 ---
        # No activation function after layer 3 in this setup
        dL_dz2 = self.fc3.backward(dL_dy) # Grad w.r.t input of fc3 (z2)

        # --- Layer 2 ---
        # Backprop through ReLU(a2)
        dL_da2 = dL_dz2 * relu_derivative(self.a2)
        # Backprop through fc2
        dL_dz1 = self.fc2.backward(dL_da2) # Grad w.r.t input of fc2 (z1)

        # --- Layer 1 ---
        # Backprop through ReLU(a1)
        dL_da1 = dL_dz1 * relu_derivative(self.a1)
        # Backprop through fc1
        # dL_dx = self.fc1.backward(dL_da1) # Grad w.r.t input of fc1 (x)
        # We don't need dL_dx for parameter updates
        _ = self.fc1.backward(dL_da1)

    def get_params_and_grads(self):
        """Collect parameters and gradients from all layers."""
        all_params = {}
        all_grads = {}
        for i, layer in enumerate(self.layers):
            params, grads = layer.get_params_and_grads()
            for k, v in params.items():
                all_params[f'l{i+1}_{k}'] = v
            for k, v in grads.items():
                all_grads[f'l{i+1}_{k}'] = v
        return all_params, all_grads

    def zero_grad(self):
        """Reset gradients in all layers."""
        for layer in self.layers:
            layer.zero_grad()

# --- Loss Function (NumPy) ---
def calculate_elbo_loss_and_grads(logits, targets, total_kl, num_batches):
    """
    Calculates the negative ELBO loss and the initial gradient dL/dlogits.
    Loss = (KL / num_batches) + NLL
    """
    batch_size = logits.shape[0]

    # 1. NLL (Negative Log Likelihood) using cross-entropy
    # Calculate softmax probabilities
    probs = softmax(logits)
    # Select the probability of the true class for each sample
    true_class_probs = probs[np.arange(batch_size), targets]
    # Calculate log likelihood (add epsilon for stability)
    log_likelihood = np.log(true_class_probs + EPSILON)
    # Calculate average NLL
    nll = -np.mean(log_likelihood)

    # 2. KL Cost
    kl_cost = total_kl / num_batches

    # 3. Total Loss (-ELBO)
    loss = kl_cost + nll

    # 4. Gradient of NLL w.r.t logits (dL/dlogits)
    # dL/dlogits = (softmax(logits) - one_hot_targets) / batch_size
    one_hot_targets = np.zeros_like(logits)
    one_hot_targets[np.arange(batch_size), targets] = 1
    dL_dlogits = (probs - one_hot_targets) / batch_size # Gradient for the NLL part

    # Note: The gradient of the KL term w.r.t parameters is handled inside the layer's backward pass.

    return loss, kl_cost, nll, dL_dlogits

# --- Optimizer (Manual Adam) ---
class AdamOptimizerNP:
    def __init__(self, params, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.params = params # Dictionary of parameters {'name': array}
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {k: np.zeros_like(v) for k, v in params.items()} # 1st moment estimate
        self.v = {k: np.zeros_like(v) for k, v in params.items()} # 2nd moment estimate
        self.t = 0 # Timestep

    def step(self, grads):
        """Update parameters based on gradients."""
        self.t += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.t) / (1.0 - self.beta1**self.t)

        for k in self.params.keys():
            if k not in grads:
                print(f"Warning: Gradient for parameter {k} not found.")
                continue

            g = grads[k]
            # Update biased first moment estimate
            self.m[k] = self.beta1 * self.m[k] + (1.0 - self.beta1) * g
            # Update biased second raw moment estimate
            self.v[k] = self.beta2 * self.v[k] + (1.0 - self.beta2) * (g**2)
            # Compute bias-corrected moment estimates (used implicitly in lr_t)
            # m_hat = self.m[k] / (1.0 - self.beta1**self.t)
            # v_hat = self.v[k] / (1.0 - self.beta2**self.t)

            # Update parameters
            update = self.m[k] / (np.sqrt(self.v[k]) + self.epsilon)
            self.params[k] -= self.lr * update # Original Adam update
            # self.params[k] -= lr_t * update # Alternative with bias correction in lr

# --- Training Loop (NumPy) ---
def train_epoch_np(model, optimizer, train_images, train_labels, num_batches, epoch):
    model.zero_grad() # Ensure grads are zero at the start
    permutation = np.random.permutation(NUM_TRAIN_SAMPLES)
    train_images_shuffled = train_images[permutation]
    train_labels_shuffled = train_labels[permutation]

    total_loss = 0.0
    total_kl_cost = 0.0
    total_nll_cost = 0.0
    correct_predictions = 0

    pbar = tqdm(range(0, NUM_TRAIN_SAMPLES, BATCH_SIZE), desc=f"Epoch {epoch+1}/{EPOCHS} [Training]")

    for i in pbar:
        # Get mini-batch
        batch_indices = range(i, min(i + BATCH_SIZE, NUM_TRAIN_SAMPLES))
        batch_x = train_images_shuffled[batch_indices]
        batch_y = train_labels_shuffled[batch_indices]

        # --- Forward and Loss Calculation (Avg over samples if NUM_SAMPLES_TRAIN > 1) ---
        # For simplicity, using NUM_SAMPLES_TRAIN = 1 here.
        # Implementing averaging would require storing grads per sample and averaging.
        if NUM_SAMPLES_TRAIN > 1:
             print("Warning: NUM_SAMPLES_TRAIN > 1 not fully implemented for NumPy gradient averaging. Using 1.")

        model.zero_grad() # Zero gradients before forward/backward for this batch

        # Forward pass
        logits, total_kl = model.forward(batch_x)

        # Calculate loss and initial gradient dL/dlogits
        loss, kl_cost, nll, dL_dlogits = calculate_elbo_loss_and_grads(
            logits, batch_y, total_kl, num_batches
        )

        # --- Backward Pass ---
        model.backward(dL_dlogits) # This calculates gradients inside each layer

        # --- Optimization Step ---
        params, grads = model.get_params_and_grads()
        optimizer.step(grads) # Update parameters using calculated gradients

        # --- Logging and Accuracy ---
        total_loss += loss * len(batch_indices)
        total_kl_cost += kl_cost * len(batch_indices)
        total_nll_cost += nll * len(batch_indices)

        preds = np.argmax(logits, axis=1)
        correct_predictions += np.sum(preds == batch_y)

        pbar.set_postfix({
            'Loss': f'{loss:.4f}', # Batch loss
            'KL': f'{kl_cost:.4f}',
            'NLL': f'{nll:.4f}',
            'Acc': f'{100. * correct_predictions / (i + len(batch_indices)):.2f}%'
        })

    avg_loss = total_loss / NUM_TRAIN_SAMPLES
    avg_kl = total_kl_cost / NUM_TRAIN_SAMPLES
    avg_nll = total_nll_cost / NUM_TRAIN_SAMPLES
    accuracy = 100. * correct_predictions / NUM_TRAIN_SAMPLES

    print(f"Epoch {epoch+1} Train Summary: Avg Loss: {avg_loss:.4f}, Avg KL: {avg_kl:.4f}, Avg NLL: {avg_nll:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy

# --- Evaluation Function (NumPy) ---
def evaluate_np(model, test_images, test_labels, num_batches, num_samples=NUM_SAMPLES_TEST):
    total_loss = 0.0
    correct_predictions = 0

    pbar = tqdm(range(0, NUM_TEST_SAMPLES, BATCH_SIZE), desc="[Evaluating]")

    for i in pbar:
        batch_indices = range(i, min(i + BATCH_SIZE, NUM_TEST_SAMPLES))
        batch_x = test_images[batch_indices]
        batch_y = test_labels[batch_indices]
        current_batch_size = len(batch_indices)

        # --- Prediction with Multiple Samples ---
        batch_log_probs_mc = []
        batch_kl_mc = 0.0
        for _ in range(num_samples):
            logits, total_kl = model.forward(batch_x)
            log_probs = log_softmax(logits) # Use log_softmax for NLL calculation
            batch_log_probs_mc.append(log_probs)
            batch_kl_mc += total_kl

        # Average KL over samples
        avg_kl = batch_kl_mc / num_samples
        kl_cost = avg_kl / num_batches # Scale KL cost

        # Average log-probabilities using LogSumExp
        # log( (1/N) * sum(exp(log_p_i)) ) = log( sum(exp(log_p_i)) ) - log(N)
        all_log_probs = np.stack(batch_log_probs_mc, axis=0) # Shape: [num_samples, batch_size, num_classes]
        log_sum_exp = np.logaddexp.reduce(all_log_probs, axis=0) # Shape: [batch_size, num_classes]
        avg_log_probs = log_sum_exp - np.log(num_samples)

        # Calculate NLL using the averaged log probabilities
        true_class_log_probs = avg_log_probs[np.arange(current_batch_size), batch_y]
        nll = -np.mean(true_class_log_probs) # Average NLL for the batch

        loss = kl_cost + nll
        total_loss += loss * current_batch_size

        # --- Calculate Accuracy ---
        preds = np.argmax(avg_log_probs, axis=1)
        correct_predictions += np.sum(preds == batch_y)

        pbar.set_postfix({
            'AvgLoss': f'{loss:.4f}',
            'Accuracy': f'{100. * correct_predictions / (i + current_batch_size):.2f}%'
        })

    avg_loss = total_loss / NUM_TEST_SAMPLES
    accuracy = 100. * correct_predictions / NUM_TEST_SAMPLES

    print(f"Test Summary: Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}% ({correct_predictions}/{NUM_TEST_SAMPLES})")
    return avg_loss, accuracy

# --- Main Execution ---
if __name__ == "__main__":
    # Check if data exists before proceeding
    if 'train_images' not in locals():
        print("Data loading failed. Exiting.")
        exit()

    # Define network dimensions
    HIDDEN_DIM1 = 100 # Reduced hidden dims for NumPy speed
    HIDDEN_DIM2 = 50

    print("Initializing NumPy model and optimizer...")
    model = BayesianMLPNP(INPUT_DIM, HIDDEN_DIM1, HIDDEN_DIM2, OUTPUT_DIM)
    params, _ = model.get_params_and_grads() # Get parameter references for optimizer
    optimizer = AdamOptimizerNP(params, learning_rate=LEARNING_RATE)

    print("Starting training (NumPy)...")
    best_test_accuracy = 0.0

    for epoch in range(EPOCHS):
        start_time = time.time()
        train_loss, train_acc = train_epoch_np(model, optimizer, train_images, train_labels, NUM_BATCHES, epoch)
        test_loss, test_acc = evaluate_np(model, test_images, test_labels, NUM_BATCHES)
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} duration: {epoch_time:.2f} seconds")

        # Save model (parameters) if it has the best test accuracy so far
        if test_acc > best_test_accuracy:
            print(f"New best test accuracy: {test_acc:.2f}%. Saving model parameters...")
            # Save parameters using np.savez
            params_to_save, _ = model.get_params_and_grads()
            np.savez(MODEL_SAVE_PATH, **params_to_save)
            best_test_accuracy = test_acc

    print("Training finished.")
    print(f"Best test accuracy achieved: {best_test_accuracy:.2f}%")

    # --- Example: Prediction with Uncertainty (NumPy) ---
    print("\nLoading best model parameters for prediction example...")
    try:
        loaded_params = np.load(MODEL_SAVE_PATH)
        # Manually assign loaded parameters back to the model
        for i, layer in enumerate(model.layers):
             layer.weight_mu = loaded_params[f'l{i+1}_w_mu']
             layer.weight_rho = loaded_params[f'l{i+1}_w_rho']
             if layer.use_bias:
                 layer.bias_mu = loaded_params[f'l{i+1}_b_mu']
                 layer.bias_rho = loaded_params[f'l{i+1}_b_rho']
        print("Model parameters loaded.")
    except FileNotFoundError:
        print(f"Saved model file not found at {MODEL_SAVE_PATH}. Skipping prediction example.")
        exit()


    # Get a single image from the test set
    image = test_images[0:1] # Keep batch dimension (shape [1, 784])
    true_label = test_labels[0]

    # Fashion MNIST Class Labels
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    print(f"\nPredicting for a sample image (True Label: {classes[true_label]})")

    # Predict multiple times using different weight samples
    num_predict_samples = 100
    predictions_mc = []
    for _ in range(num_predict_samples):
        logits, _ = model.forward(image)
        probabilities = softmax(logits) # shape [1, 10]
        predictions_mc.append(probabilities.flatten()) # Store as 1D array

    predictions_mc = np.array(predictions_mc) # Shape: [num_predict_samples, num_classes]

    # Calculate mean prediction
    mean_prediction = np.mean(predictions_mc, axis=0)
    predicted_class_index = np.argmax(mean_prediction)
    predicted_class_prob = np.max(mean_prediction)

    # Calculate predictive entropy
    predictive_entropy = -np.sum(mean_prediction * np.log(mean_prediction + EPSILON))

    # Calculate variance of predictions
    variance_prediction = np.mean(np.var(predictions_mc, axis=0))

    print(f"  Mean Predicted Class: {classes[predicted_class_index]} (Probability: {predicted_class_prob:.4f})")
    print(f"  Predictive Entropy: {predictive_entropy:.4f}")
    print(f"  Mean Prediction Variance: {variance_prediction:.6f}")

    # Plotting (optional, requires matplotlib)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.hist(predictions_mc[:, predicted_class_index], bins=20, alpha=0.7)
        plt.title(f"NumPy - Distribution of Probabilities for Predicted Class '{classes[predicted_class_index]}'")
        plt.xlabel("Probability")
        plt.ylabel("Frequency (out of 100 samples)")
        plt.grid(True)
        plt.show()
    except ImportError:
        print("\nMatplotlib not found. Skipping histogram plot.")
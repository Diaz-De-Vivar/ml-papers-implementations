import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import jax.numpy as jnp # For HMC gradients
from jax import grad

# Set random seed for reproducibility
np.random.seed(42)

# 1. Load Data (Old Faithful)
# Using a readily available version online
try:
    data = pd.read_csv('https://raw.githubusercontent.com/stantinor/datasets/master/old_faithful.csv')
    # Use standardized versions for better numerical stability
    data['eruptions_std'] = (data['eruptions'] - data['eruptions'].mean()) / data['eruptions'].std()
    data['waiting_std'] = (data['waiting'] - data['waiting'].mean()) / data['waiting'].std()
    eruptions = data['eruptions_std'].values
    waiting = data['waiting_std'].values
    print("Data loaded successfully.")
except Exception as e:
    print(f"Failed to load data: {e}")
    print("Please download 'old_faithful.csv' manually if needed.")
    # Example: provide link or instructions
    # As a fallback, create dummy data for demonstration
    print("Using dummy data for demonstration.")
    eruptions = np.random.rand(100) * 3 + 1.5 # Dummy eruption durations
    waiting = 50 + 10 * eruptions + np.random.randn(100) * 5 # Dummy waiting times
    # Standardize dummy data
    eruptions = (eruptions - eruptions.mean()) / eruptions.std()
    waiting = (waiting - waiting.mean()) / waiting.std()


# 2. Define the Log-Posterior Function
# We work with log(sigma) internally for easier sampling (unconstrained)
# but define priors and likelihood in terms of sigma.

def log_likelihood(params, x, y):
    """Calculates log-likelihood of data given parameters."""
    alpha, beta, log_sigma = params
    sigma = jnp.exp(log_sigma)
    mu = alpha + beta * x
    # Sum of log probabilities of Normal distribution
    ll = jnp.sum(stats.norm.logpdf(y, loc=mu, scale=sigma))
    return ll

def log_prior(params):
    """Calculates log-prior probability of parameters."""
    alpha, beta, log_sigma = params
    sigma = jnp.exp(log_sigma) # Transform back to sigma scale

    # Prior for alpha: Normal(0, 20) -> variance = 400
    log_prior_alpha = stats.norm.logpdf(alpha, loc=0, scale=20)
    # Prior for beta: Normal(0, 10) -> variance = 100
    log_prior_beta = stats.norm.logpdf(beta, loc=0, scale=10)
    # Prior for sigma: HalfCauchy(0, 5)
    # logpdf(sigma | loc=0, scale=5) = log(2 / (pi * scale * (1 + (x/scale)^2))) for x > 0
    if sigma <= 0:
        return -jnp.inf # Log probability is -infinity if sigma is not positive
    log_prior_sigma = jnp.log(2) - jnp.log(jnp.pi * 5 * (1 + (sigma / 5)**2))

    return log_prior_alpha + log_prior_beta + log_prior_sigma

# Use JAX numpy for compatibility with automatic differentiation if needed later
def log_posterior(params, x, y):
    """Calculates log-posterior (unnormalized)."""
    lp = log_prior(params)
    # Avoid calculating likelihood if prior is impossible
    if lp == -jnp.inf:
        return -jnp.inf
    ll = log_likelihood(params, x, y)
    return lp + ll

# --- Visualization Helper ---
def plot_results(samples, param_names=['alpha', 'beta', 'sigma']):
    """Plots traceplots and histograms of MCMC samples."""
    n_params = samples.shape[1]
    fig, axes = plt.subplots(n_params, 2, figsize=(12, 3 * n_params))

    # Transform log_sigma back to sigma for plotting
    plot_samples = samples.copy()
    if 'log_sigma' in param_names:
        sigma_idx = param_names.index('log_sigma')
        param_names[sigma_idx] = 'sigma' # Rename for plot
        plot_samples[:, sigma_idx] = np.exp(plot_samples[:, sigma_idx])

    for i in range(n_params):
        ax_trace = axes[i, 0]
        ax_hist = axes[i, 1]

        # Trace plot
        ax_trace.plot(plot_samples[:, i], alpha=0.7)
        ax_trace.set_ylabel(param_names[i])
        ax_trace.set_title(f'Trace Plot - {param_names[i]}')

        # Histogram
        sns.histplot(plot_samples[:, i], kde=True, ax=ax_hist)
        ax_hist.set_title(f'Posterior Histogram - {param_names[i]}')

    axes[-1, 0].set_xlabel('Iteration')
    axes[-1, 1].set_xlabel('Parameter Value')
    fig.tight_layout()
    plt.show()


# --- Slice Sampling Implementation ---

def slice_sampler_1d(log_target_pdf, x_current, w, args=()):
    """Performs one step of univariate slice sampling."""
    # 1. Sample vertical level u
    log_fx_current = log_target_pdf(x_current, *args)
    log_u = log_fx_current - np.random.exponential(1.0) # Sample log(u) = log(f(x)) - Exp(1)

    # 2. Find the interval [L, R] around x_current ("stepping out")
    # Randomly place the initial interval of width w around x_current
    r = np.random.uniform(0, w)
    L = x_current - r
    R = x_current + (w - r)

    # Expand interval until L and R are outside the slice
    while log_target_pdf(L, *args) > log_u:
        L -= w
    while log_target_pdf(R, *args) > log_u:
        R += w

    # 3. Sample x_new from the interval [L, R] ("shrinking in")
    while True:
        x_new = np.random.uniform(L, R)
        log_fx_new = log_target_pdf(x_new, *args)

        if log_fx_new > log_u:
            # Accept the sample
            return x_new
        else:
            # Shrink the interval
            if x_new < x_current:
                L = x_new
            else:
                R = x_new


def slice_sampler(log_posterior_fn, initial_params, n_iter, w, data_x, data_y):
    """
    Performs Slice Sampling by updating each parameter dimension one by one.

    Args:
        log_posterior_fn: Function computing log posterior (takes full param vector).
        initial_params: Starting parameter values [alpha, beta, log_sigma].
        n_iter: Number of MCMC iterations.
        w: Initial width for the slice interval (scalar or vector). Needs tuning.
        data_x: Predictor variable data.
        data_y: Response variable data.

    Returns:
        numpy array of samples.
    """
    print("\nRunning Slice Sampling...")
    current_params = np.array(initial_params)
    n_params = len(current_params)
    samples = np.zeros((n_iter, n_params))

    if isinstance(w, (int, float)):
        w = np.full(n_params, w) # Use same width for all params if scalar provided

    for i in range(n_iter):
        # Iterate through each parameter dimension
        for j in range(n_params):
            # Define a function for the log posterior varying only the j-th param
            def log_posterior_1d(param_j, current_params_frozen, j_index, x, y):
                params_temp = current_params_frozen.copy()
                params_temp[j_index] = param_j
                return log_posterior_fn(params_temp, x, y)

            # Perform 1D slice sampling for the j-th parameter
            current_params[j] = slice_sampler_1d(
                log_posterior_1d,
                current_params[j],
                w[j],
                args=(current_params, j, data_x, data_y) # Pass necessary fixed args
            )

        # Store the updated parameter vector
        samples[i] = current_params

        if (i + 1) % (n_iter // 10) == 0:
            print(f"  Iteration {i+1}/{n_iter} complete.")

    print("Slice Sampling finished.")
    return samples

# --- Run Slice Sampler ---
n_iterations_slice = 20000
burn_in_slice = 5000
initial_guess_slice = [0.0, 0.0, np.log(np.std(waiting))]
# Width parameter 'w' - requires tuning! Larger w explores more, smaller is faster locally.
slice_widths = [0.5, 0.5, 0.5] # Needs tuning!

slice_samples = slice_sampler(
    log_posterior, initial_guess_slice, n_iterations_slice, slice_widths, eruptions, waiting
)

# Discard burn-in and plot
slice_samples_burned = slice_samples[burn_in_slice:]
plot_results(slice_samples_burned, param_names=['alpha', 'beta', 'log_sigma'])
# Note: Slice sampling also has an implicit acceptance rate of 100% for valid moves.
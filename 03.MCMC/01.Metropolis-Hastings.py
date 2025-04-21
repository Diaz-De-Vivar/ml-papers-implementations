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

# --- Metropolis-Hastings Implementation ---

def metropolis_hastings(log_posterior_fn, initial_params, n_iter, proposal_sd, data_x, data_y):
    """
    Performs Metropolis-Hastings sampling.

    Args:
        log_posterior_fn: Function that computes log posterior. Accepts params, x, y.
        initial_params: Starting parameter values [alpha, beta, log_sigma].
        n_iter: Number of MCMC iterations.
        proposal_sd: Standard deviation for the Normal proposal distribution (scalar or vector).
        data_x: Predictor variable data.
        data_y: Response variable data.

    Returns:
        Tuple: (samples array, acceptance rate)
    """
    print("Running Metropolis-Hastings...")
    current_params = np.array(initial_params)
    n_params = len(current_params)
    samples = np.zeros((n_iter, n_params))
    log_post_current = log_posterior_fn(current_params, data_x, data_y)

    accepted_count = 0

    for i in range(n_iter):
        # 1. Propose a new state
        # Using a symmetric proposal (Multivariate Normal centered at current state)
        proposal = np.random.normal(loc=current_params, scale=proposal_sd, size=n_params)

        # 2. Calculate log posterior of the proposed state
        log_post_proposal = log_posterior_fn(proposal, data_x, data_y)

        # 3. Calculate acceptance probability (log scale for numerical stability)
        # For symmetric proposal q(x'|x) = q(x|x'), the Hastings ratio is 1 (0 in log scale)
        log_acceptance_ratio = log_post_proposal - log_post_current

        # Ensure the ratio calculation handles -inf correctly
        if np.isneginf(log_post_proposal) and np.isneginf(log_post_current):
             acceptance_prob = 0.0 # Avoid NaN if both are -inf
        elif np.isneginf(log_post_current):
             acceptance_prob = 1.0 # Always accept if moving from impossible state
        else:
             acceptance_prob = np.exp(min(0, log_acceptance_ratio)) # min(1, exp(log_ratio))

        # 4. Accept or reject
        u = np.random.rand()
        if u < acceptance_prob:
            # Accept the proposal
            current_params = proposal
            log_post_current = log_post_proposal
            accepted_count += 1

        # 5. Store the current state (either new or old)
        samples[i] = current_params

        if (i + 1) % (n_iter // 10) == 0:
            print(f"  Iteration {i+1}/{n_iter} complete.")

    acceptance_rate = accepted_count / n_iter
    print(f"Metropolis-Hastings finished. Acceptance rate: {acceptance_rate:.3f}")
    return samples, acceptance_rate

# --- Run MH ---
n_iterations_mh = 20000
burn_in_mh = 5000
# Initial guess (can be arbitrary, but closer helps convergence)
# Start log_sigma near log(std(waiting)) for a reasonable scale
initial_guess_mh = [0.0, 0.0, np.log(np.std(waiting))]
# Proposal SD - THIS IS CRUCIAL AND REQUIRES TUNING!
# Start with small values and adjust based on acceptance rate (aim for ~0.234 for multivariate Normal)
proposal_std_mh = [0.05, 0.05, 0.05] # Needs tuning!

mh_samples, mh_acceptance_rate = metropolis_hastings(
    log_posterior, initial_guess_mh, n_iterations_mh, proposal_std_mh, eruptions, waiting
)

# Discard burn-in and plot
mh_samples_burned = mh_samples[burn_in_mh:]
plot_results(mh_samples_burned, param_names=['alpha', 'beta', 'log_sigma'])
print(f"MH Acceptance Rate: {mh_acceptance_rate:.3f} (target ~0.2-0.4)")
# If acceptance rate is too high, increase proposal_sd. If too low, decrease proposal_sd.
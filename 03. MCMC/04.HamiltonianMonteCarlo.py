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



# --- Hamiltonian Monte Carlo (HMC) Implementation ---
# Requires JAX for automatic differentiation

# Define the gradient function using JAX
# Ensure log_posterior uses jax.numpy (jnp) internally
grad_log_posterior = grad(log_posterior, argnums=0) # Gradient w.r.t. first arg (params)

def leapfrog(q, p, grad_log_post_fn, epsilon, L, data_x, data_y):
    """
    Performs L leapfrog steps for HMC.

    Args:
        q: Current position (parameters).
        p: Current momentum.
        grad_log_post_fn: Function computing gradient of log posterior.
        epsilon: Step size.
        L: Number of leapfrog steps.
        data_x, data_y: Data.

    Returns:
        Tuple: (new_q, new_p)
    """
    q_new = q.copy()
    p_new = p.copy()

    # Half step for momentum at the beginning
    p_new += epsilon / 2.0 * grad_log_post_fn(q_new, data_x, data_y)

    # Alternate full steps for position and momentum
    for _ in range(L - 1):
        # Full step for position
        q_new += epsilon * p_new # Assuming mass matrix M=I
        # Full step for momentum
        p_new += epsilon * grad_log_post_fn(q_new, data_x, data_y)

    # Full step for position
    q_new += epsilon * p_new # Assuming mass matrix M=I
    # Half step for momentum at the end
    p_new += epsilon / 2.0 * grad_log_post_fn(q_new, data_x, data_y)

    # Negate momentum at end of trajectory to make proposal symmetric
    # Not strictly required for basic HMC acceptance, but good practice
    p_new = -p_new

    return q_new, p_new

def hamiltonian_monte_carlo(log_posterior_fn, grad_log_post_fn, initial_params,
                           n_iter, epsilon, L, data_x, data_y):
    """
    Performs Hamiltonian Monte Carlo sampling.

    Args:
        log_posterior_fn: Function computing log posterior.
        grad_log_post_fn: Function computing gradient of log posterior.
        initial_params: Starting parameter values [alpha, beta, log_sigma].
        n_iter: Number of MCMC iterations.
        epsilon: Leapfrog step size. Needs tuning.
        L: Number of leapfrog steps. Needs tuning.
        data_x: Predictor variable data.
        data_y: Response variable data.

    Returns:
        Tuple: (samples array, acceptance rate)
    """
    print("\nRunning Hamiltonian Monte Carlo (HMC)...")
    # Use JAX arrays internally for compatibility with gradient function
    current_q = jnp.array(initial_params)
    n_params = len(current_q)
    samples = np.zeros((n_iter, n_params)) # Store samples as numpy arrays

    accepted_count = 0

    for i in range(n_iter):
        # 1. Sample momentum (p) from a Normal distribution (usually N(0, I))
        current_p = jnp.array(np.random.normal(size=n_params))

        # 2. Simulate Hamiltonian dynamics using leapfrog integrator
        q_proposal, p_proposal = leapfrog(current_q, current_p, grad_log_post_fn,
                                          epsilon, L, data_x, data_y)

        # 3. Calculate Metropolis-Hastings acceptance probability
        # Evaluate potential energy (U = -log_posterior) and kinetic energy (K = 0.5 * p^T * M^-1 * p)
        # Assuming mass matrix M = I (identity)
        current_U = -log_posterior_fn(current_q, data_x, data_y)
        current_K = 0.5 * jnp.sum(current_p**2)
        proposal_U = -log_posterior_fn(q_proposal, data_x, data_y)
        proposal_K = 0.5 * jnp.sum(p_proposal**2)

        # Calculate acceptance probability (on log scale)
        log_acceptance_ratio = (current_U + current_K) - (proposal_U + proposal_K)

        # Handle infinities
        current_H = current_U + current_K
        proposal_H = proposal_U + proposal_K
        if jnp.isinf(proposal_H) and jnp.isinf(current_H):
            acceptance_prob = 0.0
        elif jnp.isinf(current_H) and not jnp.isinf(proposal_H):
             acceptance_prob = 1.0 # Always accept if moving from impossible state
        elif jnp.isinf(proposal_H):
             acceptance_prob = 0.0 # Never accept proposal to impossible state
        else:
            acceptance_prob = jnp.exp(min(0.0, log_acceptance_ratio))

        # 4. Accept or reject
        u = np.random.rand()
        if u < acceptance_prob:
            current_q = q_proposal
            accepted_count += 1
        # Else: current_q remains the same (reject proposal)

        # Store the current position (parameters)
        samples[i] = np.array(current_q) # Convert back to numpy for storage

        if (i + 1) % (n_iter // 10) == 0:
            print(f"  Iteration {i+1}/{n_iter} complete.")

    acceptance_rate = accepted_count / n_iter
    print(f"HMC finished. Acceptance rate: {acceptance_rate:.3f}")
    return samples, acceptance_rate

# --- Run HMC ---
n_iterations_hmc = 10000 # HMC often needs fewer iterations than MH due to efficiency
burn_in_hmc = 2000
initial_guess_hmc = [0.0, 0.0, np.log(np.std(waiting))]

# HMC tuning parameters - CRITICAL!
# Epsilon (step size): Too large -> low acceptance; too small -> slow exploration.
# L (number of steps): Too small -> random walk behavior; too large -> wasted computation.
hmc_epsilon = 0.05 # Needs tuning!
hmc_L = 20         # Needs tuning!

# Ensure data is in JAX arrays for grad function
eruptions_jnp = jnp.array(eruptions)
waiting_jnp = jnp.array(waiting)

hmc_samples, hmc_acceptance_rate = hamiltonian_monte_carlo(
    log_posterior, grad_log_posterior, initial_guess_hmc, n_iterations_hmc,
    hmc_epsilon, hmc_L, eruptions_jnp, waiting_jnp
)

# Discard burn-in and plot
hmc_samples_burned = hmc_samples[burn_in_hmc:]
plot_results(hmc_samples_burned, param_names=['alpha', 'beta', 'log_sigma'])
print(f"HMC Acceptance Rate: {hmc_acceptance_rate:.3f} (target often ~0.6-0.9)")

# --- NUTS Explanation ---
print("\n--- About NUTS ---")
print("The No-U-Turn Sampler (NUTS) is an extension of HMC.")
print("Key Idea: Instead of fixing the number of leapfrog steps (L), NUTS adaptively chooses L.")
print("It builds a trajectory by simulating forward and backward in time using leapfrog steps.")
print("It stops automatically when the trajectory starts to make a 'U-turn' (points back towards the start),")
print("preventing inefficient exploration or retracing steps.")
print("This eliminates the need to manually tune L, making it more robust and often more efficient.")
print("Implementing NUTS from scratch is significantly more complex due to the tree-building and U-turn check.")
print("Libraries like Stan, PyMC, NumPyro provide robust NUTS implementations.")
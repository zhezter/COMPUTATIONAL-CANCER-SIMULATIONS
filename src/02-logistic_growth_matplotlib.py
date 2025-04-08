import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors

# Model Parameters
K = 1.0  # Carrying capacity (maximum sustainable population)
a = 0.1  # Growth rate (adjusted for visualization)
initial_conditions = np.linspace(
    0.1, 2.0, 35
)  # Different initial conditions from 0.1 to 2.0


def x_t(t, K, K_prime, a):
    """
    Calculates the population size using the logistic growth model.

    Parameters:
    -----------
    >>> t : float or numpy.ndarray
            Time points at which to evaluate the function
        K : float
            Carrying capacity - maximum sustainable population
        K_prime : float
            Integration constant determined by initial conditions
        a : float
            Growth rate - speed at which population approaches K

    Returns:
    --------
    >>> float or numpy.ndarray
            Population size at time t
    """
    return (K_prime * K * np.exp(a * t)) / (1 + K_prime * np.exp(a * t))


# Time array setup
t = np.linspace(0, 100, 500)  # Time points from 0 to 100 with 500 steps

# Plot setup and styling
fig, ax = plt.subplots(figsize=(10, 6), facecolor="black")
ax.set_xlim(0, 100)
ax.set_ylim(0, 2)
ax.set_facecolor("black")
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_color("white")

# Color setup for multiple curves
colors = plt.cm.rainbow(np.linspace(0, 1, len(initial_conditions)))

# Initialize lines for each initial condition
lines = []
for color in colors:
    (line,) = ax.plot([], [], lw=2, color=color)
    lines.append(line)

# Add carrying capacity line
ax.axhline(y=K, color="orange", linestyle="--", label="K")

# Labels and titles
ax.set_xlabel("Time (t)", color="white")
ax.set_ylabel("Population x(t)", color="white")
ax.set_title(
    "Logistic Growth Model for Different Initial Tumor Cell Populations", color="white"
)
ax.legend(loc="upper right", frameon=False, fontsize=10)
ax.grid(color="gray", linestyle="--", linewidth=0.5)


def init():
    """
    Initialize the animation by setting empty data for all lines.
    Returns the list of lines to be animated.
    """
    for line in lines:
        line.set_data([], [])
    return lines


def update(frame):
    """
    Update function for animation.

    Parameters:
    -----------
    frame : int
        Current frame number in the animation

    Returns:
    --------
    list
        Updated list of lines with new data
    """
    for i, x0 in enumerate(initial_conditions):
        K_prime = x0 / (K - x0)
        x_values = x_t(t[:frame], K, K_prime, a)
        lines[i].set_data(t[:frame], x_values)

    # Reset animation if all curves reach carrying capacity
    if np.all([x_t(t[frame], K, x0 / (K - x0), a) >= K for x0 in initial_conditions]):
        init()
    return lines


# Create and display animation
ani = animation.FuncAnimation(
    fig, update, frames=len(t), init_func=init, blit=True, interval=20, repeat=True
)

plt.show()

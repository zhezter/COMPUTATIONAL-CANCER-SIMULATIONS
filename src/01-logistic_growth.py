from manim import *
import numpy as np
import itertools as it
import matplotlib as plt


class LogisticGrowth(Scene):
    """
    A Manim scene that visualizes the logistic growth model.

    This animation shows multiple solutions to the logistic differential equation
    with different initial conditions. The logistic model is commonly used to
    describe population growth with limited resources.
    """

    def __init__(
        self, 
        K: float = 1.0, 
        a: float = 0.5, 
        initial_conditions: np.ndarray = np.linspace(0.1, 2.0, 35)
    ) -> None:
        """
        Initializes the LogisticGrowth scene.

        Parameters:
        -----------
        K : float
            Carrying capacity of the environment.
        a : float
            Growth rate of the population.
        initial_conditions : np.ndarray
            Array of initial conditions for the population.
        """
        super().__init__()
        self.K = K
        self.a = a
        self.initial_conditions = initial_conditions

    def construct(self):
        # Initialize model parameters
        self.K: float = 1.0  # Carrying capacity (maximum sustainable population)
        self.a: float = 0.5  # Growth rate (speed at which the population approaches K)
        self.initial_conditions: np.ndarray = np.linspace(0.1, 2.0, 35)

        def x_t(t: float, K: float, K_prime, a) -> float:
            """
            Calculates the population size at time `t` using the logistic growth model.

            Parameters:
            -----------
            t : float
                Time point
            K : float
                Carrying capacity of the environment
            K_prime : float
                Integration constant determined by initial conditions
            a : float
                Growth rate of the population

            Returns:
            --------
            float
                Population size at time t
                Formula: x(t) = (K_prime * K * e^(at)) / (1 + K_prime * e^(at))
            """
            return (K_prime * K * np.exp(a * t)) / (1 + K_prime * np.exp(a * t))

        # Set up coordinate system
        axes = Axes(
            x_range=[0, 15, 1],
            y_range=[0, 2.3, 0.2],
            x_axis_config={
                "include_numbers": False,
                "label_direction": DOWN,
                "include_tip": True,
            },
            y_axis_config={
                "include_numbers": False,
                "label_direction": LEFT,
                "include_tip": True,
            },
        ).scale(0.8)  # Scale to 80% to fit screen better

        # Add axis labels
        axes_labels: VGroup = axes.get_axis_labels(x_label="t", y_label="x(t)")

        # Create dashed line at carrying capacity K
        K_line: DashedLine = DashedLine(
            start=axes.c2p(0, self.K), end=axes.c2p(15, self.K), color=RED
        ).set_z_index(2)  # Set z-index to ensure line appears above curves

        # Generate rainbow colors for different curves
        colors: it.cycle = it.cycle(plt.cm.rainbow(np.linspace(0, 1, len(self.initial_conditions))))

        # Create curves for each initial condition
        curves: list = []
        for x0 in self.initial_conditions:
            K_prime: float = x0 / (self.K - x0)  # Calculate integration constant
            curve: ParametricFunction = axes.plot(
                lambda t: x_t(t, self.K, K_prime, self.a), x_range=[0, 15], color=next(colors)
            )
            curves.append(curve)

        # Animate the scene
        # 1. Create coordinate system
        self.play(Create(axes, run_time=2))
        self.wait()

        # 2. Add axis labels
        self.play(Create(axes_labels))
        self.wait()

        # 3. Show carrying capacity line
        self.play(Create(K_line), run_time=0.7)
        self.wait(2)

        # 4. Animate each solution curve
        for curve in curves:
            self.play(Create(curve), run_time=0.6)
        self.wait(2)

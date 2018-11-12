import numpy as np
import random
import copy

class OUNoise:
    """Ornstein-Uhlenbeck process.
    Introduce temporally correlated exploration https://arxiv.org/pdf/1509.02971.pdf.
    This is expected to help getting to the first reward so learning can then take place
    Using parameters mu=0, theta=.15 and sigma=0.2 it took 1029 time steps to get a non-zero score
    """

    def __init__(self, size, seed, state_size, mu=0., theta=0.15, sigma=0.2, ):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.state_size = state_size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
import numpy as np
import math
import torch
class ShortTermalpha:

    def __init__(self, zeta=0.5, epsilon=0.002, eta=0.001):
        self.zeta = zeta
        self.epsilon = epsilon
        self.eta = eta
        self.dalpha = None
        self.value = None
    
    def __str__(self):
        return f"ShortTermalpha(zeta={self.zeta}, epsilon={self.epsilon}, eta={self.eta})"

    def generate_dalpha(self, dt, lambda_p, lambda_m):
        self.dalpha = math.sqrt(
            (self.eta**2 + self.epsilon**2 * (lambda_p + lambda_m)) * 3 * dt
        )

    def generate_alpha(self, alpha_threshold, decimal_places=4):
        if self.dalpha is not None:
            max_alpha = math.ceil(alpha_threshold / self.dalpha) * self.dalpha
            min_alpha = -max_alpha
            self.value = np.round(
                np.arange(min_alpha, max_alpha + self.dalpha, self.dalpha),
                decimal_places,
            )

    def get_next_alpha(self, alpha_current, isMO, buySellMO, dt):
        """
        Computes the next alpha value(s) given the current state and parameters.

        Parameters
        ----------
        alpha_current : np.ndarray
            Current alpha values, shape (Nsims,)
        isMO : np.ndarray
            Market order indicator, shape (Nsims,)
        buySellMO : np.ndarray
            Buy/sell market order direction, shape (Nsims,)
        dt : float
            Time step size.

        Returns
        -------
        np.ndarray
            Next alpha values, shape (Nsims,)
        """
        Nsims = alpha_current.shape[0]
        noise = self.eta * torch.sqrt(torch.tensor(dt)) * torch.randn(Nsims)
        mo_impact = self.epsilon * isMO * buySellMO
        decay = torch.exp(torch.tensor(-self.zeta * dt)) * alpha_current
        return decay + noise + mo_impact
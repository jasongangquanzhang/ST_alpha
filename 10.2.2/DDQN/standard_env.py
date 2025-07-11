import numpy as np
import torch


# S_t = S_0 + \sigma * W_t

class StandardEnv:
    def __init__(
        self,
        T = 300,                        # Total time
        I_max = 20,                     # Max inventory
        lambd = 50 / 300,               # Order arrival rate (for both buy and sell MOs)
        Delta = 0.01,                   # Spread
        phi = 0.01,                     # Running penalty
        varphi = 0.01,                  # Liquidation penalty TODO: experiment
        sigma = 0.001,                  # Price volatility
        S_0 = 20,                       # Initial midprice
        X_0 = 0,                        # Initial cash
        q_0 = 0,                        # Initial inventory
    ):
        self.T = T
        self.lambda_p = self.lambda_m = lambd
        self.Ndt = int((self.lambda_p + self.lambda_m) * T * 5)  # Number of discrete time steps
        self.dt = T / self.Ndt  # Time step size

        self.t = torch.linspace(0, T, self.Ndt + 1)    # Time vector

        self.I_max = I_max
        self.Delta = Delta
        self.phi = phi
        self.varphi = varphi
        self.sigma = sigma

        # TODO is this needed?
        self.S_0 = S_0
        self.X_0 = X_0
        self.q_0 = q_0
    
    def Randomize_Start(self, mini_batch_size):
        """Randomize the initial state of the environment."""
        S0 = self.S_0 + self.sigma * torch.randn(mini_batch_size)   # Initial midprice with noise
        q0 = torch.randint(-self.I_max, self.I_max + 1, size=(mini_batch_size,)).float()  # Random inventory
        X0 = torch.zeros(mini_batch_size)  # Initial cash
        return  S0, q0, X0 # TODO check if this is correct, need to include t?
    
    def step(self, t, S, X, q, action):
        """Advance the environment by one step given the current action.
        
        Args:
            t (torch.Tensor): Current time of shape (batch_size,).
            S (torch.Tensor): Current midprice of shape (batch_size,).
            X (torch.Tensor): Current cash of shape (batch_size,).
            q (torch.Tensor): Current inventory of shape (batch_size,).
            action (torch.Tensor): Action taken in the current state of shape (batch_size,).
                0: Do nothing
                1: Buy 1 unit
                2: Sell 1 unit
                3: Buy and sell 1 unit
        """
        mini_batch_size = S.shape[0]

        # Simulate arriving MOs TODO check if only either buy or sell MO arrives
        lambda_total = self.lambda_p + self.lambda_m
        isMO = torch.rand(mini_batch_size) < (
            1 - torch.exp(torch.tensor(-self.dt * (lambda_total)))
        )
        buySellMO = (
            2 * (torch.rand(mini_batch_size) < self.lambda_p / (lambda_total)) - 1
        ) # -1: sell MO, 1: buy MO

        # Time update
        t_p = t + self.dt

        # Price update
        # S_t = S_0 + \sigma * W_t
        # dS_t = \sigma * sqrt(dt) * N(0, 1)
        S_p = S + self.sigma * torch.sqrt(torch.tensor(self.dt)) * torch.randn(mini_batch_size)

        # Cash update
        isfilled_p = isMO.int() * (buySellMO == 1).int() * ((action == 2) | (action == 3)).int()   # Sell LO filled
        isfilled_m = isMO.int() * (buySellMO == -1).int() * ((action == 1) | (action == 3)).int()  # Buy LO filled
        X_p = (
            X
            - (S - 0.5 * self.Delta) * isfilled_m  # Cash spent on buying at bid price
            + (S + 0.5 * self.Delta) * isfilled_p  # Cash received from selling at ask price
        )

        # Inventory update
        q_p = q + isfilled_m - isfilled_p

        # Reward calculationo
        reward = (isfilled_p + isfilled_m) * self.Delta * 0.5 \
            + q * (S_p - S) \
            - self.phi * (q**2) * self.dt \
            - self.varphi * (q_p**2 - q**2)
        
        return t_p, S_p, X_p, q_p, reward, isMO, buySellMO

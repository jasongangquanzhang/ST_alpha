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
        phi = 0.000001,                 # Running penalty
        varphi = 0.001,                 # Liquidation penalty
        sigma = 0.001,                  # Price volatility
        gamma = 0.99,                   # Discount factor
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
        self.gamma = gamma

        self.S_0 = S_0
        self.X_0 = X_0
        self.q_0 = q_0
    
    def Randomize_Start(self, mini_batch_size):
        """Randomize the initial state of the environment.
        
        Args:
            mini_batch_size (int): Number of samples to randomize.
        Returns:
            tuple: A tuple containing:
                - t0 (torch.Tensor): Initial time of shape (mini_batch_size,).
                - S0 (torch.Tensor): Initial midprice of shape (mini_batch_size,).
                - q0 (torch.Tensor): Initial inventory of shape (mini_batch_size,).
                - X0 (torch.Tensor): Initial cash of shape (mini_batch_size,).
        """
        t0 = torch.zeros(mini_batch_size)  # Initial time (for linear setting)
        S0 = self.S_0 + self.sigma * torch.randn(mini_batch_size)   # Initial midprice with noise
        X0 = torch.zeros(mini_batch_size)  # Initial cash
        q0 = torch.randint(-self.I_max, self.I_max + 1, size=(mini_batch_size,))  # Random inventory
        return  t0, S0, X0, q0
    
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
        Returns:
            tuple: A tuple containing:
                - t_p (torch.Tensor): Updated time of shape (batch_size,).
                - S_p (torch.Tensor): Updated midprice of shape (batch_size,).
                - X_p (torch.Tensor): Updated cash of shape (batch_size,).
                - q_p (torch.Tensor): Updated inventory of shape (batch_size,).
                - reward (torch.Tensor): Reward for the action taken of shape (batch_size,).
                - isMO (torch.Tensor): Indicator of whether a market order arrived of shape (batch_size,).
                - buySellMO (torch.Tensor): Type of market order (-1 for sell, 1 for buy) of shape (batch_size,).
        """
        mini_batch_size = S.shape[0]

        # Simulate arriving MOs
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

        # Reward calculation
        # Telescoping method
        # # Earning spread + change in position + running penalty + liquidation penalty
        # discount = self.gamma ** ((self.T - t)/self.dt)
        # reward = (
        #     (isfilled_p + isfilled_m) * self.Delta * 0.5            # Earning spread
        #     + q * (S_p - S)                                         # Change in position    
        #     - self.phi * (q**2) * self.dt                           # Running penalty
        #     - discount * self.varphi * (q_p**2 - q**2)              # Liquidation penalty
        # )

        is_terminal = t_p >= self.T
        reward = torch.zeros(mini_batch_size)
        # Terminal rewards (for samples where t_p >= T)
        reward[is_terminal] = (
            X_p[is_terminal]
            + q_p[is_terminal] * (S_p[is_terminal] - (self.Delta / 2 + self.varphi * q_p[is_terminal]))
            - self.phi * (q_p[is_terminal]**2) * self.dt
        )
        # Non-terminal rewards (for samples where t_p < T)
        reward[~is_terminal] = - self.phi * (q_p[~is_terminal]**2) * self.dt

        return t_p, S_p, X_p, q_p, reward, isMO, buySellMO

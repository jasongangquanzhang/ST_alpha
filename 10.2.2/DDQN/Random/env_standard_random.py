import numpy as np
import torch
import torch.nn.functional as F


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
    
    def to_one_hot(self, action, num_actions):
        """
        Convert action(s) to one-hot tensor.
        
        Args:
            action (int, Tensor): Single int, tensor of ints, or one-hot tensor.
            num_actions (int): Total number of possible actions.
        
        Returns:
            Tensor: One-hot encoded tensor of shape (batch_size, num_actions).
        """
        if isinstance(action, torch.Tensor) and action.ndim == 2:
            return action.float()  # already one-hot

        if isinstance(action, int):
            action = torch.tensor([action], dtype=torch.long)

        if isinstance(action, torch.Tensor) and action.ndim == 1:
            return F.one_hot(action.long(), num_classes=num_actions).float()

        raise ValueError("Unsupported action format. Must be int, 1D tensor of ints, or 2D one-hot tensor.")

    def Randomize_Start(self, mini_batch_size):
        """Randomize the initial state of the environment."""
        S0 = self.S_0 + self.sigma * torch.randn(mini_batch_size)   # Initial midprice with noise
        q0 = torch.randint(-self.I_max, self.I_max + 1, size=(mini_batch_size,)).float()  # Random inventory
        X0 = torch.zeros(mini_batch_size)  # Initial cash
        return  S0, q0, X0
    
    def step(self, t, S, X, q, action_onehot):
        """Advance the environment by one step given the current action.
        
        Args:
            t (torch.Tensor): Current time of shape (batch_size,).
            S (torch.Tensor): Current midprice of shape (batch_size,).
            X (torch.Tensor): Current cash of shape (batch_size,).
            q (torch.Tensor): Current inventory of shape (batch_size,).
            action_onehot (torch.Tensor): One-hot encoded action of shape (batch_size, action_size).
                For each row:
                - [1, 0, 0, 0] represents "do nothing"
                - [0, 1, 0, 0] represents "buy 1 unit"
                - [0, 0, 1, 0] represents "sell 1 unit"
                - [0, 0, 0, 1] represents "buy and sell 1 unit"
        """
        action = torch.argmax(action_onehot, dim=1)  # Convert one-hot to action indices
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
        done = t_p >= self.T  # Check if the episode is done

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
        # Earning spread + change in position + running penalty + liquidation penalty
        # discount = self.gamma ** ((self.T - t)/self.dt)
        reward = (
            (isfilled_p + isfilled_m) * self.Delta * 0.5              # Earning spread
            # + q * (S_p - S)                                         # Change in position    
            - self.phi * (q**2) * self.dt                             # Running penalty
            # - discount * self.varphi * (q_p**2 - q**2)              # Liquidation penalty
            - self.varphi * (q_p**2 - q**2)                           # Liquidation penalty
        )

        # for done in the next step, add another liquidation penalty
        reward += - self.varphi * (q_p**2) * done.float()  # Liquidation penalty if done
        
        return t_p, S_p, X_p, q_p, reward, done, isMO, buySellMO

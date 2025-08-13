import numpy as np
import torch


# S_t = S_0 + \sigma * W_t


class StandardEnv:
    def __init__(
        self,
        T=300,  # Total time
        I_max=20,  # Max inventory
        lambd=50 / 300,  # Order arrival rate (for both buy and sell MOs)
        Delta=0.01,  # Spread
        phi=0.01,  # Running penalty
        varphi=0.01,  # Liquidation penalty TODO: experiment
        sigma=0.01,  # Price volatility
        S_0=20,  # Initial midprice
        X_0=0,  # Initial cash
        q_0=0,  # Initial inventory
        Nq=20,
        gamma=0.99,  # Discount factor
    ):
        self.T = T
        self.lambda_p = self.lambda_m = lambd
        self.Ndt = int(
            (self.lambda_p + self.lambda_m) * T * 5
        )  # Number of discrete time steps
        self.dt = T / self.Ndt  # Time step size

        self.t = torch.linspace(0, T, self.Ndt + 1)  # Time vector

        self.I_max = I_max
        self.Delta = Delta
        self.phi = phi
        self.varphi = varphi
        self.sigma = sigma
        self.gamma = gamma
        self.Nq = Nq

        # TODO is this needed?
        self.S_0 = S_0
        self.X_0 = X_0
        self.q_0 = q_0

    def __str__(self):
        return f"StandardEnv(T={self.T}, I_max={self.I_max}, gamma={self.gamma}, lambd={self.lambda_p}, Delta={self.Delta}, phi={self.phi}, varphi={self.varphi}, sigma={self.sigma}, S_0={self.S_0}, X_0={self.X_0}, q_0={self.q_0}, Nq={self.Nq})"

    def Randomize_Start(self, t, mini_batch_size):
        """Randomize the initial state of the environment."""
        S0 = self.S_0 + self.sigma * torch.sqrt(t * self.dt) * torch.randn(
            mini_batch_size
        )  # Initial midprice with noise
        q0 = torch.randint(
            -self.I_max,
            self.I_max + 1,
            size=(mini_batch_size,)
        ).float()
        
        X0 = torch.zeros(mini_batch_size)  # Initial cash
        return S0, q0, X0  # TODO check if this is correct, need to include t?

    def Simulate(self, mini_batch_size=10):
        t = torch.zeros((mini_batch_size, self.Ndt)).float()
        S = torch.zeros((mini_batch_size, self.Ndt)).float()
        X = torch.zeros((mini_batch_size, self.Ndt)).float()
        q = torch.zeros((mini_batch_size, self.Ndt)).float()
        r = torch.zeros((mini_batch_size, self.Ndt)).float()
        action = torch.zeros((mini_batch_size, 1)).squeeze(1).float()
        t[:, 0] = 0
        S[:, 0] = self.S_0
        X[:, 0] = self.X_0
        q[:, 0] = self.q_0

        for i in range(self.Ndt - 1):
            # not finished

            (
                t[:, i + 1],
                S[:, i + 1],
                X[:, i + 1],
                q[:, i + 1],
                r[:, i],
                isMO,
                buySellMO,
            ) = self.step(
                t=t[:, i],
                S=S[:, i],
                X=X[:, i],
                q=q[:, i],
                action=action,
            )

        return S, X, q, r

    def Zero_Start(self, mini_batch_size):
        """Return zero initial state."""
        S0 = torch.zeros(mini_batch_size) + self.S_0
        q0 = torch.zeros(mini_batch_size)
        X0 = torch.zeros(mini_batch_size)
        return S0, q0, X0

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
        )  # -1: sell MO, 1: buy MO

        # Price update
        # S_t = S_0 + \sigma * W_t
        # dS_t = \sigma * sqrt(dt) * N(0, 1)
        S_p = S + self.sigma * torch.sqrt(torch.tensor(self.dt)) * torch.randn(
            mini_batch_size
        )
        # Time update

        t_p = t + 1
        # Cash update
        isfilled_p = (
            isMO.int() * (buySellMO == 1).int() * ((action == 2) | (action == 3)).int()
        )  # Sell LO filled
        isfilled_m = (
            isMO.int() * (buySellMO == -1).int() * ((action == 1) | (action == 3)).int()
        )  # Buy LO filled
        X_p = (
            X
            - (S - 0.5 * self.Delta) * isfilled_m  # Cash spent on buying at bid price
            + (S + 0.5 * self.Delta)
            * isfilled_p  # Cash received from selling at ask price
        )

        # Inventory update
        q_p = q + isfilled_m - isfilled_p
        # --- running reward ---
        reward = (
            (isfilled_p + isfilled_m) * self.Delta * 0.5
            - self.phi * (q**2) * self.dt
        )

        # --- check terminal condition (vector) ---
        done = (t_p >= self.Ndt)  # shape (B,)

        if done.any():
            mask = done.float()

            # Forced liquidation proceeds at ask/bid with slippage
            liquidation_cash = (S_p + 0.5 * self.Delta * torch.sign(q_p)) * q_p

            # Apply to cash
            X_p = X_p + mask * liquidation_cash

            # Add liquidation penalty to reward
            reward = reward - mask * (self.varphi * (q_p**2))

            # Clear inventory at terminal
            q_p = q_p * (1 - mask)
        return t_p, S_p, X_p, q_p, reward, isMO, buySellMO, done

# S_t = S_0 + \sigma * W_t


class TelescopeEnv:
    def __init__(
        self,
        T=300,  # Total time
        I_max=20,  # Max inventory
        lambd=50 / 300,  # Order arrival rate (for both buy and sell MOs)
        Delta=0.01,  # Spread
        phi=0.01,  # Running penalty
        varphi=0.01,  # Liquidation penalty TODO: experiment
        sigma=0.01,  # Price volatility
        S_0=20,  # Initial midprice
        X_0=0,  # Initial cash
        q_0=0,  # Initial inventory
        Nq=20,
        gamma=0.99,  # Discount factor
    ):
        self.T = T
        self.lambda_p = self.lambda_m = lambd
        self.Ndt = int(
            (self.lambda_p + self.lambda_m) * T * 5
        )  # Number of discrete time steps
        self.dt = T / self.Ndt  # Time step size

        self.t = torch.linspace(0, T, self.Ndt + 1)  # Time vector

        self.I_max = I_max
        self.Delta = Delta
        self.phi = phi
        self.varphi = varphi
        self.sigma = sigma
        self.gamma = gamma
        self.Nq = Nq

        # TODO is this needed?
        self.S_0 = S_0
        self.X_0 = X_0
        self.q_0 = q_0

    def __str__(self):
        return f"StandardEnv(T={self.T}, I_max={self.I_max}, gamma={self.gamma}, lambd={self.lambda_p}, Delta={self.Delta}, phi={self.phi}, varphi={self.varphi}, sigma={self.sigma}, S_0={self.S_0}, X_0={self.X_0}, q_0={self.q_0}, Nq={self.Nq})"

    def Randomize_Start(self, t, mini_batch_size):
        """Randomize the initial state of the environment."""
        S0 = self.S_0 + self.sigma * torch.sqrt(t * self.dt) * torch.randn(
            mini_batch_size
        )  # Initial midprice with noise
        q0 = torch.randint(
            -torch.minimum(t.long(), torch.tensor(self.I_max)),
            torch.minimum(t.long(), torch.tensor(self.I_max)) + 1,
            size=(mini_batch_size,)
        ).float()
        X0 = torch.zeros(mini_batch_size)  # Initial cash
        return S0, q0, X0  # TODO check if this is correct, need to include t?

    def Simulate(self, mini_batch_size=10):
        t = torch.zeros((mini_batch_size, self.Ndt)).float()
        S = torch.zeros((mini_batch_size, self.Ndt)).float()
        X = torch.zeros((mini_batch_size, self.Ndt)).float()
        q = torch.zeros((mini_batch_size, self.Ndt)).float()
        r = torch.zeros((mini_batch_size, self.Ndt)).float()
        action = torch.zeros((mini_batch_size, 1)).squeeze(1).float()
        t[:, 0] = 0
        S[:, 0] = self.S_0
        X[:, 0] = self.X_0
        q[:, 0] = self.q_0

        for i in range(self.Ndt - 1):
            # not finished

            (
                t[:, i + 1],
                S[:, i + 1],
                X[:, i + 1],
                q[:, i + 1],
                r[:, i],
                isMO,
                buySellMO,
            ) = self.step(
                t=t[:, i],
                S=S[:, i],
                X=X[:, i],
                q=q[:, i],
                action=action,
            )

        return S, X, q, r

    def Zero_Start(self, mini_batch_size):
        """Return zero initial state."""
        S0 = torch.zeros(mini_batch_size) + self.S_0
        q0 = torch.zeros(mini_batch_size)
        X0 = torch.zeros(mini_batch_size)
        return S0, q0, X0

    def step(self, t, S, X, q, action):
        """
        Environment step with telescoping reward.
        Reward includes incremental liquidation penalty (-varphi * (q_{t+1}^2 - q_t^2)),
        so that the final liquidation penalty is distributed across steps.
        """
        batch_size = S.shape[0]

        # --- simulate MO arrivals ---
        lambda_total = self.lambda_p + self.lambda_m
        isMO = torch.rand(batch_size) < (
            1 - torch.exp(torch.tensor(-self.dt * lambda_total))
        )
        buySellMO = 2 * (torch.rand(batch_size) < self.lambda_p / lambda_total) - 1
        # -1 = sell MO, 1 = buy MO

        # --- price update ---
        S_p = S + self.sigma * torch.sqrt(torch.tensor(self.dt)) * torch.randn(batch_size)

        # --- time update ---
        t_p = t + 1

        # --- cash update from LO fills ---
        isfilled_p = (
            isMO.int() * (buySellMO == 1).int() * ((action == 2) | (action == 3)).int()
        )
        isfilled_m = (
            isMO.int() * (buySellMO == -1).int() * ((action == 1) | (action == 3)).int()
        )
        X_p = (
            X - (S - 0.5 * self.Delta) * isfilled_m
            + (S + 0.5 * self.Delta) * isfilled_p
        )

        # --- inventory update ---
        q_p = q + isfilled_m - isfilled_p

        # --- running reward with telescoping sum ---
        spread_reward = (isfilled_p + isfilled_m) * self.Delta * 0.5
        holding_cost = - self.phi * (q**2) * self.dt
        liquidation_term = - self.varphi * (q_p**2 - q**2)

        reward = spread_reward + holding_cost + liquidation_term

        # --- terminal check ---
        done = (t_p >= self.Ndt)

        return t_p, S_p, X_p, q_p, reward, isMO, buySellMO, done
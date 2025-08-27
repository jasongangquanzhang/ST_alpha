# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 11:41:23 2022

@author: sebja
"""

import numpy as np
from tqdm import tqdm
import pdb
import torch
import math

# from shot_term_alpha_helpers import *
from ShortTermalpha import ShortTermalpha


#
# S_t = S_0 + \int_0^t (\upsilon+ \alpha_u)\,du + \sigma \,W_t
#
class ST_alpha_env:

    def __init__(
        self,
        ShortTermalpha: ShortTermalpha,
        nu=0,
        T=60,
        POV=0.2,
        Nq=20,
        gamma = 0.9998,
        S_0=1300,
        X_0=0,
        q_0=0,
        Delta=0.01,
        varphi=0.01,
        # Running penalty
        phi=0,
        # Price volatility
        sigma=0.005,
    ):
        self.ShortTermalpha = ShortTermalpha
        self.nu = nu
        self.S_0 = S_0
        self.X_0 = X_0
        self.q_0 = q_0
        self.POV = POV
        self.Nq = Nq
        self.gamma = gamma
        self.Delta = Delta
        self.varphi = varphi
        self.phi = phi
        self.sigma = sigma

        # order entry parameters
        self.lambda_p = 0.5 * Nq / POV / T
        self.lambda_m = 0.5 * Nq / POV / T
        self.Ndt = math.floor((self.lambda_p + self.lambda_m) * T * 5)
        self.T = T  # total time
        self.dt = T / self.Ndt

    def lognormal(self, sigma, mini_batch_size=10):
        return torch.exp(-0.5 * sigma**2 + sigma * torch.randn(mini_batch_size))

    def Randomize_Start(
        self, t, mini_batch_size=10
    ):  # TODO in each time of the training, all things should be randomized
        S0 = self.S_0 + self.sigma * torch.sqrt(t * self.dt) * torch.randn(
            mini_batch_size
        )
        q0 = torch.randint(low=-20, high=20, size=(mini_batch_size,))
        X0 = torch.zeros(mini_batch_size)
        alpha0 = (torch.rand(mini_batch_size) - 0.5) * 0.4  # Uniform in [-0.2, 0.2]
        return S0, q0, X0, alpha0

    def Zero_Start(self, mini_batch_size=10):
        S0 = torch.zeros(mini_batch_size) + self.S_0
        q0 = torch.zeros(mini_batch_size)
        X0 = torch.zeros(mini_batch_size)
        alpha0 = torch.zeros(mini_batch_size)
        return S0, q0, X0, alpha0

    def Simulate(self, mini_batch_size=10):

        S = torch.zeros((mini_batch_size, self.Ndt)).float()
        X = torch.zeros((mini_batch_size, self.Ndt)).float()
        alpha = torch.zeros((mini_batch_size, self.Ndt)).float()
        q = torch.zeros((mini_batch_size, self.Ndt)).float()
        r = torch.zeros((mini_batch_size, self.Ndt)).float()
        action = torch.zeros((mini_batch_size, 1)).squeeze(1).float()
        S[:, 0] = self.S_0
        X[:, 0] = self.X_0
        alpha[:, 0] = 0
        q[:, 0] = self.q_0

        for t in range(self.Ndt - 1):
            # not finished
            t_Ndt = t * self.dt / self.T
            (
                S[:, t + 1],
                X[:, t + 1],
                alpha[:, t + 1],
                q[:, t + 1],
                r[:, t],
                isMO,
                buySellMO,
            ) = self.step(
                t_Ndt=t_Ndt,
                S=S[:, t],
                X=X[:, t],
                alpha=alpha[:, t],
                q=q[:, t],
                action=action,
            )

        return S, X, alpha, q, r

    def step(
        self,
        t,
        S: torch.Tensor,
        X: torch.Tensor,
        alpha: torch.Tensor,
        q: torch.Tensor,
        action: torch.Tensor,
    ):
        """
        Advance the environment by one step given the current state and action.

        Args:
            S (torch.Tensor): Current price, shape (Nsims,)
            X (torch.Tensor): Current cash, shape (Nsims,)
            alpha (torch.Tensor): Current alpha, shape (Nsims,)
            q (torch.Tensor): Current inventory, shape (Nsims,)
            action (torch.Tensor): 0 = do nothing, 1 = buy, 2 = sell, 3 = buy and sell, shape (Nsims,)

        Returns:
            S_p (torch.Tensor): Next price, shape (Nsims,)
            X_p (torch.Tensor): Next cash, shape (Nsims,)
            alpha_p (torch.Tensor): Next alpha, shape (Nsims,)
            q_p (torch.Tensor): Next inventory, shape (Nsims,)
            isMO (torch.Tensor): Market order indicator, shape (Nsims,)
            buySellMO (torch.Tensor): Buy/sell indicator, shape (Nsims,)
            reward (torch.Tensor): Reward, shape (Nsims,)
        """

        Nsims = S.size(0)
        # Action is a 2D tensor with shape (Nsims, 4)
        # action[:, 0] is prob of do nothing, action[:, 1] is buy, action[:, 2] is sell, action[:, 3] is buy and sell

        isMO = torch.rand(Nsims) < (
            1 - torch.exp(torch.tensor(-self.dt * (self.lambda_p + self.lambda_m)))
        )

        buySellMO = (
            2 * (torch.rand(Nsims) < self.lambda_p / (self.lambda_p + self.lambda_m))
            - 1
        )  # -1 for sell, 1 for buy
        # Price state
        S_p = (
            S
            + (self.nu + alpha) * self.dt
            + self.sigma * torch.sqrt(torch.tensor(self.dt)) * torch.randn(Nsims)
        )

        # alpha state
        alpha_p = self.ShortTermalpha.get_next_alpha(alpha, isMO, buySellMO, self.dt)
        # time state
        # t_p = t + self.dt
        # Update Inventory
        isfilled_p = (
            ((action == 2) | (action == 3)).int() * isMO.int() * (buySellMO == 1).int()
        )
        isfilled_m = (
            ((action == 1) | (action == 3)).int() * isMO.int() * (buySellMO == -1).int()
        )
        t_p = t + 1
        q_p = q + isfilled_m - isfilled_p
        # update cash
        X_p = (
            X
            - (S - 0.5 * self.Delta) * isfilled_m
            + (S + 0.5 * self.Delta) * isfilled_p
        )
        # calculate reward
        liquidation_price = S_p - 0.5 * self.Delta - self.varphi * q_p

        # Final reward (no running cost since φ = 0)
        # reward = (X_p - X) + (q_p - q) * liquidation_price
        # reward = (X_p - X) + (q_p - q) * S_p
        # reward = (X_p - X) + q_p * (S_p - 0.5 * self.Delta - self.varphi * q_p) - q * (S - 0.5 * self.Delta - self.varphi * q)
        # discount = self.gamma ** (self.Ndt - t.float())
        reward = (
            (isfilled_p + isfilled_m) * 0.5 * self.Delta
            + q * (S_p - S)
            - self.phi * (q**2) * self.dt
            - self.varphi * (q_p**2 - q**2)
        )

        done = t_p >= self.Ndt

        return t_p, S_p, X_p, alpha_p, q_p, reward, isMO, buySellMO, done


# TODO: add experience replay buffer

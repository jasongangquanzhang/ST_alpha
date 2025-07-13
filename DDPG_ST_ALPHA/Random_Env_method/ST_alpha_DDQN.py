# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 10:39:56 2022

@author: sebja
"""

from ST_alpha_env import ST_alpha_env as Environment

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn

from tqdm import tqdm

import copy

import pdb

from datetime import datetime


class DQN(nn.Module):
    def __init__(
        self,
        n_in=2,
        n_out=4,
        nNodes=24,
        nLayers=4,
        activation="relu",
        normalization=False,
    ):
        super(DQN, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(n_in, nNodes))
        for _ in range(nLayers - 1):
            self.hidden_layers.append(nn.Linear(nNodes, nNodes))
        self.output_layer = nn.Linear(nNodes, n_out)
        # Add LayerNorm for each hidden layer
        if normalization:
            self.norms = nn.ModuleList([nn.LayerNorm(nNodes) for _ in range(nLayers)])
        else:
            # If normalization is not used, we still create norms for the hidden layers
            # but they will not be applied in the forward pass.
            # This is to maintain the same structure as the original code.
            self.norms = nn.ModuleList([nn.Identity() for _ in range(nLayers)])

        # Activation function
        if activation == "silu":
            self.g = nn.SiLU()
        elif activation == "relu":
            self.g = nn.ReLU()
        elif activation == "gelu":
            self.g = nn.GELU()
        elif activation == "leakyrelu":
            self.g = nn.LeakyReLU(0.01)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        for i, layer in enumerate(self.hidden_layers):
            x = self.g(self.norms[i](layer(x)))
        return self.output_layer(x)


class ST_alpha_DDQN:

    def __init__(
        self,
        env: Environment,
        gamma=0.99,
        n_nodes=36,
        n_layers=6,
        lr=1e-3,
        sched_step_size=100,
        tau=0.01,
        name="",
    ):

        self.env = env
        self.gamma = gamma
        self.n_nodes = n_nodes
        self.n_layers = n_layers
        self.name = name
        self.sched_step_size = sched_step_size
        self.lr = lr
        self.Nq = env.Nq

        self.__initialize_NNs__()

        self.S = []
        self.q = []
        self.X = []
        self.alpha = []
        self.r = []
        self.epsilon = []

        self.Q_loss = []

        self.tau = tau

    def __initialize_NNs__(self):

        # Q - function approximation
        #
        # features =t, alpha, q
        #
        self.Q_main = {
            "net": DQN(n_in=3, n_out=4, nNodes=self.n_nodes, nLayers=self.n_layers)
        }

        self.Q_main["optimizer"], self.Q_main["scheduler"] = self.__get_optim_sched__(
            self.Q_main
        )

        self.Q_target = copy.deepcopy(self.Q_main)

    def __get_optim_sched__(self, net):

        optimizer = optim.AdamW(net["net"].parameters(), lr=self.lr)

        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=self.sched_step_size, gamma=self.gamma
        )

        return optimizer, scheduler

    def __stack_state__(self, t, S, X, alpha, q):
        """
        Stack the state variables into a single tensor.

        Args:
            t (torch.Tensor): Time variable.
            S (torch.Tensor): Price variable.
            X (torch.Tensor): Cash variable.
            alpha (torch.Tensor): Alpha variable.
            q (torch.Tensor): Inventory variable.

        Returns:
            torch.Tensor: Stacked state tensor.
        """
        return torch.cat(
            (
                t.unsqueeze(-1) / self.env.Ndt,
                # S.unsqueeze(-1) / self.env.S_0 - 1.0,
                alpha.unsqueeze(-1)/0.02,
                # X.unsqueeze(-1), #TODO：try normalize in network
                q.unsqueeze(-1) / self.Nq,
            ),
            axis=-1,
        ).float()

    def __grab_mini_batch__(self, mini_batch_size):
        """
        Grab a mini-batch of data from the environment.
        Args:
            mini_batch_size (int): Size of the mini-batch.
        Returns:
            tuple: A tuple containing the time, state, cash, alpha, and inventory tensors.
        """
        # t is relative time from end
        t = torch.randint(0, self.env.Ndt, (mini_batch_size,))
        # t[-int(mini_batch_size*0.05):] = self.env.N

        S, q, X, alpha = self.env.Randomize_Start(t, mini_batch_size)
        # TODO: plot these to check these
        return t, S, q, X, alpha

    def update_Q(self, n_iter=10, mini_batch_size=256, epsilon=0.02):# TODO: make it time update not random
        for i in range(n_iter):

            t, S, q, X, alpha = self.__grab_mini_batch__(mini_batch_size)

            self.Q_main["optimizer"].zero_grad()

            # concatenate states
            state = self.__stack_state__(t=t, S=S, q=q, X=X, alpha=alpha)

            Q = self.Q_main["net"](state)  # (batch_size, n_actions)

            # --- Epsilon-greedy action selection ---
            batch_size = Q.shape[0]
            rand_vals = torch.rand(batch_size)
            random_actions = torch.randint(0, Q.shape[1], (batch_size,))
            greedy_actions = Q.argmax(dim=1)
            # With probability epsilon choose random, else greedy
            actions = torch.where(rand_vals < epsilon, random_actions, greedy_actions)

            # Gather Q-value for chosen action
            Q_value = Q.gather(1, actions.unsqueeze(1)).squeeze(1)

            # Step in environment to get next state and reward
            t_p, S_p, X_p, alpha_p, q_p, r, isMO, buySellMO = self.env.step(
                t, S, X, alpha, q, actions
            )
            # New state
            state_p = self.__stack_state__(
                t=t_p, S=S_p, X=X_p, alpha=alpha_p, q=q_p
            )
            Q_p = self.Q_main["net"](state_p)
            # Next greedy action for Double DQN
            next_greedy_actions = Q_p.argmax(dim=1, keepdim=True)
            # Target value using target net
            target_q_values = (
                self.Q_target["net"](state_p).gather(1, next_greedy_actions).squeeze(1)
            )

            # Compute target
            target = r + self.gamma * target_q_values
            target = target.detach()

            # Loss
            loss = torch.mean((Q_value - target) ** 2)
            loss.backward()
            self.Q_main["optimizer"].step()
            self.Q_main["scheduler"].step()
            self.Q_loss.append(loss.item())

            # Target network soft update
            self.soft_update(self.Q_main["net"], self.Q_target["net"])

    def soft_update(self, main, target):

        for param, target_param in zip(main.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def train(self, n_iter=1_000, n_iter_Q=10, mini_batch_size=256, n_plot=100):

        self.run_strategy(
            nsims=1_000, name=datetime.now().strftime("%H_%M_%S")
        )  # intital evaluation

        C = 50
        D = 100

        if len(self.epsilon) == 0:
            self.count = 0

        # for i in tqdm(range(n_iter)):
        for i in range(n_iter):

            epsilon = np.maximum(C / (D + self.count), 0.02)
            self.epsilon.append(epsilon)
            self.count += 1

            # pdb.set_trace()

            self.update_Q(
                n_iter=n_iter_Q, mini_batch_size=mini_batch_size, epsilon=epsilon
            )

            if np.mod(i + 1, n_plot) == 0:

                self.loss_plots()
                self.run_strategy(
                    1_000, name=datetime.now().strftime("%H_%M_%S")
                )
                self.plot_policy()
                # self.plot_policy(name=datetime.now().strftime("%H_%M_%S"))

    def moving_average(self, x, n):

        y = np.zeros(len(x))
        y_err = np.zeros(len(x))
        y[0] = np.nan
        y_err[0] = np.nan

        for i in range(1, len(x)):

            if i < n:
                y[i] = np.mean(x[:i])
                y_err[i] = np.std(x[:i])
            else:
                y[i] = np.mean(x[i - n : i])
                y_err[i] = np.std(x[i - n : i])

        return y, y_err

    def loss_plots(self):

        def plot(x, label, show_band=True):

            mv, mv_err = self.moving_average(x, 100)

            if show_band:
                plt.fill_between(
                    np.arange(len(mv)), mv - mv_err, mv + mv_err, alpha=0.2
                )
            plt.plot(mv, label=label, linewidth=1)
            plt.legend()
            plt.ylabel("loss")
            plt.yscale("symlog")

        fig = plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plot(self.Q_loss, r"$Q$", show_band=False)

        plt.tight_layout()
        plt.show()

    def run_strategy(self, nsims: int = 10_000, name: str = "", N: int = None):
        """Run the trading strategy simulation. Seem to be an evaluation for current policy network.
        The function simulates the evolution of the system over a specified number of time steps and plots the results.

        Args:
            nsims (int, optional): The number of simulations to run. Defaults to 10_000.
            name (str, optional): The name of the simulation. Defaults to "".
            N (int, optional): The number of time steps to simulate. Defaults to None.

        Returns:
            _type_: _description_
        """

        if N is None:
            N = self.env.Ndt  # number of time steps
        time = torch.zeros((nsims, N + 1)).float()
        S = torch.zeros((nsims, N + 1)).float()
        q = torch.zeros((nsims, N + 1)).float()
        X = torch.zeros((nsims, N + 1)).float()
        alpha = torch.zeros((nsims, N + 1)).float()
        action = torch.zeros((nsims, N)).float()
        r = torch.zeros((nsims, N)).float()
        # keep track of market orders
        isMO = torch.zeros((nsims, N)).float()
        buySellMO = torch.zeros((nsims, N)).float()

        S[:, 0], q[:, 0], X[:, 0], alpha[:, 0] = self.env.Zero_Start(
            mini_batch_size=nsims
        )

        ones = torch.ones(nsims)

        for step in range(N):

            state = self.__stack_state__(t=time[:, step], S=S[:, step], q=q[:, step], X=X[:, step], alpha=alpha[:, step])
            Q = self.Q_main["net"](state)
            action[:, step] = Q.argmax(dim=1)
            (   time[:, step + 1],
                S[:, step + 1],
                X[:, step + 1],
                alpha[:, step + 1],
                q[:, step + 1],
                r[:, step],
                isMO[:, step],
                buySellMO[:, step],
            ) = self.env.step(
                time[:, step], S[:, step], X[:, step], alpha[:, step], q[:, step], action[:, step]
            )


        # Clear position at the end of the simulation
        X[:, X.shape[1] - 1] += np.multiply(
            S[:, S.shape[1] - 1]
            + (0.5 * self.env.Delta) * np.sign(q[:, q.shape[1] - 1])
            + self.env.varphi * q[:, q.shape[1] - 1],
            q[:, q.shape[1] - 1],
        )
        q[:, q.shape[1] - 1] = 0

        # extract everything
        time = time.detach().numpy()
        S = S.detach().numpy()
        X = X.detach().numpy()
        alpha = alpha.detach().numpy()
        q = q.detach().numpy()
        r = r.detach().numpy()
        action = action.detach().numpy()
        isMO = isMO.detach().numpy()
        buySellMO = buySellMO.detach().numpy()
        t = self.env.dt * np.arange(0, N + 1) / self.env.T

        plt.figure(figsize=(10, 10))
        n_paths = 10

        def plot(t, x, plt_i, title):

            # print(x.shape)
            # pdb.set_trace()

            qtl = np.quantile(x, [0.05, 0.5, 0.95], axis=0)
            # print(qtl.shape)

            plt.subplot(2, 3, plt_i)

            plt.fill_between(t, qtl[0, :], qtl[2, :], alpha=0.5)
            plt.plot(t, qtl[1, :], color="k", linewidth=1)
            plt.plot(t, x[:n_paths, :].T, linewidth=1)

            # plt.xticks([0,0.5,1])
            plt.title(title)
            plt.xlabel(r"$t$")

        plot(t, (S), 1, r"$S_t$")
        plot(t, alpha, 2, r"$\alpha_t$")
        # plot(t[1:], q[:, 1:] - q[:, :-1], 2, r"$q_t - q_{t-1}$")
        plot(t, q, 3, r"$q_t$")

        plot(t[:-1], np.cumsum(r, axis=1), 4, r"$r_t$")
        plot(t, X + S * q, 5, r"$Wealth$")

        plt.subplot(2, 3, 6)
        # plt.hist(np.sum(r, axis=1), bins=51)

        plt.tight_layout()

        # plt.savefig(
        #     "path_" + self.name + "_" + name + ".pdf", format="pdf", bbox_inches="tight"
        # )
        plt.show()

        # zy0 = self.env.swap_price(zx[0,0], rx[0,0], ry[0,0])
        # plt.hist(zy[:,-1],bins=np.linspace(51780,51810,31), density=True, label='optimal')
        # qtl_levels = [0.05,0.5,0.95]
        # qtl = np.quantile(zy[:,-1],qtl_levels)
        # c=['r','b','g']
        # for i, q in enumerate(qtl):
        #     plt.axvline(qtl[i], linestyle='--',
        #                 linewidth=2,
        #                 color=c[i],
        #                 label=r'${0:0.2f}$'.format(qtl_levels[i]))
        # plt.axvline(zy0,linestyle='--',color='k', label='swap-all')
        # plt.xlabel(r'$z_T^y$')
        # plt.legend()
        # plt.savefig('ddqn_zy_T.pdf', format='pdf',bbox_inches='tight')
        # plt.show()

        # print(zy0, np.mean(zy[:,-1]>zy0))
        # print(qtl)

        pass


    def plot_policy(self, name=""):
        """Plots the policy as a single heatmap showing the action with the highest probability."""
        num_alpha_points = 51
        num_inventory_points = 51
        alpha_values = torch.linspace(-0.2, 0.2, num_alpha_points)
        inventory_levels = torch.linspace(-self.Nq, self.Nq, num_inventory_points)

        max_prob_action = np.zeros((num_inventory_points, num_alpha_points))
        max_prob_value = np.zeros((num_inventory_points, num_alpha_points))
        with torch.no_grad():
            for i, q in enumerate(inventory_levels):
                for j, alpha in enumerate(alpha_values):
                    state = self.__stack_state__(
                        t=0.5 * torch.ones(1),
                        S=self.env.S_0 * torch.ones(1),
                        X=torch.zeros(1),
                        alpha=torch.tensor([alpha]),
                        q=torch.tensor([q]),
                    )
                    policy_output = self.Q_main["net"](state).argmax(dim=1, keepdim=False).squeeze().numpy()
                    max_prob_action[i, j] = policy_output
                    max_prob_value[i, j] = np.max(policy_output)

        plt.figure(figsize=(10, 8))
        plt.contourf(
            alpha_values.numpy(),
            inventory_levels.numpy(),
            max_prob_action,
            levels=np.arange(-0.5, 4, 1),
            cmap="viridis",
            alpha=0.8,
        )
        plt.colorbar(ticks=range(4), label="Action")
        plt.axhline(0, linestyle="--", color="k", linewidth=0.8)
        plt.axvline(0, linestyle="--", color="k", linewidth=0.8)
        plt.title("Policy Heatmap - Action", fontsize=16)
        plt.xlabel(r"$\alpha$", fontsize=14)
        plt.ylabel("Inventory", fontsize=14)
        plt.tight_layout()
        plt.show()


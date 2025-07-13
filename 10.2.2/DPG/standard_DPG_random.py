# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 10:39:56 2022

@author: sebja
"""

from standard_env import StandardEnv as Environment

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn

from tqdm import tqdm
import copy
import pdb
from datetime import datetime


class ANN(nn.Module):
    def __init__(
        self,
        n_in,
        n_out,
        nNodes,
        nLayers,
        activation="relu",
        out_activation=None,
        temperature=30,
        scale=1,
    ):
        super(ANN, self).__init__()

        self.prop_in_to_h = nn.Linear(n_in, nNodes)

        self.prop_h_to_h = nn.ModuleList(
            [nn.Linear(nNodes, nNodes) for _ in range(nLayers - 1)]
        )

        # Add LayerNorm for each hidden layer
        self.norms = nn.ModuleList([nn.LayerNorm(nNodes) for _ in range(nLayers - 1)])

        self.prop_h_to_out = nn.Linear(nNodes, n_out)

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

        self.out_activation = out_activation
        self.temperature = temperature
        self.scale = scale

    def forward(self, x):
        h = self.g(self.prop_in_to_h(x))

        for i, layer in enumerate(self.prop_h_to_h):
            h = self.g(self.norms[i](layer(h)))

        y = self.prop_h_to_out(h)

        if self.out_activation == "tanh":
            y = torch.tanh(y)
        elif self.out_activation == "sigmoid":
            y = torch.sigmoid(y)
        elif self.out_activation == "softmax":
            y = torch.softmax(y / self.temperature, dim=1)

        return y


class DPGAgent:

    def __init__(
        self,
        env: Environment,
        n_nodes=36,
        n_layers=6,
        gamma = 0.99,
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
        self.r = []
        self.epsilon = []

        self.Q_loss = []
        self.pi_loss = []

        self.tau = tau

    def __initialize_NNs__(self):

        # policy approximation
        #
        # features = t/T,S,  q
        #
        self.pi_main = {
            "net": ANN(
                n_in=3,
                n_out=4,
                nNodes=self.n_nodes,
                nLayers=self.n_layers,
                out_activation="softmax",
                scale=self.Nq,
            )
        }

        self.pi_main["optimizer"], self.pi_main["scheduler"] = self.__get_optim_sched__(
            self.pi_main
        )

        self.pi_target = copy.deepcopy(self.pi_main)

        # Q - function approximation
        #
        # features = t,S, q, action
        #
        self.Q_main = {
            "net": ANN(n_in=4, n_out=1, nNodes=self.n_nodes, nLayers=self.n_layers)
        }

        self.Q_main["optimizer"], self.Q_main["scheduler"] = self.__get_optim_sched__(
            self.Q_main
        )

        self.Q_target = copy.deepcopy(self.Q_main)

    def __get_optim_sched__(self, net):

        optimizer = optim.AdamW(net["net"].parameters(), lr=self.lr)

        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=self.sched_step_size, gamma=0.99
        )

        return optimizer, scheduler

    def __stack_state__(self, t, S, X, q):
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
                (S.unsqueeze(-1) - self.env.S_0) / self.env.sigma,
                # X.unsqueeze(-1),
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
        # t is randomly sampled from 0 to Ndt
        t = torch.randint(0, self.env.Ndt, (mini_batch_size,))
        # t[-int(mini_batch_size*0.05):] = self.env.N
        # NOTE: in a brownian motion, the standard deviation is proportional to the square root of time
        S, q, X = self.env.Randomize_Start(t, mini_batch_size)

        return t, S, q, X

    def update_Q(self, n_iter=1, mini_batch_size=256, epsilon=0.02):

        for i in range(n_iter):

            t, S, q, X= self.__grab_mini_batch__(mini_batch_size)

            self.Q_main["optimizer"].zero_grad()

            # concatenate states
            state = self.__stack_state__(t=t, S=S, q=q, X=X)

            # compute the action

            if np.random.rand() < epsilon:
                # Random action (uniform exploration)
                action = torch.randint(0, 4, (mini_batch_size,)).unsqueeze(1)
            else:
                # Sample from the policy's softmax output
                
                action = torch.multinomial(
                    self.pi_main["net"](state).detach(), num_samples=1
                )

            # compute the value of the action I_p given state X
            Q = self.Q_main["net"](torch.cat((state, action), axis=1))

            # step in the environment get the next state and reward
            t_p,S_p, X_p, q_p, r, isMO, buySellMO = self.env.step(
                t=t, S=S, X=X, q=q, action=action.squeeze(1)
            )
            # compute the Q(S', a*)
            # concatenate new state
            state_p = self.__stack_state__(
                t=t_p, S=S_p, q=q_p, X=X_p
            )

            # optimal policy at t+1 get the next action action_p
            if np.random.rand() < epsilon:
                # Random action (uniform exploration)
                action_p = torch.randint(0, 4, (mini_batch_size,)).unsqueeze(1)
                # print("RANDOM ACTION:", action_p)
            else:
                # Sample from the policy's softmax output
                action_p = torch.multinomial(
                    self.pi_main["net"](state_p).detach(), num_samples=1
                )
                # print("POLICY ACTION:", action_p)

            # compute the target for Q
            
            target = r.reshape(-1, 1).detach() + self.env.gamma * self.Q_target["net"](
                torch.cat((state_p, action_p), axis=1)
            )

            loss = torch.mean((target.detach() - Q) ** 2)

            # compute the gradients
            loss.backward()

            # perform step using those gradients
            self.Q_main["optimizer"].step()
            self.Q_main["scheduler"].step()

            self.Q_loss.append(loss.item())

            self.soft_update(self.Q_main["net"], self.Q_target["net"])

    def update_pi(self, n_iter=1, mini_batch_size=256, epsilon=0.02):

        for i in range(n_iter):

            t, S, q, X = self.__grab_mini_batch__(mini_batch_size)

            self.pi_main["optimizer"].zero_grad()

            # concatenate states
            state = self.__stack_state__(t=t, S=S, q=q, X=X)
            probs = self.pi_main["net"](state)
            action = torch.multinomial(probs, num_samples=1)  # Sample action from the policy distribution
            log_probs = torch.log(probs + 1e-8)  # prevent log(0)
            selected_log_probs = log_probs.gather(1, action) # NOTE: log prob so that it got updated
            Q = self.Q_main["net"](torch.cat((state, action), axis=1))
            # entropy = -torch.mean(action * torch.log(action + 1e-8))
            # loss = -torch.mean(Q) + 0.01 * entropy
            loss = -torch.mean(selected_log_probs * Q.detach())

            loss.backward()
            self.pi_main["optimizer"].step()
            self.pi_main["scheduler"].step()

            self.pi_loss.append(loss.item())

    def soft_update(self, main, target):

        for param, target_param in zip(main.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def train(
        self, n_iter=1_000, n_iter_Q=10, n_iter_pi=5, mini_batch_size=256, n_plot=100
    ):

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

            self.update_pi(
                n_iter=n_iter_pi, mini_batch_size=mini_batch_size, epsilon=epsilon
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

        plt.subplot(1, 2, 2)
        plot(self.pi_loss, r"$\pi$")

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
        action = torch.zeros((nsims, N)).float()
        r = torch.zeros((nsims, N)).float()
        # keep track of market orders
        isMO = torch.zeros((nsims, N)).float()
        buySellMO = torch.zeros((nsims, N)).float()

        S[:, 0], q[:, 0], X[:, 0]= self.env.Zero_Start(nsims)
        ones = torch.ones(nsims)

        for step in range(N):  # t = 0->N-1
            # concatenate states
            state = self.__stack_state__(
                t=time[:, step], S=S[:, step], X=X[:, step], q=q[:, step]
            )
            # compute the action
            action[:, step] = torch.multinomial(
                self.pi_main["net"](state), num_samples=1
            ).squeeze(1)

            (   time[:, step + 1],
                S[:, step + 1],
                X[:, step + 1],
                q[:, step + 1],
                r[:, step],
                isMO[:, step],
                buySellMO[:, step],
            ) = self.env.step(
                t=time[:, step],
                S=S[:, step],
                X=X[:, step],
                q=q[:, step],
                action=action[:, step],
            )
        # Clear position at the end of the simulation
        X[:, X.shape[1] - 1] += np.multiply(
            S[:, S.shape[1] - 1]
            + (0.5 * self.env.Delta) * np.sign(q[:, q.shape[1] - 1])
            - self.env.varphi * q[:, q.shape[1] - 1],
            q[:, q.shape[1] - 1],
        )
        q[:, q.shape[1] - 1] = 0

        # extract everything
        time = time.detach().numpy()
        S = S.detach().numpy()
        X = X.detach().numpy()
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

        plot(t, S, 1, r"$S_t$")

        plot(t[1:], q[:, 1:] - q[:, :-1], 2, r"$q_t - q_{t-1}$")
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
        num_inventory_points = 51
        num_time_points = 51
        inventory_values = torch.linspace(-self.Nq, self.Nq, num_inventory_points)
        time_steps = torch.linspace(0, self.env.Ndt, num_time_points)

        max_prob_action = np.zeros((num_time_points, num_inventory_points))
        max_prob_value = np.zeros((num_time_points, num_inventory_points))
        with torch.no_grad():
            for i, q in enumerate(inventory_values):
                for j, t in enumerate(time_steps):
                    state = self.__stack_state__(
                        t=torch.tensor([t]),
                        S=self.env.S_0 * torch.ones(1),
                        X=torch.zeros(1),
                        q=torch.tensor([q]),
                    )
                    policy_output = self.pi_main["net"](state).squeeze().numpy()
                    max_prob_action[i, j] = np.argmax(policy_output)
                    max_prob_value[i, j] = np.max(policy_output)

        plt.figure(figsize=(10, 8))
        plt.contourf(
            time_steps.numpy(),
            inventory_values.numpy(),
            max_prob_action,
            levels=np.arange(-0.5, 4, 1),
            cmap="viridis",
            alpha=0.8,
        )
        plt.colorbar(ticks=range(4), label="Action with Max Probability")
        plt.axhline(0, linestyle="--", color="k", linewidth=0.8)
        plt.axvline(0, linestyle="--", color="k", linewidth=0.8)
        plt.title("Policy Heatmap - Action with Max Probability", fontsize=16)
        plt.xlabel(r"$t$", fontsize=14)
        plt.ylabel("Inventory", fontsize=14)
        plt.tight_layout()
        plt.show()

    # def plot_policy2(self, name=""):
    #     """Plots the policy as heatmaps showing the probabilities for both buy and sell actions."""
    #     num_alpha_points = 51
    #     num_inventory_points = 51
    #     alpha_values = torch.linspace(-0.1, 0.1, num_alpha_points)
    #     inventory_levels = torch.linspace(-40, 40, num_inventory_points)

    #     policy_buy = np.zeros((num_inventory_points, num_alpha_points))
    #     policy_sell = np.zeros((num_inventory_points, num_alpha_points))
    #     with torch.no_grad():
    #         for i, q in enumerate(inventory_levels):
    #             for j, alpha in enumerate(alpha_values):
    #                 state = self.__stack_state__(
    #                     t_Ndt=0.5 * torch.ones(1),
    #                     S=self.env.S_0 * torch.ones(1),
    #                     X=torch.zeros(1),
    #                     alpha=torch.tensor([alpha]),
    #                     q=torch.tensor([q]),
    #                 )
    #                 policy_output = self.pi_main["net"](state).squeeze().numpy()
    #                 # Assuming policy_output[0] is buy probability and policy_output[1] is sell probability
    #                 policy_buy[i, j] = policy_output[0]
    #                 policy_sell[i, j] = policy_output[1]

    #     # Plot buy probability heatmap
    #     plt.figure(figsize=(8, 6))
    #     plt.contourf(
    #         alpha_values.numpy(),
    #         inventory_levels.numpy(),
    #         policy_buy,
    #         levels=np.linspace(0, 1, 21),
    #         cmap="Blues",
    #         alpha=0.8,
    #     )
    #     plt.colorbar(label="Buy Probability")
    #     plt.axhline(0, linestyle="--", color="k", linewidth=0.8)
    #     plt.axvline(0, linestyle="--", color="k", linewidth=0.8)
    #     plt.title("Buy Probability Heatmap", fontsize=16)
    #     plt.xlabel(r"$\alpha$", fontsize=14)
    #     plt.ylabel("Inventory", fontsize=14)
    #     plt.tight_layout()
    #     plt.show()

    #     # Plot sell probability heatmap
    #     plt.figure(figsize=(8, 6))
    #     plt.contourf(
    #         alpha_values.numpy(),
    #         inventory_levels.numpy(),
    #         policy_sell,
    #         levels=np.linspace(0, 1, 21),
    #         cmap="Reds",
    #         alpha=0.8,
    #     )
    #     plt.colorbar(label="Sell Probability")
    #     plt.axhline(0, linestyle="--", color="k", linewidth=0.8)
    #     plt.axvline(0, linestyle="--", color="k", linewidth=0.8)
    #     plt.title("Sell Probability Heatmap", fontsize=16)
    #     plt.xlabel(r"$\alpha$", fontsize=14)
    #     plt.ylabel("Inventory", fontsize=14)
    #     plt.tight_layout()
    #     plt.show()

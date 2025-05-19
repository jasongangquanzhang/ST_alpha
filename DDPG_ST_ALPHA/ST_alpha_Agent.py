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


class ANN(nn.Module):

    def __init__(
        self,
        n_in,
        n_out,
        nNodes,
        nLayers,
        activation="relu",
        out_activation=None,
        scale=1,
    ):
        super(ANN, self).__init__()

        self.prop_in_to_h = nn.Linear(n_in, nNodes)

        self.prop_h_to_h = nn.ModuleList(
            [nn.Linear(nNodes, nNodes) for i in range(nLayers - 1)]
        )

        self.prop_h_to_out = nn.Linear(nNodes, n_out)

        if activation == "silu":
            self.g = nn.SiLU()
        elif activation == "relu":
            self.g = nn.ReLU()
        elif activation == "sigmoid":
            self.g = torch.sigmoid()

        self.out_activation = out_activation
        self.scale = scale

    def forward(self, x):

        # input into  hidden layer
        h = self.g(self.prop_in_to_h(x))

        for prop in self.prop_h_to_h:
            h = self.g(prop(h))

        # hidden layer to output layer
        y = self.prop_h_to_out(h)

        if self.out_activation == "tanh":
            y = torch.tanh(y)
        elif self.out_activation == "sigmoid":
            y = torch.sigmoid(y)

        # y = self.scale * y

        return y


class ST_alpha_Agent:

    def __init__(
        self,
        env: Environment,
        gamma=0.99,
        n_nodes=36,
        n_layers=6,
        lr=1e-3,
        sched_step_size=100,
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
        self.pi_loss = []

        self.tau = 0.001

    def __initialize_NNs__(self):

        # policy approximation
        #
        # features = t/T,S, X, alpha, q
        #
        self.pi_main = {
            "net": ANN(
                n_in=5,
                n_out=2,
                nNodes=self.n_nodes,
                nLayers=self.n_layers,
                out_activation="sigmoid",
                scale=self.Nq,
            )
        }

        self.pi_main["optimizer"], self.pi_main["scheduler"] = self.__get_optim_sched__(
            self.pi_main
        )

        self.pi_target = copy.deepcopy(self.pi_main)

        # Q - function approximation
        #
        # features = t/T, S, X, alpha, q, action
        #
        self.Q_main = {
            "net": ANN(n_in=7, n_out=1, nNodes=self.n_nodes, nLayers=self.n_layers)
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

    def __stack_state__(self, t_Ndt, S, X, alpha, q):
        """
        Stack the state variables into a single tensor.

        Args:
            t_Ndt (torch.Tensor): Time variable.
            S (torch.Tensor): Price variable.
            X (torch.Tensor): Cash variable.
            alpha (torch.Tensor): Alpha variable.
            q (torch.Tensor): Inventory variable.

        Returns:
            torch.Tensor: Stacked state tensor.
        """
        return torch.cat(
            (
                t_Ndt.unsqueeze(-1),
                S.unsqueeze(-1) / self.env.S_0 - 1.0,
                X.unsqueeze(-1),
                alpha.unsqueeze(-1),
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
        t = torch.randint(0, self.env.Ndt, (mini_batch_size,)) / self.env.Ndt
        # t[-int(mini_batch_size*0.05):] = self.env.N

        S, q, X, alpha = self.env.Randomize_Start(mini_batch_size)

        return t, S, q, X, alpha

    def update_Q(self, n_iter=10, mini_batch_size=256, epsilon=0.02):

        for i in range(n_iter):

            t_Ndt, S, q, X, alpha = self.__grab_mini_batch__(mini_batch_size)

            self.Q_main["optimizer"].zero_grad()

            # concatenate states
            state = self.__stack_state__(t_Ndt=t_Ndt, S=S, q=q, X=X, alpha=alpha)

            # compute the action
            action = self.pi_main["net"](state).detach()

            # compute the value of the action I_p given state X
            Q = self.Q_main["net"](torch.cat((state, action), axis=1))

            # step in the environment get the next state and reward
            S_p, X_p, alpha_p, q_p, r, isMO, buySellMO = self.env.step(
                t_Ndt=t_Ndt, S=S, X=X, alpha=alpha, q=q, action=action
            )

            # compute the Q(S', a*)
            # concatenate new state
            state_p = self.__stack_state__(
                t_Ndt=t_Ndt, S=S_p, q=q_p, X=X_p, alpha=alpha_p
            )

            # optimal policy at t+1 get the next action action_p
            action_p = self.pi_main["net"](state_p).detach()

            # compute the target for Q
            # NOTE: the target is not clipped and Q_target is used
            target = r.reshape(-1, 1) + self.gamma * self.Q_target["net"](
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

    def update_pi(self, n_iter=10, mini_batch_size=256, epsilon=0.02):

        for i in range(n_iter):

            t_Ndt, S, q, X, alpha = self.__grab_mini_batch__(mini_batch_size)

            self.pi_main["optimizer"].zero_grad()

            # concatenate states
            state = self.__stack_state__(t_Ndt=t_Ndt, S=S, q=q, X=X, alpha=alpha)

            action = self.pi_main["net"](state)

            Q = self.Q_main["net"](torch.cat((state, action), axis=1))

            loss = -torch.mean(Q)

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
                    1_000, name=datetime.now().strftime("%H_%M_%S"), N=100
                )
                # TODO self.plot_policy()
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

        S = torch.zeros((nsims, N + 1)).float()
        q = torch.zeros((nsims, N + 1)).float()
        X = torch.zeros((nsims, N + 1)).float()
        alpha = torch.zeros((nsims, N + 1)).float()
        action = torch.zeros((nsims, N, 2)).float()
        r = torch.zeros((nsims, N)).float()
        # keep track of market orders
        isMO = torch.zeros((nsims, N)).float()
        buySellMO = torch.zeros((nsims, N)).float()

        S[:, 0], q[:, 0], X[:, 0], alpha[:, 0] = self.env.Randomize_Start(
            mini_batch_size=nsims
        )

        ones = torch.ones(nsims)

        for t in range(N):  # t = 0->N-1
            # concatenate states
            t_Ndt = torch.full((nsims,), t / N)
            state = self.__stack_state__(
                t_Ndt=t_Ndt, S=S[:, t], X=X[:, t], alpha=alpha[:, t], q=q[:, t]
            )
            # compute the action
            action[:, t] = self.pi_main["net"](state)  # this returns a probability

            (
                S[:, t + 1],
                X[:, t + 1],
                alpha[:, t + 1],
                q[:, t + 1],
                r[:, t],
                isMO[:, t],
                buySellMO[:, t],
            ) = self.env.step(
                t_Ndt=t_Ndt,
                S=S[:, t],
                X=X[:, t],
                alpha=alpha[:, t],
                q=q[:, t],
                action=action[:, t],
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
        S = S.detach().numpy()
        X = X.detach().numpy()
        alpha = alpha.detach().numpy()
        q = q.detach().numpy()
        r = r.detach().numpy()
        action = action.detach().numpy()
        isMO = isMO.detach().numpy()
        buySellMO = buySellMO.detach().numpy()

        t = self.env.dt * np.arange(0, N + 1) / self.env.T

        plt.figure(figsize=(5, 5))
        n_paths = 3

        def plot(t, x, plt_i, title):

            # print(x.shape)
            # pdb.set_trace()

            qtl = np.quantile(x, [0.05, 0.5, 0.95], axis=0)
            # print(qtl.shape)

            plt.subplot(2, 2, plt_i)

            plt.fill_between(t, qtl[0, :], qtl[2, :], alpha=0.5)
            plt.plot(t, qtl[1, :], color="k", linewidth=1)
            plt.plot(t, x[:n_paths, :].T, linewidth=1)

            # plt.xticks([0,0.5,1])
            plt.title(title)
            plt.xlabel(r"$t$")

        # plot(t, (S - S[:, 0].reshape(S.shape[0], -1)), 1, r"$S_t-S_0$")
        plot(t, alpha, 1, r"$\alpha_t$")
        # plot(t[1:], q[:, 1:] - q[:, :-1], 2, r"$q_t - q_{t-1}$")
        plot(t, q, 2, r"$q_t$")
        
        plot(t[:-1], np.cumsum(r, axis=1), 3, r"$r_t$")

        plt.subplot(2, 2, 4)
        plt.hist(np.sum(r, axis=1), bins=51)

        plt.tight_layout()

        plt.savefig(
            "path_" + self.name + "_" + name + ".pdf", format="pdf", bbox_inches="tight"
        )
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
        return S, X, alpha, q, r, action, isMO, buySellMO

    # def plot_policy(self, name=""):

    #     NS = 101

    #     S = torch.linspace(
    #         self.env.S_0 - 3 * self.env.inv_vol, self.env.S_0 + 3 * self.env.inv_vol, NS
    #     )
    #     NI = 51
    #     I = torch.linspace(-self.I_max, self.I_max, NI)

    #     Sm, Im = torch.meshgrid(S, I, indexing="ij")

    #     def plot(a, title):

    #         fig, ax = plt.subplots()
    #         plt.title("Inventory vs Price Heatmap for Time T")

    #         cs = plt.contourf(
    #             Sm.numpy(),
    #             Im.numpy(),
    #             a,
    #             levels=np.linspace(-self.I_max, self.I_max, 21),
    #             cmap="RdBu",
    #         )
    #         plt.axvline(self.env.S_0, linestyle="--", color="g")
    #         plt.axvline(self.env.S_0 - 2 * self.env.inv_vol, linestyle="--", color="k")
    #         plt.axvline(self.env.S_0 + 2 * self.env.inv_vol, linestyle="--", color="k")
    #         plt.axhline(0, linestyle="--", color="k")
    #         plt.axhline(self.I_max / 2, linestyle="--", color="k")
    #         plt.axhline(-self.I_max / 2, linestyle="--", color="k")
    #         ax.set_xlabel("Price")
    #         ax.set_ylabel("Inventory")
    #         ax.set_title(title)

    #         cbar = fig.colorbar(cs, ax=ax, shrink=0.9)
    #         cbar.set_ticks(np.linspace(-self.I_max, self.I_max, 11))
    #         cbar.ax.set_ylabel("Action")

    #         plt.tight_layout()
    #         plt.show()

    #     # X = torch.cat( ((Sm.unsqueeze(-1)/self.env.S_0-1.0),
    #     #                 Im.unsqueeze(-1)/self.I_max), axis=-1)

    #     X = self.__stack_state__(Sm, Im)

    #     a = self.pi_main["net"](X).detach().squeeze()

    #     plot(a, r"")


    def plot_policy(self, name=""):

        NI = 51
        I = torch.linspace(-self.Nq, self.Nq, NI)
        NA = 51
        A = torch.linspace(0, 1, NA)

        Im, Am = torch.meshgrid(I, A, indexing="ij")

        def plot(a, title):

            fig, ax = plt.subplots()
            plt.title("Inventory vs Alpha Heatmap for Action")

            cs = plt.contourf(
                Im.numpy(),
                Am.numpy(),
                a,
                levels=np.linspace(0, 1, 21),
                cmap="RdBu",
            )
            plt.axhline(0.5, linestyle="--", color="k")
            plt.axvline(0, linestyle="--", color="k")
            ax.set_xlabel("Inventory")
            ax.set_ylabel("Alpha")
            ax.set_title(title)

            cbar = fig.colorbar(cs, ax=ax, shrink=0.9)
            cbar.set_ticks(np.linspace(0, 1, 11))
            cbar.ax.set_ylabel("Action")

            plt.tight_layout()
            plt.show()

        X = self.__stack_state__(
            t_Ndt=torch.zeros_like(Im.flatten()),
            S=torch.zeros_like(Im.flatten()),
            X=torch.zeros_like(Im.flatten()),
            alpha=Am.flatten(),
            q=Im.flatten(),
        )

        a = self.pi_main["net"](X).detach().squeeze()

        plot(a.reshape(Im.shape), r"")

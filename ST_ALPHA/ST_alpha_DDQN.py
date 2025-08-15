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
        terminal_frac=0.2,
        exploration_rate=0.02,
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
        self.terminal_frac = terminal_frac  # fraction of terminal samples in mini-batch
        self.exploration_rate = exploration_rate

        self.__initialize_NNs__()

        self.S = []
        self.q = []
        self.X = []
        self.alpha = []
        self.r = []
        self.epsilon = []

        self.Q_loss = []
        self.Q_value_lst = []
        self.Q_target_lst = []

        self.tau = tau

    def __str__(self):
        return f"ST_alpha_DDQN(gamma={self.gamma}, n_nodes={self.n_nodes}, n_layers={self.n_layers}, lr={self.lr}, sched_step_size={self.sched_step_size}, tau={self.tau})"

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
                alpha.unsqueeze(-1) / 0.02,
                # X.unsqueeze(-1), #TODO：try normalize in network
                q.unsqueeze(-1) / self.Nq,
            ),
            axis=-1,
        ).float()

    def __grab_mini_batch__(self, mini_batch_size, terminal_frac=0.2):
        """
        Grab a mini-batch of data from the environment.
        Args:
            mini_batch_size (int): Size of the mini-batch.
        Returns:
            tuple: A tuple containing the time, state, cash, alpha, and inventory tensors.
        """
        # t is relative time from end
        t = torch.randint(0, self.env.Ndt, (mini_batch_size,))

        # number of terminal samples
        k = int(terminal_frac * mini_batch_size)

        if k > 0:
            half_k = k // 2
            # force some samples to Ndt-1
            t[-half_k:] = self.env.Ndt - 1
            # force some samples to Ndt-2
            t[-k:-half_k] = self.env.Ndt - 2
        S, q, X, alpha = self.env.Randomize_Start(t, mini_batch_size)
        # TODO: plot these to check these
        return t, S, q, X, alpha



    def update_Q(
        self, n_iter=10, mini_batch_size=256, exploration_rate=0.02
    ):  # TODO: make it time update not random
        for i in range(n_iter):

            t, S, q, X, alpha = self.__grab_mini_batch__(mini_batch_size, terminal_frac=self.terminal_frac)  # should oversample t=9 but with decay

            self.Q_main["optimizer"].zero_grad()

            # concatenate states
            state = self.__stack_state__(t=t, S=S, q=q, X=X, alpha=alpha)

            Q = self.Q_main["net"](state)  # (batch_size, n_actions)

            # --- Epsilon-greedy action selection ---
            batch_size = Q.shape[0]
            rand_vals = torch.rand(batch_size)
            random_actions = torch.randint(0, Q.shape[1], (batch_size,))
            greedy_actions = Q.argmax(dim=1)
            # With probability exploration_rate choose random, else greedy
            actions = torch.where(rand_vals < exploration_rate, random_actions, greedy_actions)

            # Gather Q-value for chosen action
            Q_value = Q.gather(1, actions.unsqueeze(1)).squeeze(1)

            # Step in environment to get next state and reward
            t_p, S_p, X_p, alpha_p, q_p, r, isMO, buySellMO, done = self.env.step(
                t, S, X, alpha, q, actions
            )
            # New state
            state_p = self.__stack_state__(t=t_p, S=S_p, X=X_p, alpha=alpha_p, q=q_p)
            Q_p = self.Q_main["net"](state_p)
            # Next greedy action for Double DQN
            next_greedy_actions = Q_p.argmax(dim=1, keepdim=True)
            # Target value using target net
            done_mask = (1 - done.float())   # 1 if not terminal, 0 if terminal
            target_q_values = Q_p.gather(1, next_greedy_actions).squeeze(1) * done_mask
            # Compute target
            target = r + self.env.gamma * target_q_values
            target = target.detach()
            huber = torch.nn.SmoothL1Loss()
            loss = huber(Q_value, target.detach()) 
            # Loss
            # loss = torch.mean((Q_value - target) ** 2)
            loss.backward()
            self.Q_main["optimizer"].step()
            self.Q_main["scheduler"].step()
            self.Q_loss.append(loss.item())
            self.Q_value_lst.append(Q_value.detach().numpy())
            self.Q_target_lst.append(target.detach().numpy())
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
                n_iter=n_iter_Q, mini_batch_size=mini_batch_size, exploration_rate=self.exploration_rate
            )

            if np.mod(i + 1, n_plot) == 0:

                self.loss_plots()
                self.run_strategy(1_000, name=datetime.now().strftime("%H_%M_%S"))
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
        plt.subplot(1, 3, 1)
        plot(self.Q_loss, r"$Q$", show_band=False)
        plt.subplot(1, 3, 2)
        plot(self.Q_value_lst, r"$Q_{value}$", show_band=False)
        plt.subplot(1, 3, 3)
        plot(self.Q_target_lst, r"$Q_{target}$", show_band=False)

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
        with torch.no_grad():
            for step in range(N):

                state = self.__stack_state__(
                    t=time[:, step],
                    S=S[:, step],
                    q=q[:, step],
                    X=X[:, step],
                    alpha=alpha[:, step],
                )
                
                Q = self.Q_main["net"](state)
                action[:, step] = Q.argmax(dim=1)
                (
                    time[:, step + 1],
                    S[:, step + 1],
                    X[:, step + 1],
                    alpha[:, step + 1],
                    q[:, step + 1],
                    r[:, step],
                    isMO[:, step],
                    buySellMO[:, step],
                    done
                ) = self.env.step(
                    time[:, step],
                    S[:, step],
                    X[:, step],
                    alpha[:, step],
                    q[:, step],
                    action[:, step],
                )

        # # Clear position at the end of the simulation
        # X[:, X.shape[1] - 1] += np.multiply(
        #     S[:, S.shape[1] - 1]
        #     + (0.5 * self.env.Delta) * np.sign(q[:, q.shape[1] - 1])
        #     + self.env.varphi * q[:, q.shape[1] - 1],
        #     q[:, q.shape[1] - 1],
        # )
        # q[:, q.shape[1] - 1] = 0

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

        plt.suptitle(
            f"Simulation - {self}  \n  {self.env} \n {self.env.ShortTermalpha}",
            fontsize=12,
            y=1.02,
        )
        plot(t, (S), 1, r"$S_t$")
        plot(t, alpha, 2, r"$\alpha_t$")
        # plot(t[1:], q[:, 1:] - q[:, :-1], 2, r"$q_t - q_{t-1}$")
        plot(t, q, 3, r"$q_t$")

        plot(t[:-1], np.cumsum(r, axis=1), 4, r"$r_t$")
        plot(t, X + S * q, 5, r"$Wealth$")

        plt.subplot(2, 3, 6)
        plt.hist(X[:, -1]+S[:, -1]*q[:, -1], bins=51)
        plt.title("Terminal Wealth")
        plt.xlabel("Wealth")
        plt.ylabel("Frequency")

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

    def plot_policy_t(self, t=0.5, name=""):
        """Plots the policy as a single heatmap showing the action with the highest probability."""
        num_alpha_points = 51
        num_inventory_points = 51
        alpha_values = torch.linspace(-0.02, 0.02, num_alpha_points)
        inventory_levels = torch.linspace(-self.Nq, self.Nq, num_inventory_points)

        max_prob_action = np.zeros((num_inventory_points, num_alpha_points))
        max_prob_value = np.zeros((num_inventory_points, num_alpha_points))
        with torch.no_grad():
            for i, q in enumerate(inventory_levels):
                for j, alpha in enumerate(alpha_values):
                    state = self.__stack_state__(
                        t=t * torch.ones(1),
                        S=self.env.S_0 * torch.ones(1),
                        X=torch.zeros(1),
                        alpha=torch.tensor([alpha]),
                        q=torch.tensor([q]),
                    )
                    policy_output = (
                        self.Q_main["net"](state)
                        .argmax(dim=1, keepdim=False)
                        .squeeze()
                        .numpy()
                    )
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
        plt.suptitle(f"Policy - {self}  \n  {self.env} \n {self.env.ShortTermalpha}", fontsize=12, y=1.02)
        plt.title("Policy Heatmap - Action", fontsize=16)
        plt.xlabel(r"$\alpha$", fontsize=14)
        plt.ylabel("Inventory", fontsize=14)
        plt.tight_layout()
        plt.show()

    def plot_policy(self, name=""):
        """Plots policy heatmaps at t = 0.1, 0.5, 0.9 of the episode."""
        num_alpha_points = 51
        num_inventory_points = 51
        alpha_values = torch.linspace(-0.02, 0.02, num_alpha_points)
        inventory_levels = torch.linspace(-self.Nq, self.Nq, num_inventory_points)

        # Time points: 10%, 50%, 90% of episode
        time_fractions = [0.1, 0.5, 0.9]
        time_points = [f * self.env.Ndt for f in time_fractions]

        fig, axs = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

        for idx, t in enumerate(time_points):
            max_prob_action = np.zeros((num_inventory_points, num_alpha_points))

            with torch.no_grad():
                for i, q in enumerate(inventory_levels):
                    for j, alpha in enumerate(alpha_values):
                        state = self.__stack_state__(
                            t=torch.tensor([t]),
                            S=self.env.S_0 * torch.ones(1),
                            X=torch.zeros(1),
                            alpha=torch.tensor([alpha]),
                            q=torch.tensor([q]),
                        )
                        policy_output = (
                            self.Q_main["net"](state)
                            .argmax(dim=1, keepdim=False)
                            .squeeze()
                            .item()
                        )
                        max_prob_action[i, j] = policy_output

            ax = axs[idx]
            ctf = ax.contourf(
                alpha_values.numpy(),
                inventory_levels.numpy(),
                max_prob_action,
                levels=np.arange(-0.5, 4, 1),
                cmap="viridis",
                alpha=0.8,
            )
            ax.axhline(0, linestyle="--", color="k", linewidth=0.8)
            ax.axvline(0, linestyle="--", color="k", linewidth=0.8)
            ax.set_title(f"t = {time_fractions[idx]:.1f} × Ndt", fontsize=14)
            ax.set_xlabel(r"$\alpha$")
            if idx == 0:
                ax.set_ylabel("Inventory")

        fig.suptitle(f"Policy Heatmaps at Different Times\n{self.env}", fontsize=16)
        fig.colorbar(
            ctf,
            ax=axs,
            orientation="vertical",
            fraction=0.015,
            pad=0.04,
            ticks=range(4),
            label="Action",
        )
        plt.show()
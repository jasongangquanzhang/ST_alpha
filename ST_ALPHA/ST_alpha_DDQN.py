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


class DDQN(nn.Module):
    def __init__(
        self,
        n_in=2,
        n_out=4,
        nNodes=24,
        nLayers=4,
        activation="relu",
        normalization=False,
    ):
        super(DDQN, self).__init__()
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


class DDQNAgent:

    def __init__(
        self,
        env: Environment,
        gamma=0.99,
        n_nodes=36,
        n_layers=6,
        lr=1e-3,
        sched_step_size=100,
        sched_gamma=0.99,
        opt_weight_decay=0.0001,
        tau=0.01,
        name="",
    ):

        self.env = env
        self.gamma = gamma
        self.n_nodes = n_nodes
        self.n_layers = n_layers
        self.name = name
        self.sched_step_size = sched_step_size
        self.sched_gamma = sched_gamma
        self.opt_weight_decay = opt_weight_decay
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
        self.lrs = []
        self.epsilons = []

        self.tau = tau

    def __initialize_NNs__(self):
        """
        Initializes the main and target Q-networks.

        Creates the main Q-network (DDQN) with input size 3 (t, alpha, q),
        output size 4 (actions), and hidden structure defined by
        n_nodes and n_layers. Also sets up the optimizer and scheduler,
        and makes a copy of the main network as the target network.
        """
        self.Q_main = {
            "net": DDQN(n_in=3, n_out=4, nNodes=self.n_nodes, nLayers=self.n_layers)
        }

        self.Q_main["optimizer"], self.Q_main["scheduler"] = self.__get_optim_sched__(
            self.Q_main
        )

        self.Q_target = copy.deepcopy(self.Q_main)

    def __get_optim_sched__(self, net):
        """
        Sets up the optimizer and learning rate scheduler for the given network.

        Args:
            net (dict): The network dictionary containing the 'net' key.

        Returns:
            tuple: A tuple containing the optimizer and scheduler.
        """
        optimizer = optim.AdamW(net["net"].parameters(), lr=self.lr, weight_decay=self.opt_weight_decay)

        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=self.sched_step_size, gamma=self.sched_gamma
        )

        return optimizer, scheduler

    def __stack_state__(self, t, alpha, q):
        """
        Stack the state variables into a single tensor after applying normalization.

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
                alpha.unsqueeze(-1)/0.02,
                q.unsqueeze(-1) / self.Nq,
            ),
            dim=-1,
        ).float()

    def __grab_mini_batch__(self, mini_batch_size, terminal_frac=0.2):
        """
        Grabs a mini-batch of experiences from the environment.

        Args:
            mini_batch_size (int): The size of the mini-batch to sample.
            terminal_frac (float): The fraction of terminal states to include.

        Returns:
            tuple: A tuple containing the sampled time steps, states, actions, and rewards.
        """
        t = torch.randint(0, self.env.Ndt, (mini_batch_size,))
        k = int(terminal_frac * mini_batch_size)
        if k > 0:
            half_k = k // 2
            t[-half_k:] = self.env.Ndt - 1
            t[-k:-half_k] = self.env.Ndt - 2
        S, q, X, alpha = self.env.Randomize_Start(t, mini_batch_size)
        return t, S, q, X, alpha

    def update_Q(self, n_iter_Q=10, mini_batch_size=256, epsilon=0.02):
        """
        Updates the Q-network by performing a number of training iterations.

        Args:
            n_iter_Q (int): The number of iterations to perform per epsilon value.
            mini_batch_size (int): The size of the mini-batch to sample.
            epsilon (float): The exploration rate for epsilon-greedy action selection.
        """
        for i in range(n_iter_Q):

            t, S, q, X, alpha = self.__grab_mini_batch__(mini_batch_size)

            self.Q_main["optimizer"].zero_grad()

            # concatenate states
            state = self.__stack_state__(t=t, alpha=alpha, q=q)

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
            t_p, S_p, X_p, alpha_p, q_p, r, isMO, buySellMO, done = self.env.step(
                t, S, X, alpha, q, actions
            )
            # New state
            state_p = self.__stack_state__(
                t=t_p, alpha=alpha_p, q=q_p
            )
            Q_p = self.Q_main["net"](state_p)
            # Next greedy action for Double DQN
            next_greedy_actions = Q_p.argmax(dim=1, keepdim=True)
            # Target value using target net
            target_q_values = (
                self.Q_target["net"](state_p).gather(1, next_greedy_actions).squeeze(1)
            )
            # Zero out target_q_values for done states
            target_q_values[done] = 0

            # Compute target
            target = r + self.gamma * target_q_values
            target = target.detach()

            # Loss
            loss = torch.mean((Q_value - target) ** 2)
            loss.backward()
            self.Q_main["optimizer"].step()
            self.Q_main["scheduler"].step()

            self.Q_loss.append(loss.item())
            self.lrs.append(self.Q_main["optimizer"].param_groups[0]["lr"])
            self.epsilons.append(epsilon)

            # Target network soft update
            self.soft_update(self.Q_main["net"], self.Q_target["net"])

    def soft_update(self, main, target):
        """
        Soft updates the target network parameters towards the main network parameters using Polyak averaging.
        """
        for param, target_param in zip(main.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def train(self, n_iter=1_000, n_iter_Q=10, mini_batch_size=256, n_plot=100):
        """
        Trains the DDQN agent by performing a number of iterations, while plotting the loss,
        simulating a trading environment with current policy, and showing the policy itself.

        Args:
            n_iter (int): The number of iterations to perform.
            n_iter_Q (int): The number of iterations to perform per epsilon value.
            mini_batch_size (int): The size of the mini-batch to sample.
            n_plot (int): The number of iterations between each plot.
        """
        self.run_strategy(
            nsims=1_000, name=datetime.now().strftime("%H_%M_%S")
        )  # intital evaluation

        C = 500
        D = 1000

        if len(self.epsilon) == 0:
            self.count = 0

        for i in tqdm(range(n_iter)):
        # for i in range(n_iter):

            epsilon = np.maximum(C / (D + self.count), 0.02)
            self.epsilon.append(epsilon)
            self.count += 1

            # pdb.set_trace()

            self.update_Q(
                n_iter_Q=n_iter_Q, mini_batch_size=mini_batch_size, epsilon=epsilon
            )

            if np.mod(i + 1, n_plot) == 0:
                print(f"Iteration {i}, epsilon: {epsilon}")
                self.loss_plots()
                self.run_strategy(
                    1_000, name=datetime.now().strftime("%H_%M_%S")
                )
                self.plot_policy()
                # self.plot_policy(name=datetime.now().strftime("%H_%M_%S"))
        
        # Combined figure with three subplots: loss, lr, and epsilon
        fig, axs = plt.subplots(1, 3, figsize=(8, 6), sharex=False)

        # Left subplot: Loss
        axs[0].plot(self.Q_loss, color='tab:blue')
        axs[0].set_ylabel("Loss")
        axs[0].set_title("Training Loss over Time")
        axs[0].set_yscale("symlog")
        axs[0].grid(True)

        # Middle subplot: Learning Rate
        axs[1].plot(self.lrs, color='tab:orange')
        axs[1].set_xlabel("Training Steps")
        axs[1].set_ylabel("Learning Rate")
        axs[1].set_title("Learning Rate over Time")
        axs[1].grid(True)

        # Right subplot: Epsilon
        axs[2].plot(self.epsilons, color='tab:green')
        axs[2].set_xlabel("Training Steps")
        axs[2].set_ylabel("Epsilon")
        axs[2].set_title("Epsilon over Time")
        axs[2].grid(True)

        plt.tight_layout()
        plt.show()

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
            if N is None:
                N = self.env.Ndt
            time = torch.zeros((nsims, N + 1)).float()
            S = torch.zeros((nsims, N + 1)).float()
            q = torch.zeros((nsims, N + 1)).float()
            X = torch.zeros((nsims, N + 1)).float()
            alpha = torch.zeros((nsims, N + 1)).float()
            action = torch.zeros((nsims, N)).float()
            r = torch.zeros((nsims, N)).float()
            S[:, 0], q[:, 0], X[:, 0], alpha[:, 0] = self.env.Zero_Start(nsims)
            for step in range(N):
                state = self.__stack_state__(
                    t=time[:, step], alpha=alpha[:, step], q=q[:, step]
                )
                with torch.no_grad():
                    probs = self.Q_main["net"](state)
                    action[:, step] = probs.argmax(dim=1, keepdim=False)
                (
                    time[:, step + 1],
                    S[:, step + 1],
                    X[:, step + 1],
                    alpha[:, step + 1],
                    q[:, step + 1],
                    r[:, step],
                    _,
                    _,
                    done,
                ) = self.env.step(
                    t=time[:, step],
                    S=S[:, step],
                    X=X[:, step],
                    alpha=alpha[:, step],
                    q=q[:, step],
                    action=action[:, step],
                )
                if done.all():
                    break
            # --- Convert to numpy ---
            time = time.numpy()
            S = S.numpy()
            q = q.numpy()
            X = X.numpy()
            r = r.numpy()
            t_axis = np.arange(0, N + 1)
            plt.figure(figsize=(20, 10))
            n_paths = 1
            def plot(t, x, idx, title):
                qtl = np.quantile(x, [0.05, 0.5, 0.95], axis=0)
                plt.subplot(2, 3, idx)
                plt.fill_between(t, qtl[0, :], qtl[2, :], color="lightblue", alpha=0.5)
                plt.plot(t, qtl[1, :], color="black", linewidth=1)
                plt.plot(t, x[:n_paths, :].T, linewidth=1)
                plt.title(label=title, fontdict={"fontsize": 20, "weight": "bold"})
                plt.xlabel(r"$t$", fontsize=20, fontweight="bold")
                plt.yticks(fontsize=16, fontweight="bold")
                plt.xticks(fontsize=16, fontweight="bold")
            plot(t_axis, S, 1, r"$S_t$")
            plot(t_axis, alpha, 4, r"$\alpha_t$")
            plot(t_axis, q, 3, r"$q_t$")
            plot(t_axis[:-1], np.cumsum(r, axis=1), 2, r"$r_t$")
            plot(t_axis, X + S * q, 5, r"Wealth_Path")
            plt.suptitle(f"Simulation - DDQN \n {self.env}\n {self}", fontsize=12, y=1.02)
            plt.subplot(2, 3, 6)
            plt.hist(X[:, -1] + S[:, -1] * q[:, -1], bins=51, alpha=0.7)
            terminal_wealth = X[:, -1] + S[:, -1] * q[:, -1]
            qtl=[0.01, 0.10, 0.50, 0.90, 0.99]
            quantiles = np.quantile(terminal_wealth, qtl)
            colors = ["blue", "green", "orange", "purple", "brown"]
            for q, label, color in zip(quantiles, qtl, colors):
                plt.axvline(q, color=color, linestyle="--", linewidth=1.5, label=f"qtl={label}")
            plt.title("Terminal Wealth", fontdict={"fontsize": 20, "fontweight": "bold"})
            plt.xlabel("Wealth", fontsize=20, fontweight="bold")
            plt.ylabel("Frequency", fontsize=20, fontweight="bold")
            plt.yticks(fontsize=16, fontweight="bold")
            plt.legend(fontsize=14)
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
                # ax.set_title(f"t = {time_fractions[idx]:.1f} × Ndt", fontsize=24, fontweight="bold")
                ax.set_xticks([-0.02,-0.01, 0.00, 0.01, 0.02])
                ax.set_xticklabels(["-0.02", "-0.01", "0.00", "0.01", "0.02"], fontsize=20, fontweight="bold")
                ax.tick_params(axis="x", which="major", pad=10)
                ax.set_yticks([-20,-10, 0, 10, 20])
                ax.set_yticklabels(["-20", "-10", "0", "10", "20"], fontsize=20, fontweight="bold")
                for label in ax.get_xticklabels():
                    label.set_fontweight("bold")
                # ax.set_xlabel(r”$\alpha$", fontsize=16, fontweight="bold”)
                if idx == 0:
                    # ax.set_ylabel("Inventory”, fontsize=16, fontweight="bold”)
                    for label in ax.get_yticklabels():
                        label.set_fontweight("bold")
                else:
                    ax.set_yticks([])  # remove y tick values
                # ax.set_xticks([])  # remove x tick values
            # cbar = fig.colorbar(
            #     ctf, ax=axs, orientation="vertical”, fraction=0.02, pad=0.04, ticks=range(4)
            # )
            # cbar.ax.set_yticklabels(["Do Nothing”, "Buy”, "Sell”, "Buy/Sell”])
            # cbar.set_label("Action with Max Probability”)
            plt.show()


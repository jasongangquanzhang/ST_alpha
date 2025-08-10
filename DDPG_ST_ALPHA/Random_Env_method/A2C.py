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
        temperature=1,
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


class A2CAgent:
    def __init__(
        self,
        env: Environment,
        n_nodes=128,
        n_layers=6,
        scheduler_gamma=0.99,
        sched_step_size=1000,
        lr_actor=1e-3,
        lr_critic=1e-3,
        entropy_coef=0.02,
        terminal_frac=0.4,
    ):
        self.env = env
        self.scheduler_gamma = scheduler_gamma
        self.sched_step_size = sched_step_size
        self.entropy_coef = entropy_coef
        self.Nq = env.Nq
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.terminal_frac = terminal_frac
        # --- Actor ---
        self.actor = ANN(
            n_in=3,  # (t, alpha, q)
            n_out=4,
            nNodes=n_nodes,
            nLayers=n_layers,
            out_activation="softmax",
        )
        self.actor_optim = optim.AdamW(
            self.actor.parameters(), lr=lr_actor, weight_decay=1e-4
        )
        self.actor_sched = optim.lr_scheduler.StepLR(
            self.actor_optim, step_size=self.sched_step_size, gamma=self.scheduler_gamma
        )
        # --- Critic ---
        self.critic = ANN(n_in=3, n_out=1, nNodes=n_nodes, nLayers=n_layers)
        self.critic_optim = optim.AdamW(
            self.critic.parameters(), lr=lr_critic, weight_decay=1e-4
        )
        self.critic_sched = optim.lr_scheduler.StepLR(
            self.critic_optim,
            step_size=self.sched_step_size,
            gamma=self.scheduler_gamma,
        )
        self.actor_loss_hist = []
        self.advantage_hist = []
        self.critic_loss_hist = []
        self.terminal_frac_lst = []

    def __str__(self):
        return f"A2C Agent with lr_actor: {self.lr_actor}, lr_critic: {self.lr_critic}, Entropy Coefficient: {self.entropy_coef}, Scheduler Gamma: {self.scheduler_gamma}, Scheduler Step Size: {self.sched_step_size}, Terminal Fraction: {self.terminal_frac}"

    def __stack_state__(self, t, alpha, q):
        return torch.cat(
            (
                t.unsqueeze(-1) / self.env.Ndt,
                alpha.unsqueeze(-1) / 0.02,
                q.unsqueeze(-1) / self.env.Nq,
            ),
            dim=-1,
        ).float()

    def __grab_mini_batch__(self, mini_batch_size, terminal_frac=0.2):
        t = torch.randint(0, self.env.Ndt, (mini_batch_size,))
        k = int(terminal_frac * mini_batch_size)
        if k > 0:
            half_k = k // 2
            t[-half_k:] = self.env.Ndt - 1
            t[-k:-half_k] = self.env.Ndt - 2

        S, q, X, alpha = self.env.Randomize_Start(t, mini_batch_size)
        return t, S, q, X, alpha

    def update(self, n_iter=1, mini_batch_size=256, terminal_frac=0.2):
        for _ in range(n_iter):
            # --- Sample states ---
            t, S, q, X, alpha = self.__grab_mini_batch__(mini_batch_size, terminal_frac)
            state = self.__stack_state__(t, alpha, q)

            # --- Actor forward ---
            probs = self.actor(state)
            action = torch.multinomial(probs, num_samples=1)
            log_probs = torch.log(probs + 1e-8).gather(1, action)

            # --- Critic forward ---
            V = self.critic(state)

            # --- Environment step ---
            t_p, S_p, X_p, alpha_p, q_p, r, _, _, done = self.env.step(
                t, S, X, alpha, q, action.squeeze(1)
            )
            next_state = self.__stack_state__(t_p, alpha_p, q_p)

            with torch.no_grad():
                V_next = self.critic(next_state)
                target = (
                    r.unsqueeze(-1)
                    + (1 - done.float().unsqueeze(-1)) * self.env.gamma * V_next
                )

            # --- Losses ---
            adv_actor = target - V
            adv_norm = (adv_actor - adv_actor.mean()) / (adv_actor.std() + 1e-8)

            # Actor loss
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
            actor_loss = (
                -(log_probs * adv_norm.detach()).mean() - self.entropy_coef * entropy
            )

            # Critic loss (true TD error, no normalization)
            critic_loss = (adv_actor**2).mean()

            # --- Backprop ---
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            self.actor_sched.step()

            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()
            self.critic_sched.step()

            # --- Log ---
            self.actor_loss_hist.append(actor_loss.item())
            self.critic_loss_hist.append(critic_loss.item())
            self.advantage_hist.append(adv_norm.mean().item())

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

        fig = plt.figure(figsize=(10, 4))
        plt.subplot(1, 3, 1)
        plot(self.advantage_hist, r"$Advantage$")

        plt.subplot(1, 3, 2)
        plot(self.actor_loss_hist, r"actor_loss")
        plt.subplot(1, 3, 3)
        plot(self.critic_loss_hist, r"critic_loss")

        plt.tight_layout()
        plt.show()

    def train(self, n_iter=1_000, n_iter_net=5, mini_batch_size=256, n_plot=100):
        self.run_strategy(
            nsims=1_000, name=datetime.now().strftime("%H_%M_%S")
        )  # initial evaluation

        C = 2000
        D = 4000

        if len(self.terminal_frac_lst) == 0:
            self.count = 0

        for i in range(n_iter):
            terminal_frac = max(self.terminal_frac * (0.99 ** (self.count / 100)), 0.1)
            self.terminal_frac_lst.append(terminal_frac)

            self.count += 1

            self.update(
                n_iter=n_iter_net,
                mini_batch_size=mini_batch_size,
                terminal_frac=terminal_frac,
            )

            if np.mod(i + 1, n_plot) == 0:
                self.loss_plots()
                self.run_strategy(1_000, name=datetime.now().strftime("%H_%M_%S"))
                self.plot_policy()

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
                probs = self.actor(state)
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

        t_axis = self.env.dt * np.arange(0, N + 1) / self.env.T

        plt.figure(figsize=(10, 10))
        n_paths = 2

        def plot(t, x, idx, title):
            qtl = np.quantile(x, [0.05, 0.5, 0.95], axis=0)
            plt.subplot(2, 3, idx)
            plt.fill_between(t, qtl[0, :], qtl[2, :], alpha=0.5)
            plt.plot(t, qtl[1, :], color="k", linewidth=1)
            plt.plot(t, x[:n_paths, :].T, linewidth=1)
            plt.title(title)
            plt.xlabel(r"$t$")

        plot(t_axis, S, 1, r"$S_t$")
        plot(t_axis, alpha, 2, r"$\alpha_t$")
        plot(t_axis, q, 3, r"$q_t$")
        plot(t_axis[:-1], np.cumsum(r, axis=1), 4, r"$r_t$")
        plot(t_axis, X + S * q, 5, r"Wealth")

        plt.suptitle(f"Simulation - A2C \n {self.env}\n {self}", fontsize=12, y=1.02)
        plt.subplot(2, 3, 6)
        plt.hist(X[:, -1] + S[:, -1] * q[:, -1], bins=51)
        plt.title("Terminal Wealth")
        plt.xlabel("Wealth")
        plt.ylabel("Frequency")

        plt.tight_layout()
        plt.show()

    def plot_policy(self):
        num_alpha_points = 51
        num_inventory_points = 51
        alpha_values = torch.linspace(-0.02, 0.02, num_alpha_points)
        inventory_levels = torch.linspace(-self.Nq, self.Nq, num_inventory_points)

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
                        policy_output = self.actor(state).squeeze().numpy()
                        max_prob_action[i, j] = np.argmax(policy_output)

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

        cbar = fig.colorbar(
            ctf, ax=axs, orientation="vertical", fraction=0.02, pad=0.04, ticks=range(4)
        )
        cbar.ax.set_yticklabels(["Do Nothing", "Buy", "Sell", "Buy/Sell"])
        cbar.set_label("Action with Max Probability")
        plt.show()

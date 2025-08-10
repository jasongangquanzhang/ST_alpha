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
        env,
        n_nodes=64,
        n_layers=4,
        gamma=0.99,
        lr_actor=1e-3,
        lr_critic=1e-3,
        tau=0.01,
        entropy_coef=0.01,
    ):
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.entropy_coef = entropy_coef
        self.Nq = env.Nq

        # --- policy (actor) ---
        self.actor = ANN(
            n_in=2, n_out=4, nNodes=n_nodes, nLayers=n_layers, out_activation="softmax"
        )
        self.actor_optim = optim.AdamW(self.actor.parameters(), lr=lr_actor)

        # --- value function (critic) ---
        self.critic = ANN(n_in=2, n_out=1, nNodes=n_nodes, nLayers=n_layers)
        self.critic_optim = optim.AdamW(self.critic.parameters(), lr=lr_critic)

        self.actor_loss_hist = []
        self.critic_loss_hist = []
        self.terminal_frac_lst = []

    def __stack_state__(self, t, q):
        return torch.cat(
            (t.unsqueeze(-1) / self.env.Ndt, q.unsqueeze(-1) / self.env.Nq), dim=-1
        ).float()

    def __grab_mini_batch__(self, mini_batch_size, terminal_frac=0.2):
        """
        Grab a minibatch of states.
        Oversample terminal steps with probability = terminal_frac.
        
        Args:
            mini_batch_size (int): batch size
            terminal_frac (float): fraction of samples to force at t = Ndt-1
        Returns:
            (t, S, q, X)
        """
        # sample uniformly from [0, Ndt)
        t = torch.randint(0, self.env.Ndt, (mini_batch_size,))

        # number of terminal samples
        k = int(terminal_frac * mini_batch_size)

        if k > 0:
            half_k = k // 2
            # force some samples to Ndt-1
            t[-half_k:] = self.env.Ndt - 1
            # force some samples to Ndt-2
            t[-k:-half_k] = self.env.Ndt - 2

        # sample states
        S, q, X = self.env.Randomize_Start(t, mini_batch_size)
        return t, S, q, X

    def update(self, n_iter=1, mini_batch_size=256, terminal_frac=0.2):
        for _ in range(n_iter):
            # sample states
            t, S, q, X = self.__grab_mini_batch__(mini_batch_size, terminal_frac)
            state = self.__stack_state__(t, q)

            # policy distribution
            probs = self.actor(state)
            action = torch.multinomial(probs, num_samples=1)
            log_probs = torch.log(probs + 1e-8).gather(1, action)

            # critic value
            V = self.critic(state)

            # environment step
            t_p, S_p, X_p, q_p, r, _, _, done = self.env.step(
                t, S, X, q, action.squeeze(1)
            )
            next_state = self.__stack_state__(t_p, q_p)
            with torch.no_grad():
                V_next = self.critic(next_state)
                target = (
                    r.unsqueeze(-1)*100
                    + (1 - done.float().unsqueeze(-1)) * self.gamma * V_next
                )

            # advantage
            advantage = target - V
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            # --- actor loss ---
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
            actor_loss = (
                -(log_probs * advantage.detach()).mean() - self.entropy_coef * entropy
            )

            # --- critic loss ---
            critic_loss = (advantage**2).mean()

            # optimize actor
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            # optimize critic
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

            # store
            self.actor_loss_hist.append(actor_loss.item())
            self.critic_loss_hist.append(critic_loss.item())

    def plot_losses(self):
        plt.plot(self.actor_loss_hist, label="Actor Loss")
        plt.plot(self.critic_loss_hist, label="Critic Loss")
        plt.legend()
        plt.show()

    def train(
        self, n_iter=1_000, n_iter_Q=10, n_iter_pi=5, mini_batch_size=256, n_plot=100
    ):

        self.run_strategy(
            nsims=1_000, name=datetime.now().strftime("%H_%M_%S")
        )  # intital evaluation

        C = 2000
        D = 4000

        if len(self.terminal_frac_lst) == 0:
            self.count = 0

        # for i in tqdm(range(n_iter)):
        for i in range(n_iter):

            terminal_frac = np.maximum(C / (D + self.count), 0.1)
            self.terminal_frac_lst.append(terminal_frac)

            self.count += 1

            # pdb.set_trace()

            self.update(
                n_iter=n_iter_pi,
                mini_batch_size=mini_batch_size,
                terminal_frac=terminal_frac,
            )
            if np.mod(i + 1, n_plot) == 0:

                self.plot_losses()
                self.run_strategy(1_000, name=datetime.now().strftime("%H_%M_%S"))
                self.plot_policy()
                # self.plot_policy(name=datetime.now().strftime("%H_%M_%S"))

    def run_strategy(self, nsims: int = 10_000, name: str = "", N: int = None):
        """Run the trading strategy simulation with the current actor (policy network).
        This evaluates the learned policy by simulating environment rollouts.
        """

        if N is None:
            N = self.env.Ndt  # total time steps

        time = torch.zeros((nsims, N + 1)).float()
        S = torch.zeros((nsims, N + 1)).float()
        q = torch.zeros((nsims, N + 1)).float()
        X = torch.zeros((nsims, N + 1)).float()
        action = torch.zeros((nsims, N)).float()
        r = torch.zeros((nsims, N)).float()
        isMO = torch.zeros((nsims, N)).float()
        buySellMO = torch.zeros((nsims, N)).float()

        # --- initial state ---
        S[:, 0], q[:, 0], X[:, 0] = self.env.Zero_Start(nsims)

        for step in range(N):
            # build state (only t and q)
            state = self.__stack_state__(t=time[:, step], q=q[:, step])

            # sample action from policy
            with torch.no_grad():
                probs = self.actor(state)  # (nsims, 4)
                action[:, step] = torch.multinomial(probs, num_samples=1).squeeze(1)

            # environment step
            (
                time[:, step + 1],
                S[:, step + 1],
                X[:, step + 1],
                q[:, step + 1],
                r[:, step],
                isMO[:, step],
                buySellMO[:, step],
                done,
            ) = self.env.step(
                t=time[:, step],
                S=S[:, step],
                X=X[:, step],
                q=q[:, step],
                action=action[:, step],
            )

            if done.all():
                break

        # --- convert to numpy ---
        time = time.numpy()
        S = S.numpy()
        q = q.numpy()
        X = X.numpy()
        r = r.numpy()
        action = action.numpy()

        t = self.env.dt * np.arange(0, N + 1) / self.env.T

        # --- plots ---
        plt.figure(figsize=(10, 10))
        n_paths = 10

        def plot(t, x, idx, title):
            qtl = np.quantile(x, [0.05, 0.5, 0.95], axis=0)
            plt.subplot(2, 3, idx)
            plt.fill_between(t, qtl[0, :], qtl[2, :], alpha=0.5)
            plt.plot(t, qtl[1, :], color="k", linewidth=1)
            plt.plot(t, x[:n_paths, :].T, linewidth=1)
            plt.title(title)
            plt.xlabel(r"$t$")

        plot(t, S, 1, r"$S_t$")
        plot(t, q, 2, r"$q_t$")
        plot(t[:-1], np.cumsum(r, axis=1), 3, r"$r_t$")
        plot(t, X + S * q, 4, r"Wealth")

        plt.suptitle(f"Simulation - A2C \n {self.env}", fontsize=12, y=1.02)
        plt.subplot(2, 3, 6)
        plt.hist(X[:, -1] + S[:, -1] * q[:, -1], bins=51)
        plt.title("Terminal Wealth")
        plt.xlabel("Wealth")
        plt.ylabel("Frequency")

        plt.tight_layout()
        plt.show()

    def plot_policy(self, name=""):
        """Plots the A2C policy as a heatmap of the action with the highest probability."""

        num_inventory_points = 51
        num_time_points = 51

        inventory_values = torch.linspace(-self.env.Nq, self.env.Nq, num_inventory_points)
        time_steps = torch.linspace(0, self.env.Ndt, num_time_points)

        max_prob_action = np.zeros((num_time_points, num_inventory_points))
        max_prob_value = np.zeros((num_time_points, num_inventory_points))

        with torch.no_grad():
            for i, q in enumerate(inventory_values):
                for j, t in enumerate(time_steps):
                    state = self.__stack_state__(
                        t=torch.tensor([t]),
                        q=torch.tensor([q]),
                    )
                    policy_output = self.actor(state).squeeze().numpy()
                    max_prob_action[i, j] = np.argmax(policy_output)
                    max_prob_value[i, j] = np.max(policy_output)

        plt.figure(figsize=(10, 8))
        plt.contourf(
            time_steps.numpy(),
            inventory_values.numpy(),
            max_prob_action,   # transpose so inventory is vertical axis
            levels=np.arange(-0.5, 4, 1),
            cmap="viridis",
            alpha=0.8,
        )
        cbar = plt.colorbar(ticks=range(4))
        cbar.ax.set_yticklabels(["Do Nothing", "Buy", "Sell", "Buy/Sell"])
        cbar.set_label("Action with Max Probability")
        plt.axhline(0, linestyle="--", color="k", linewidth=0.8)
        plt.axvline(0, linestyle="--", color="k", linewidth=0.8)
        plt.title("Policy Heatmap - Action with Max Probability", fontsize=16)
        plt.suptitle(f"Policy Heatmap - A2C \n {self.env}", fontsize=12, y=1.02)
        plt.xlabel(r"$t$", fontsize=14)
        plt.ylabel("Inventory", fontsize=14)
        plt.tight_layout()
        plt.show()

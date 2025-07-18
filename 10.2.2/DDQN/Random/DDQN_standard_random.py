# Random Method

from standard_env import StandardEnv

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import random

from tqdm import tqdm

class QNetwork(nn.Module):
    """
    A neural network for approximating the Q-value function using linear layers and ReLU activations (for non-output layer).
    """
    def __init__(self, state_size, action_size, n_nodes, n_hidden_layers):
        """
        Args:
            state_size (int): Size of the state space.
            action_size (int): Size of the action space.
            n_nodes (int): Number of nodes in each hidden layer.
            n_hidden_layers (int): Number of hidden layers in the network.
        """
        assert n_hidden_layers >= 1, "Number of hidden layers must be at least 1"
        super(QNetwork, self).__init__()
        self.hidden_layers= nn.ModuleList()
        self.hidden_layers.append(nn.Linear(state_size, n_nodes))
        for _ in range(n_hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(n_nodes, n_nodes))
        self.output_layer = nn.Linear(n_nodes, action_size)
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, state_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, action_size) representing Q-values for each action.
        """
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        return self.output_layer(x)


class ReplayBuffer:
    def __init__(self, capacity):
        """
        Args:
            capacity (int): Maximum number of experiences to store in the buffer.
                The buffer will overwrite old experiences when it reaches its capacity.
        """
        self.capacity = capacity
        self.buffer = []
        self.index = 0
    
    def __len__(self):
        return len(self.buffer)
    
    def push(self, state, action, reward, next_state):
        """
        Store a new experience in the replay buffer. If the buffer is full, it will overwrite the oldest experience.
        Args:
            state (torch.Tensor): Current state of shape (batch_size, state_size).
            action(torch.Tensor): Current action of shape (batch_size,).
            reward (torch.Tensor): Current reward of shape (batch_size,).
            next_state (torch.Tensor): Next state after taking the action, of shape (batch_size, state_size).
        """
        batch_size = state.shape[0]
        for i in range(batch_size):
            s = state[i]
            a = action[i]
            r = reward[i]
            ns = next_state[i]

            if (len(self.buffer) < self.capacity):
                self.buffer.append(None)
            self.buffer[self.index] = (s, a, r, ns)
            self.index = (self.index + 1) % self.capacity
    
    def sample(self, batch_size):
        """
        Sample a batch of experiences from the replay buffer.
        Args:
            batch_size (int): Number of experiences to sample.
        Returns:
            tuple: A tuple containing:
                - states (torch.Tensor): Batch of states of shape (batch_size, state_size).
                - actions (torch.Tensor): Batch of actions of shape (batch_size,).
                - rewards (torch.Tensor): Batch of rewards of shape (batch_size,).
                - next_states (torch.Tensor): Batch of next states of shape (batch_size, state_size).
        """
        random_indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        states, actions, rewards, next_states = [], [], [], []
        for i in random_indices:
            state, action, reward, next_state, = self.buffer[i]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)

        return (
            torch.stack(states),
            torch.tensor(actions),
            torch.tensor(rewards),
            torch.stack(next_states)
        )

class DDQNAgent:
    def __init__(self, env: StandardEnv, state_size, action_size, seed, lr=1e-3,
                weight_decay=1e-2, sched_step_size=1000, sched_gamma=0.99,
                tau=1e-3, update_every=4, sample_size=64, capacity=1000000, name='',
                n_nodes=16, n_hidden_layers=6):
        """
        Args:
            env (StandardEnv): Environment instance.
            state_size (int): Size of the state space.
            action_size (int): Size of the action space.
            seed (int): Random seed for reproducibility.
            lr (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay for the optimizer.
            sched_step_size (int): Step size for the learning rate scheduler.
            sched_gamma (float): Gamma value for the learning rate scheduler.
            tau (float): Soft update parameter for the target network.
            update_every (int): Number of steps between updates to the target network.
            sample_size (int): Size of the mini-batch taken from the replay buffer for training.
            capacity (int): Maximum size of the replay buffer.
            name (str): Name of the agent.
            n_nodes (int): Number of nodes in each hidden layer of the Q-network.
            n_hidden_layers (int): Number of hidden layers in the Q-network.
        """
        print("Entering DDQNAgent.__init__")
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.lr = lr
        self.tau = tau
        self.update_every = update_every
        self.sample_size = sample_size
        self.name = name

        self.steps = 0
        self.action_spaces = torch.tensor([0, 1, 2, 3], dtype=int)  # do nothing, buy, sell, buy and sell

        self.qnetwork_main = QNetwork(state_size, action_size, n_nodes, n_hidden_layers)
        self.qnetwork_target = QNetwork(state_size, action_size, n_nodes, n_hidden_layers)
        self.update_target_network()
        # self.optimizer = optim.Adam(self.qnetwork_main.parameters(), lr=self.lr)
        self.optimizer = optim.AdamW(self.qnetwork_main.parameters(), lr=self.lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                step_size = sched_step_size,
                                                gamma=sched_gamma)
        self.replay_buffer = ReplayBuffer(capacity)

        self.epsilons = []
        self.losses = []
        self.count = 0  # Counter for the number of episodes
        self.lrs = []   # List to store learning rates for plotting
    
    def step(self, state, action, reward, next_state):
        """
        Store the experience in the replay buffer and perform a learning step periodically.
        Args:
            state (torch.Tensor): Current state of shape (batch_size, state_size).
            action (int): Action taken in the current state.
            reward (float): Reward received after taking the action.
            next_state (torch.Tensor): Next state after taking the action, of shape (batch_size, state_size).
        """
        # Store the experience in the replay buffer
        self.replay_buffer.push(state, action, reward, next_state)

        # Learn every `update_every` steps
        self.steps += 1
        if self.steps % self.update_every == 0:
            if (len(self.replay_buffer) >= self.sample_size):
                experiences = self.replay_buffer.sample(self.sample_size)
                self.learn(experiences)
    
    def act(self, state, epsilon):
        """
        Select an action based on the current state and epsilon-greedy policy.
        Args:
            state (torch.Tensor): Current state of shape (batch_size, state_size).
            epsilon (float): Epsilon value for the epsilon-greedy policy.
        Returns:
            torch.Tensor: Selected action index (0: do nothing, 1: buy, 2: sell, 3: buy and sell).
        """
        # state = state.unsqueeze(0)
        self.qnetwork_main.eval()
        with torch.no_grad():
            action_values = self.qnetwork_main(state)
        self.qnetwork_main.train()

        # Epsilon-greedy action selection
        if random.random() > epsilon:
            return torch.argmax(action_values, dim=1)  # Greedy action
        else:
            return torch.randint(0, self.action_size, (state.shape[0],))   # Random action

    def learn(self, experiences):
        """
        Update the Q-network using the sampled experiences from the replay buffer.
        Args:
            experiences (tuple): A tuple containing:
                - states (torch.Tensor): Batch of states of shape (batch_size, state_size).
                - actions (torch.Tensor): Batch of actions of shape (batch_size,).
                - rewards (torch.Tensor): Batch of rewards of shape (batch_size,).
                - next_states (torch.Tensor): Batch of next states of shape (batch_size, state_size).
        """
        states, actions, rewards, next_states = experiences
        
        # Get max predicted Q values for next states from the target network
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_target = rewards.unsqueeze(1) + self.env.gamma * Q_targets_next

        # Get expected Q values from the main network
        Q_main = self.qnetwork_main(states).gather(1, actions.unsqueeze(1))

        # Compute and minimize the loss
        loss = F.mse_loss(Q_main, Q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        # Track learning rate
        for param_group in self.optimizer.param_groups:
            self.lrs.append(param_group['lr'])

        # Store loss for plotting
        self.losses.append(loss.item())
        # Update the target network
        self.update_target_network()
    
    def update_target_network(self):
        """Soft update target network parameters with polyak averaging."""
        for target_param, main_param in zip(self.qnetwork_target.parameters(), self.qnetwork_main.parameters()):
            target_param.data.copy_(self.tau * main_param.data + (1.0 - self.tau) * target_param.data)
    
    def __stack_state__(self, t, S, X, q, dim):
        """
        Stack the state components into a single tensor.
        Args:
            t (float): Current time.
            X (float): Current cash.
            S (float): Current midprice.
            q (float): Current inventory.
            dim (int): Dimension along which to stack the state components.
        Returns:
            torch.Tensor: Stacked state tensor of shape (batch_size, state_size).
        """
        return torch.stack((
            t / self.env.T,                     # Normalize time to [0, 1]
            S / self.env.S_0 - 1.0,             # Relative price change
            X / (self.env.S_0 * self.env.I_max),# Normalize cash to [-1, 1]
            q / self.env.I_max                  # Normalize inventory to [-1, 1]
        ), dim=dim)

    def train(self, n_iter, n_plot, eps_start, eps_end, eps_decay, batch_size):
        """
        Args:
            n_iter (int): Number of iterations to train the agent.
            n_plot (int): Number of iterations after which to plot the results.
            eps_start (float): Initial epsilon value for the epsilon-greedy policy.
            eps_end (float): Final epsilon value for the epsilon-greedy policy.
            eps_decay (float): Decay factor for epsilon.
            batch_size (int): Size of the mini-batch per iteration.
        """
        eps = eps_start
        total_rewards = []
        scores = []     # Average reward per episode
        
        for i in tqdm(range(n_iter)):
            eps = max(eps_end, eps * eps_decay)
            self.epsilons.append(eps)
            self.count += 1

            # Run one random time step
            # Randomize t between 0 and T, S between S_0 and S_0 + sigma, X = 0, q between -I_max and I_max
            k = int(np.ceil(self.env.T / self.env.dt))
            t = torch.randint(0, k, size=(batch_size,)).float() * self.env.dt
            # Randomize S = S_0 + sigma * W_t
            W_t = torch.sqrt(t) * torch.randn(batch_size)
            S = self.env.S_0 + self.env.sigma * W_t

            # Randomize q between -I_max and I_max
            q = torch.randint(-self.env.I_max, self.env.I_max + 1, size=(batch_size,))

            # X = 0 since cash doesn't matter
            X = torch.zeros(batch_size)

            # t = torch.zeros(batch_size)
            # S, q, X = self.env.Randomize_Start(batch_size)

            state = self.__stack_state__(t, S, X, q, dim=1)
            total_reward = torch.zeros(batch_size)

            # Select an action
            action = self.act(state, eps)
            # Take a step in the environment
            t_p, S_p, X_p, q_p, reward, isMO, buySellMO = self.env.step(t, S, X, q, action)
            # Store the experience in the replay buffer and perform a learning step
            next_state = self.__stack_state__(t_p, S_p, X_p, q_p, dim=1)
            self.step(state, action, reward, next_state)

            # # Update the state and score
            # state = next_state
            # t, S, X, q = t_p, S_p, X_p, q_p
            # total_reward += reward

            if i != 0 and i % n_plot == 0:
                print(f"Replay Buffer Size: {len(self.replay_buffer)}")
                for param_group in self.optimizer.param_groups:
                    print(f"Learning Rate: {param_group['lr']}")
                print(f"Iteration {i}, Epsilon: {eps}, Total Reward: {reward.mean().item()}")
                self.plot_training_batch_full(t, S, X, q, action, reward, i, batch_size)
                self.loss_plots()
                self.run_strategy(10_000, name=f"Iteration_{i}")
                self.plot_policy(name=f"Iteration_{i}")
            
            # Save the total reward for this episode
            total_rewards.append(reward.mean().item())
            scores.append(reward.mean().item())
        
        # torch.save(self.qnetwork_main.state_dict(), "mr_ddqn_model.pth")

        # plt.ylabel("Score")
        # plt.xlabel("Episode")
        # plt.plot(range(len(rewards)), rewards)
        # plt.plot(range(len(scores)), scores)
        # plt.legend(['Reward', "Score"])
        # plt.savefig("mr_ddqn_scores.png")
        # plt.show()

        # Combined figure with two subplots: Loss and Learning Rate
        fig, axs = plt.subplots(1, 3, figsize=(8, 6), sharex=False)

        # Left subplot: Loss
        axs[0].plot(self.losses, color='tab:blue')
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
        plt.savefig("loss_and_lr.png")
        plt.show()

        # Plot the policy heatmap and simulation results
        print(f"Epsilon: {eps}")
        self.loss_plots()
        self.run_strategy(10_000, name=f"Final_Episode")
        self.plot_policy(name=f"Final_Episode")
    
    # Functions to plot the results
    def moving_average(self, x, n):
        
        y = np.zeros(len(x))
        y_err = np.zeros(len(x))
        y[0] = np.nan
        y_err[0] = np.nan
        
        for i in range(1,len(x)):
            
            if i < n:
                y[i] = np.mean(x[:i])
                y_err[i] = np.std(x[:i])
            else:
                y[i] = np.mean(x[i-n:i])
                y_err[i] = np.std(x[i-n:i])
                
        return y, y_err

    def loss_plots(self):
        
        def plot(x, label, show_band=True):

            mv, mv_err = self.moving_average(x, 100)
        
            if show_band:
                plt.fill_between(np.arange(len(mv)), mv-mv_err, mv+mv_err, alpha=0.2)
            plt.plot(mv, label=label, linewidth=1) 
            plt.legend()
            plt.ylabel('loss')
            plt.yscale('symlog')
        
        fig = plt.figure(figsize=(8,4))
        plt.subplot(1,2,1)
        plot(self.losses, r'$Q$', show_band=False)
        
        plt.title("Losses")
        plt.tight_layout()
        plt.show()
    
    def run_strategy(self, nsims=10_000, name="", N=None):
        if N is None:
            N = self.env.Ndt  # number of time steps

        # Preallocate tensors
        S = torch.zeros((nsims, N + 1))
        X = torch.zeros((nsims, N + 1))
        q = torch.zeros((nsims, N + 1))
        r = torch.zeros((nsims, N))

        # Initial conditions
        S[:, 0] = self.env.S_0
        X[:, 0] = 0
        q[:, 0] = 0

        ones = torch.ones(nsims)

        for t in range(N):
            t_now = t * self.env.dt * ones

            # Stack state [t, S, X, q]
            state = self.__stack_state__(t_now, S[:, t], X[:, t], q[:, t], dim=1)

            with torch.no_grad():
                actions = torch.argmax(self.qnetwork_main(state), dim=1)
                action_real = self.action_spaces[actions]

            # Step the environment forward
            t_p, S[:, t + 1], X[:, t + 1], q[:, t + 1], r[:, t], _, _ = self.env.step(
                t_now, S[:, t], X[:, t], q[:, t], action_real
            )

        # Detach to numpy
        S = S.numpy()
        X = X.numpy()
        q = q.numpy()
        r = r.numpy()

        t = self.env.dt * np.arange(N + 1)

        # Plotting
        plt.figure(figsize=(12, 9))

        def plot_stat(ax, x, title, ylabel, t=None):
            if t is None:
                t = self.env.dt * np.arange(x.shape[1])
            qtl = np.quantile(x, [0.05, 0.5, 0.95], axis=0)
            ax.fill_between(t, qtl[0], qtl[2], alpha=0.3)
            ax.plot(t, qtl[1], color='black', linewidth=1)
            ax.plot(t, x[:3].T, linewidth=0.8)
            ax.set_title(title)
            ax.set_xlabel("Time")
            ax.set_ylabel(ylabel)
            ax.grid(True)


        # Subplots
        ax1 = plt.subplot(3, 2, 1)
        plot_stat(ax1, S - S[:, [0]], "Price Deviation", r"$S_t - S_0$")

        ax2 = plt.subplot(3, 2, 2)
        plot_stat(ax2, q, "Inventory", r"$q_t$")

        ax3 = plt.subplot(3, 2, 3)
        plot_stat(ax3, np.cumsum(r, axis=1), "Cumulative Reward", r"$\sum r_t$", t=t[:-1])


        ax4 = plt.subplot(3, 2, 4)
        ax4.hist(np.sum(r, axis=1), bins=51, color='gray')
        ax4.set_title("Final Reward Distribution")
        ax4.set_xlabel("Total Reward")
        ax4.set_ylabel("Count")
        ax4.grid(True)

        # Plot wealth = X + q * S
        ax5 = plt.subplot(3, 2, 5)
        wealth = X[:, :-1] + q[:, :-1] * S[:, :-1]  # shape (nsims, N)
        plot_stat(ax5, wealth, "Wealth Over Time", r"Wealth = $X_t + q_t \cdot S_t$")

        plt.suptitle(f"Strategy Simulation Results - {name}", fontsize=16)
        plt.tight_layout()
        plt.show()

        return t, S, X, q, r

    
    # def run_strategy(self, nsims=10_000, name="", N = None):
        
    #     if N is None:
    #         N = self.env.Ndt
        
    #     t = torch.zeros((nsims, N+1)).float()
    #     S = torch.zeros((nsims, N+1)).float()
    #     X = torch.zeros((nsims, N+1)).float()
    #     I  = torch.zeros((nsims, N+1)).float()
    #     I_p = torch.zeros((nsims, N+1)).float()
    #     r = torch.zeros((nsims, N)).float()
        
    #     S[:,0] = self.env.S_0
    #     X[:,0] = 0
    #     I[:,0] = 0
        
    #     ones = torch.ones(nsims)
        
    #     for t in range(N):
    #         t_now = t * self.env.dt * ones
    #         state = self.__stack_state__(t_now, S[:,t], X[:,t], I[:,t], dim=1)
            
    #         with torch.no_grad():
    #             actions = torch.argmax(self.qnetwork_main(X), dim=1)
    #             I_p[:, i] = self.action_spaces[actions]

    #         # I_p[:,i] = self.pi_main['net'](X).reshape(-1)

    #         S[:,i+1], I[:,i+1], r[:,i] = \
    #             self.env.step(i*ones, S[:,i], I[:,i], I_p[:,i])
                
    #     S = S.detach().numpy()
    #     I  = I.detach().numpy()
    #     I_p = I_p.detach().numpy()
    #     r = r.detach().numpy()

    #     i = self.env.dt*np.arange(0, N+1)/self.env.T
        
    #     plt.figure(figsize=(5,5))
    #     n_paths = 3
        
    #     def plot(i, x, plt_i, title ):
            
    #         # print(x.shape)
    #         # pdb.set_trace()
            
    #         qtl= np.quantile(x, [0.05, 0.5, 0.95], axis=0)
    #         # print(qtl.shape)
            
    #         plt.subplot(2, 2, plt_i)
            
    #         plt.fill_between(i, qtl[0,:], qtl[2,:], alpha=0.5)
    #         plt.plot(i, qtl[1,:], color='k', linewidth=1)
    #         plt.plot(i, x[:n_paths, :].T, linewidth=1)
            
    #         # plt.xticks([0,0.5,1])
    #         plt.title(title)
    #         plt.xlabel(r"$i$")
            
    #     plot(i, (S-S[:,0].reshape(S.shape[0],-1)), 1, r"$S_t-S_0$" )
    #     plot(i, I, 2, r"$I_t$")
    #     plot(i[:-1], np.cumsum(r, axis=1), 3, r"$r_t$")

    #     plt.subplot(2,2, 4)
    #     plt.hist(np.sum(r,axis=1), bins=51)


    #     plt.tight_layout()
        
    #     # plt.savefig("path_" + name + ".pdf", format='pdf', bbox_inches='tight')
    #     plt.show()
        
    #     # zy0 = self.env.swap_price(zx[0,0], rx[0,0], ry[0,0])
    #     # plt.hist(zy[:,-1],bins=np.linspace(51780,51810,31), density=True, label='optimal')
    #     # qtl_levels = [0.05,0.5,0.95]
    #     # qtl = np.quantile(zy[:,-1],qtl_levels)
    #     # c=['r','b','g']
    #     # for i, q in enumerate(qtl):
    #     #     plt.axvline(qtl[i], linestyle='--', 
    #     #                 linewidth=2, 
    #     #                 color=c[i],
    #     #                 label=r'${0:0.2f}$'.format(qtl_levels[i]))
    #     # plt.axvline(zy0,linestyle='--',color='k', label='swap-all')
    #     # plt.xlabel(r'$z_T^y$')
    #     # plt.legend()
    #     # plt.savefig('ddqn_zy_T.pdf', format='pdf',bbox_inches='tight')
    #     # plt.show()
        
    #     # print(zy0, np.mean(zy[:,-1]>zy0))
    #     # print(qtl)        
        
    #     return i, S, I, I_p

    def plot_policy(self, name=""):
        Nq = int(2 * self.env.I_max + 1)  # Number of inventory points
        Nt = int(self.env.T / 3 + 1)      # Number of time points

        t_vals = torch.linspace(0, self.env.T, Nt)
        q_vals = torch.linspace(-self.env.I_max, self.env.I_max, Nq)
        t_grid, q_grid = torch.meshgrid(t_vals, q_vals, indexing='ij')

        S_fixed = self.env.S_0 * torch.ones_like(t_grid).reshape(-1)
        X_fixed = torch.zeros_like(S_fixed)

        t_flat = t_grid.reshape(-1)
        q_flat = q_grid.reshape(-1)
        state_batch = self.__stack_state__(t_flat, S_fixed, X_fixed, q_flat, dim=1)

        with torch.no_grad():
            Q_values = self.qnetwork_main(state_batch)
            greedy_actions = torch.argmax(Q_values, dim=1).reshape(t_grid.shape)

        # Custom colors for actions: 0=nothing, 1=buy, 2=sell, 3=buy/sell
        cmap = mcolors.ListedColormap(['#FFFFFF', '#ADD8E6', '#FF9999', '#90EE90'])
        bounds = [0, 1, 2, 3, 4]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(greedy_actions.T, aspect="auto", origin="lower",
                    extent=[0, self.env.T, -self.env.I_max, self.env.I_max],
                    cmap=cmap, norm=norm)

        # Colorbar with labels
        cbar = fig.colorbar(im, ticks=[0.5, 1.5, 2.5, 3.5])
        cbar.ax.set_yticklabels(['nothing', 'buy', 'sell', 'buy/sell'])
        cbar.set_label('Greedy Action')

        ax.set_title("Optimal Policy Heatmap")
        ax.set_xlabel("Time")
        ax.set_ylabel("Inventory")
        ax.axhline(0, linestyle='--', color='k')
        plt.tight_layout()
        plt.show()
    
    def plot_training_batch_full(self, t, S, X, q, action, reward, iteration, batch_size):
        """
        Plot histograms and scatter plots for a batch of (t, S, X, q, action, reward).
        """
        t = t.numpy()
        S = S.numpy()
        X = X.numpy()
        q = q.numpy()
        a = action.numpy()
        r = reward.numpy()
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        # Histograms
        axes[0, 0].hist(S, bins=30, color='skyblue', edgecolor='black')
        axes[0, 0].set_title("Midprice Distribution")

        axes[0, 1].hist(q, bins=range(int(q.min())-1, int(q.max())+2), color='salmon', edgecolor='black')
        axes[0, 1].set_title("Inventory Distribution")

        axes[1, 0].hist(t, bins=30, color='lightgreen', edgecolor='black')
        axes[1, 0].set_title("Time Distribution")

        axes[1, 1].scatter(q, a, alpha=0.5, c=r, cmap='coolwarm', s=10)
        axes[1, 1].set_xlabel("Inventory")
        axes[1, 1].set_ylabel("Action")
        axes[1, 1].set_title("Action vs Inventory")

        plt.suptitle(f"Batch Diagnostics (batch_size={batch_size})- Iteration {iteration}", fontsize=16)
        plt.tight_layout()
        # plt.savefig(f"batch_diagnostics_iter_{iteration}.png")
        plt.show()

from standard_env import StandardEnv

import numpy as np
import matplotlib.pyplot as plt

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
        # TODO consider putting normalization here
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
        # print each tensor
        # print(f"states: {states}, actions: {actions}, rewards: {rewards}, next_states: {next_states}")
        # # print each list length
        # print(f"states: {len(states)}, actions: {len(actions)}, rewards: {len(rewards)}, next_states: {len(next_states)}")
        # # print each tensor shape
        # print(f"states: {states[0].shape}, actions: {actions[0].shape}, rewards: {rewards[0].shape}, next_states: {next_states[0].shape}")
        return (
            torch.stack(states),
            torch.tensor(actions),
            torch.tensor(rewards),
            torch.stack(next_states)
        )

class DDQNAgent:
    def __init__(self, env: StandardEnv, state_size, action_size, seed, lr=1e-3, gamma=0.99,
                 tau=1e-3, update_every=4, sample_size=64, capacity=1000000, name='',
                 n_nodes=16, n_hidden_layers=6):
        """
        Args:
            env (StandardEnv): Environment instance.
            state_size (int): Size of the state space.
            action_size (int): Size of the action space.
            seed (int): Random seed for reproducibility.
            lr (float): Learning rate for the optimizer.
            gamma (float): Discount factor for future rewards.
            tau (float): Soft update parameter for the target network.
            update_every (int): Number of steps between updates to the target network.
            batch_size (int): Size of the mini-batch for training.
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
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.sample_size = sample_size
        self.name = name

        self.steps = 0
        self.action_spaces = torch.tensor([0, 1, 2, 3], dtype=int)  # do nothing, buy, sell, buy and sell

        self.qnetwork_main = QNetwork(state_size, action_size, n_nodes, n_hidden_layers)
        self.qnetwork_target = QNetwork(state_size, action_size, n_nodes, n_hidden_layers)
        self.update_target_network()
        self.optimizer = optim.Adam(self.qnetwork_main.parameters(), lr=self.lr)
        self.replay_buffer = ReplayBuffer(capacity)

        self.epsilons = []
        self.losses = []
        self.count = 0  # Counter for the number of episodes
    
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
        Q_target = rewards + self.gamma * Q_targets_next

        # Get expected Q values from the main network
        # TODO actions = actions.long()
        Q_main = self.qnetwork_main(states).gather(1, actions.unsqueeze(1))

        # Compute and minimize the loss
        # print the shape of each
        print(f"Q_main: {Q_main.shape}, Q_target: {Q_target.shape}, actions: {actions.shape}")
        loss = F.mse_loss(Q_main, Q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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
            t / self.env.T,                  # Normalize time to [0, 1]
            S / self.env.S_0 - 1.0,          # Relative price change
            X,                               # TODO: normalize cash if needed
            q / self.env.I_max               # Normalize inventory to [-1, 1]
        ), dim=dim)

    def train(self, n_episode, n_plot, eps_start, eps_end, eps_decay, batch_size):
        """
        Args:
            n_episode (int): Number of episodes to train the agent.
            n_plot (int): Number of episodes after which to plot the results.
            eps_start (float): Initial epsilon value for the epsilon-greedy policy.
            eps_end (float): Final epsilon value for the epsilon-greedy policy.
            eps_decay (float): Decay factor for epsilon.
        """
        print("Entering DDQNAgent.train")
        eps = eps_start
        total_rewards = []
        scores = []     # Average reward per episode
        
        for i_episode in tqdm(range(n_episode)):
            eps = max(eps_end, eps * eps_decay)
            self.epsilons.append(eps)
            self.count += 1

            # Run the episode
            t = torch.zeros(batch_size)
            S, q, X = self.env.Randomize_Start(batch_size)
            state = self.__stack_state__(t, S, X, q, dim=1)
            total_reward = torch.zeros(batch_size)

            for _ in range(self.env.Ndt):
                # Select an action
                action = self.act(state, eps)
                # Take a step in the environment
                t_p, S_p, X_p, q_p, reward, isMO, buySellMO = self.env.step(t, S, X, q, action)
                # Store the experience in the replay buffer and perform a learning step
                next_state = self.__stack_state__(t_p, S_p, X_p, q_p, dim=1)
                self.step(state, action, reward, next_state)
                # Update the state and score
                state = next_state
                t, S, X, q = t_p, S_p, X_p, q_p
                total_reward += reward
            
            if i_episode != 0 and i_episode % n_plot == 0:
                print(f"Episode {i_episode}, Epsilon: {eps}, Total Reward: {total_reward.mean().item()}")
                self.loss_plots()
                self.run_strategy(10_000, name=f"Episode_{i_episode}")
                self.plot_policy(name=f"Episode_{i_episode}")
            
            # Save the total reward for this episode
            total_rewards.append(total_reward.mean().item())
            scores.append(total_reward.mean().item())
        
        # TODO torch.save(self.qnetwork_main.state_dict(), "mr_ddqn_model.pth")

        # plt.ylabel("Score")
        # plt.xlabel("Episode")
        # plt.plot(range(len(rewards)), rewards)
        # plt.plot(range(len(scores)), scores)
        # plt.legend(['Reward', "Score"])
        # plt.savefig("mr_ddqn_scores.png")
        # plt.show()

        # Plot the loss
        plt.figure()
        plt.plot(self.losses)
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.title("Training Loss over Time")
        plt.savefig("loss_curve.png")
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
        
        plt.tight_layout()
        plt.show()
    
    def run_strategy(self, nsims=10_000, name="", N = None):
        
        if N is None:
            N = self.env.N
        
        S = torch.zeros((nsims, N+1)).float()
        I  = torch.zeros((nsims, N+1)).float()
        I_p = torch.zeros((nsims, N+1)).float()
        r = torch.zeros((nsims, N)).float()

        S0 = self.env.S_0
        I0 = 0

        S[:,0] = S0
        I[:,0] = 0
        
        ones = torch.ones(nsims)
        
        for t in range(N):

            X = self.__stack_state__(S[:,t], I[:,t], dim=1)
            
            with torch.no_grad():
                actions = torch.argmax(self.qnetwork_main(X), dim=1)
                I_p[:, t] = self.action_spaces[actions]

            # I_p[:,t] = self.pi_main['net'](X).reshape(-1)

            S[:,t+1], I[:,t+1], r[:,t] = \
                self.env.step(t*ones, S[:,t], I[:,t], I_p[:,t])
                
        S = S.detach().numpy()
        I  = I.detach().numpy()
        I_p = I_p.detach().numpy()
        r = r.detach().numpy()

        t = self.env.dt*np.arange(0, N+1)/self.env.T
        
        plt.figure(figsize=(5,5))
        n_paths = 3
        
        def plot(t, x, plt_i, title ):
            
            # print(x.shape)
            # pdb.set_trace()
            
            qtl= np.quantile(x, [0.05, 0.5, 0.95], axis=0)
            # print(qtl.shape)
            
            plt.subplot(2, 2, plt_i)
            
            plt.fill_between(t, qtl[0,:], qtl[2,:], alpha=0.5)
            plt.plot(t, qtl[1,:], color='k', linewidth=1)
            plt.plot(t, x[:n_paths, :].T, linewidth=1)
            
            # plt.xticks([0,0.5,1])
            plt.title(title)
            plt.xlabel(r"$t$")
            
        plot(t, (S-S[:,0].reshape(S.shape[0],-1)), 1, r"$S_t-S_0$" )
        plot(t, I, 2, r"$I_t$")
        plot(t[:-1], np.cumsum(r, axis=1), 3, r"$r_t$")

        plt.subplot(2,2, 4)
        plt.hist(np.sum(r,axis=1), bins=51)


        plt.tight_layout()
        
        # plt.savefig("path_" + name + ".pdf", format='pdf', bbox_inches='tight')
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
        
        return t, S, I, I_p

    def plot_policy(self, name=""):
        NS = 101
        S = torch.linspace(self.env.S_0 - 3*self.env.inv_vol, 
                           self.env.S_0 + 3*self.env.inv_vol,
                           NS)
        NI = 51
        I = torch.linspace(-self.env.I_max, self.env.I_max, NI)
        
        Sm, Im = torch.meshgrid(S, I,indexing='ij')
        X = self.__stack_state__(Sm.reshape(-1), Im.reshape(-1), dim=1)  # shape [NS * NI, 2]
        with torch.no_grad():
            Q_values = self.qnetwork_main(X)
            greedy_actions = torch.argmax(Q_values, dim=1).cpu()
        
        # Convert discrete actions back to real values
        A = self.action_spaces[greedy_actions.numpy()].reshape(Sm.shape)

        fig, ax = plt.subplots()
        plt.title("Optimal Policy Heatmap for Time T")
        cs = plt.contourf(Sm.numpy(), Im.numpy(), A,
                          levels=np.linspace(-self.env.I_max, self.env.I_max, 21),
                          cmap='RdBu')
        ax.axvline(self.env.S_0, linestyle='--', color='g')
        ax.axvline(self.env.S_0 - 2 * self.env.inv_vol, linestyle='--', color='k')
        ax.axvline(self.env.S_0 + 2 * self.env.inv_vol, linestyle='--', color='k')
        ax.axhline(0, linestyle='--', color='k')
        ax.axhline(self.env.I_max / 2, linestyle='--', color='k')
        ax.axhline(-self.env.I_max / 2, linestyle='--', color='k')
        ax.set_xlabel("Price")
        ax.set_ylabel("Inventory")

        cbar = fig.colorbar(cs, ax=ax, shrink=0.9)
        cbar.set_ticks(np.linspace(-self.env.I_max, self.env.I_max, 11))
        cbar.ax.set_ylabel('Action')

        plt.tight_layout()
        # plt.savefig("policy_" + name + ".pdf", format='pdf', bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    from standard_env import StandardEnv

    env = StandardEnv(
        T = 300,
        I_max = 20,
        lambd = 50/300,
        Delta = 0.01,
        phi = 0.01,
        varphi = 0.01,
        sigma = 0.001,
        S_0 = 20,
        X_0 = 0,
        q_0 = 0
    )

    agent = DDQNAgent(
        env=env, state_size = 4, action_size = 4, seed=42, lr=1e-3, gamma=0.99,
        tau=1e-3, update_every=4, sample_size=64, capacity=1_000_000, name='trial',
        n_nodes=16, n_hidden_layers=6
    )

    agent.train(n_episode=5_000, n_plot=100, eps_start=1.0, eps_end=0.01, eps_decay=0.993, batch_size=10)
from env import normal_env
from DDQN import normal_DDQN
import numpy as np
import torch
import matplotlib.pyplot as plt

env = normal_env(
    T = 300,
    I_max = 20,
    lambd = 50/300,
    Delta = 0.01,
    phi = 0.01,
    sigma = 0.001,
    POV = 0.2,
    Nq = 20,
    S_0 = 1300,
    X_0 = 0,
    q_0 = 0,
    varphi = 0.01
)
agent = normal_DDQN(env=env, gamma=0.9999, lr=1e-4, n_layers=6, n_nodes=36, name="test")

agent.train(n_iter=5000, n_plot=200, n_iter_Q=5)
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

plt.style.use("paper.mplstyle")
from ShortTermalpha import ShortTermalpha
from ST_alpha_env import ST_alpha_env
from DDPG_ST_ALPHA.ST_alpha_DPG import ST_alpha_Agent

# %%
short_term_alpha = ShortTermalpha(zeta=0.5, epsilon=0.002, eta=0.001)

env = ST_alpha_env(
    ShortTermalpha=short_term_alpha,
    T=60,
    POV=0.2,
    Nq=20,
    S_0=30,
    X_0=0,
    q_0=0,
    Delta=0.01,
    varphi=0.01,
    phi=0,
    sigma=0.005,
)

# ddpg = DDPG(env, I_max=10, gamma=0.999, lr=1e-3, name="test")

# # %%
# ddpg.train(n_iter=10_000, n_plot=200, n_iter_Q=5, n_iter_pi=5)

# %%
# import necessary libraries
import numpy as np
import math
import matplotlib as mpl
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pylab as pylab
params = {'legend.fontsize': 10,
          'figure.figsize': (8, 4),
         'axes.labelsize': 20,
         'axes.titlesize': 20,
         'xtick.labelsize': 15,
         'ytick.labelsize': 15}
pylab.rcParams.update(params)

# %%
# these are helper functions for the main solver

# for the various constants of interest from the alpha process
class ShortTermAlpha:

    def __init__(self, zeta, epsilon, eta):
        self.zeta = zeta
        self.epsilon = epsilon
        self.eta = eta
        self.dalpha = None
        self.value = None

    def generate_dalpha(self, dt, lambda_p, lambda_m):
        self.dalpha = math.sqrt((self.eta ** 2 + self.epsilon ** 2 * (lambda_p + lambda_m)) * 3 * dt)

    def generate_alpha(self, alpha_threshold, decimal_places=4):
        if self.dalpha is not None:
            max_alpha = math.ceil(alpha_threshold / self.dalpha) * self.dalpha
            min_alpha = -max_alpha
            self.value = np.round(np.arange(min_alpha, max_alpha + self.dalpha, self.dalpha), decimal_places)
            
# for displaying progress in solving the DPE
def counter(current_idx, total_idx):
    M = round(total_idx/10)
    percent = int(round(current_idx/total_idx*100,0))
    if np.mod(current_idx,M) == 0 :
        print("Processing: " + str(percent) + " percent completed.")


def nan_to_num(x, nan):
    """ Change the NaNs in a numpy array to the desired values.
    :param x: a numpy array
    :param nan: desired value
    :return: a deep copy of x with changed values
    """
    y = np.copy(x)
    y[np.isnan(y)] = nan
    return y


def linear_extrapolation(x, y, e):
    """ Extrapolation and interpolation
    
    :param x: a numpy array
    :param y: a numpy array
    :param e: a numpy array, equivalent of x
    :return: a numpy array
    """
    new_x = np.sort(x)
    new_y = y[np.argsort(x)]

    def point_wise(ep):
        if ep < new_x[0]:
            return new_y[0] + (ep - new_x[0]) * (new_y[1] - new_y[0]) / (new_x[1] - new_x[0])
        elif ep > new_x[-1]:
            return new_y[-1] + (ep - new_x[-1]) * (new_y[-1] - new_y[-2]) / (new_x[-1] - new_x[-2])
        else:
            return np.interp([ep], x, y)[0]

    return np.array([point_wise(i) for i in e])

# %%
def generate_h_matrix(short_term_alpha, q, t, Delta, varphi, dt, phi, lambda_p, lambda_m):
    """ Solves the DPE on page 269.
    
    :param short_term_alpha: a short_term_alpha object
    :param q: an inventory vector
    :param t: a time 
    :param Delta: variable
    :param varphi: variable
    :param dt: variable
    :param phi: variable
    :param lambda_p: a numpy vector
    :param lambda_m: a numpy vector
    :return: h value matrix, l+ and l- of Equation 10.32
    """
    
    # set up the time, inventory, and space grids
    dalpha = short_term_alpha.dalpha
    alpha = short_term_alpha.value
    eta = short_term_alpha.eta
    zeta = short_term_alpha.zeta
    Nalpha = alpha.shape[0]
    epsilon = short_term_alpha.epsilon
    alpha_up = alpha + epsilon
    alpha_down = alpha - epsilon

    Nt = t.shape[0]
    Nq = q.shape[0]

    # stores the h as a function of q, alpha, t
    h = np.full((Nq, Nalpha, Nt), fill_value=np.nan)
    
    # stores the optimal posting strategies as a function of q, alpha, t
    lp = np.zeros((Nq, Nalpha, Nt), dtype=bool)
    lm = np.zeros((Nq, Nalpha, Nt), dtype=bool)

    # Terminal conditions for all q
    h[:, :, h.shape[-1] - 1] = np.around((-(0.5 * Delta * np.sign(q) + varphi * q) * q), decimals=4)[:, np.newaxis]

    d_alpha_h = np.zeros(Nalpha)

    # Index of alpha smaller than 0
    idx_m = np.where(alpha < 0)[0]
    # Index of alpha greater than 0
    idx_p = np.where(alpha > 0)[0]
    # Index of alpha equals to 0
    idx_0 = np.where(alpha == 0)[0]

    for i in range(h.shape[2] - 2, -1, -1):

        # print counter
        counter(Nt - i, Nt)

        # Solve the HJB in the continuation region  backward from t + dt to t
        for k in range(0, h.shape[0], 1):
            
            # compute the optimal strategy over the time interval [t,t+dt) Eq. (10.32)
            # interpolate/extrapolate to alpha + eta
            h_p2 = linear_extrapolation(alpha, h[k, :, i + 1], alpha_up)
            if q[k] > -(Nq - 1) / 2:
                h_p1 = linear_extrapolation(alpha, h[k - 1, :, i + 1], alpha_up)
                post_h_p = 0.5 * Delta + h_p1 - h_p2
            else:
                post_h_p = np.zeros(Nalpha)

            # interpolate/extrapolate to alpha - eta
            h_m2 = linear_extrapolation(alpha, h[k, :, i + 1], alpha_down)
            if q[k] < (Nq - 1) / 2:
                h_m1 = linear_extrapolation(alpha, h[k + 1, :, i + 1], alpha_down)
                post_h_m = 0.5 * Delta + h_m1 - h_m2
            else:
                post_h_m = np.zeros(Nalpha)

            lp[k, :, i + 1] = post_h_p > 0
            lm[k, :, i + 1] = post_h_m > 0
            
            
            # solve DPE for h function one time-step backwards
            d_alpha_h[idx_m] = (h[k, idx_m + 1, i + 1] - h[k, idx_m, i + 1]) / dalpha
            d_alpha_h[idx_p] = (h[k, idx_p, i + 1] - h[k, idx_p - 1, i + 1]) / dalpha
            d_alpha_h[idx_0] = (h[k, idx_0 + 1, i + 1] - h[k, idx_0 - 1, i + 1]) / (2 * dalpha)

            d2_alpha_h = (h[k, 2:h.shape[1], i + 1] - 2 * h[k, 1:(h.shape[1] - 1), i + 1] + h[k, 0:(h.shape[1] - 2),
                                                                                            i + 1]) / (dalpha ** 2)

            h[k, 1:(h.shape[1] - 1), i] = h[k, 1:(h.shape[1] - 1), i + 1] \
                                          + dt * (-zeta * alpha[1:(alpha.shape[0] - 1)] * d_alpha_h[
                                                                                          1:(d_alpha_h.shape[0] - 1)]
                                                  + 0.5 * (eta ** 2) * d2_alpha_h
                                                  + zeta * alpha[1:(alpha.shape[0] - 1)] * q[k] - phi * (q[k] ** 2)
                                                  + lambda_p * np.maximum(post_h_p[1:(post_h_p.shape[0] - 1)], 0)
                                                  + lambda_m * np.maximum(post_h_m[1:(post_h_m.shape[0] - 1)], 0)
                                                  + lambda_p * (h_p2[1:(h_p2.shape[0] - 1)] - h[k, 1:(h.shape[1] - 1), i + 1])
                                                  + lambda_m * (h_m2[1:(h_m2.shape[0] - 1)] - h[k, 1:(h.shape[1] - 1), i + 1]))

            # impose second derivative vanishes along maximum and minimum values of alpha grid
            h[k, h.shape[1] - 1, i] = 2 * h[k, h.shape[1] - 2, i] - h[k, h.shape[1] - 3, i]
            h[k, 0, i] = 2 * h[k, 1, i] - h[k, 2, i]
            
    print("done!")

    return h, lp, lm

def generate_sell_buy_posts(short_term_alpha, t, q, lp, lm):
    """ Calculate the inventory level above/below which it is optimal to post buy/sell limit orders
    based on lp amd lm solved from the DPE previously.
    
    :param short_term_alpha: a short_term_alpha object
    :param t: a time vector
    :param q: an inventory vector
    :param lp: a 3d matrix
    :param lm: a 3d mrtix
    :return: sell side posts and buy side posts
    """
    Nt = t.shape[0]
    Nalpha = short_term_alpha.value.shape[0]  #27
    lp_b = np.full((Nalpha, Nt), np.NaN)
    lm_b = np.full((Nalpha, Nt), np.NaN)

    for k in range(0, Nt):
        for j in range(0, Nalpha):
            idx = np.where(lp[:, j, k] == 1)[0]
            if idx.size != 0:
                # First Index
                idx = idx[0]
                lp_b[j, k] = q[idx]
            idx = np.where(lm[:, j, k] == 1)[0]
            if idx.size != 0:
                # Last Index
                idx = idx[-1]
                lm_b[j, k] = q[idx]
    return lp_b, lm_b


# %%
# plotting functions for visualizing the solutions
def plot_post_surface(t, alpha, inventory, q, title):
    """ Generate the plots (b) or (c) of Figure 10.10 on page 270. The surface plot with Inventory as a function
    vs alpha and t. You can change the cmap parameter to see more beatiful suface plots, for example, let cmap='coolwarm'.
    """
    x, y = np.meshgrid(alpha, t)
    z = np.transpose(inventory)
    z[np.isnan(z)] = np.nan
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='rainbow', vmin=np.nanmin(z),vmax=np.nanmax(z), linewidth=0, antialiased=False)
    
    #ax.xaxis._axinfo['label']['space_factor'] = 20.0
    #ax.yaxis._axinfo['label']['space_factor'] = 20.0
    #ax.zaxis._axinfo['label']['space_factor'] = 20.0
    
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$t$')
    ax.set_zlabel(r'$Inventory$')
    ax.set_zlim(np.min(q), np.max(q))
    plt.title(label=title)
    ax.dist=10

    plt.show()

def plot_optimal_posting(alpha, lp_b, lm_b, title):
    """ Plot the figure 10.10 (a) on Page 270. Using the buy and sell postings calculated using the function
    generate_sell_buy_posts.
    """
    x = alpha
    y1 = lp_b[:, 1]
    y2 = lm_b[:, 1]
    plt.plot(x, y1, color="black", linewidth=1, linestyle="solid")
    plt.plot(x, y2, color="black", linewidth=1, linestyle="solid")
    plt.ylim(min(np.nanmin(y1), np.nanmin(y2)), max(np.nanmax(y1), np.nanmax(y2)))
    plt.xlim((np.nanmin(alpha), np.nanmax(alpha)))

    ax = plt.gca()
    area1 = ax.fill_between(x, np.maximum(y1, y2), max(np.nanmax(y1), np.nanmax(y2)), facecolor="red", interpolate=True
                            , alpha=0.5)
    area2 = ax.fill_between(x, y1, y2, where=(y2 >= y1), facecolor="green", interpolate=True, alpha=0.5)
    area3 = ax.fill_between(x, np.minimum(y1, y2), min(np.nanmin(y1), np.nanmin(y2)), facecolor="blue", interpolate=True
                            , alpha=0.5)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$Inventory$')
    plt.legend([area1, area2, area3], [r'$sell$', r'$buy+sell$', r'$buy$'])
    plt.title(label=title)
    plt.show()

# %%
def generate_simulation(Nsims, t, dt, Delta, sigma, lambda_p, lambda_m, short_term_alpha, lm, lp, Nq, varphi):
    """ Generate simulation.
    # TODO: the implementation missed nu in the setting of 10.30.
    
    :param Nsims: number of simulations needed
    :param t: a time vector
    :param dt: granularity of time vector
    :param sigma: a variable
    :param lambda_p: a numpy vector
    :param lambda_m: a numpy vector
    :param short_term_alpha: a short-term-alpha object
    :return:
    """
    np.random.seed(20)
    
    # set up time grid
    Nt = t.shape[0]
    alpha = short_term_alpha.value
    dalpha = short_term_alpha.dalpha

    # store the paths of postings, fills, inventory, cash, short-term alpha, and asset price
    lp_path = np.full((Nsims, Nt), np.nan)
    lm_path = np.full((Nsims, Nt), np.nan)
    
    isfilled_p = np.full((Nsims, Nt), np.nan)
    isfilled_m = np.full((Nsims, Nt), np.nan)
    
    q_path = np.full((Nsims, Nt), np.nan)
    q_path[:, 0] = np.zeros(Nsims)

    X = np.full((Nsims, Nt), np.nan)
    X[:, 0] = 0

    alpha_path = np.full((Nsims, Nt), np.nan)
    alpha_path[:, 0] = 0    
    
    s_path = np.full((Nsims, Nt), np.nan)
    s_path[:, 0] = 30

    # used for noting if a  market order arrives and if it is buy/sell
    isMO = np.zeros((Nsims, Nt), dtype=bool)
    buySellMO = np.zeros((Nsims, Nt))


    zeta = short_term_alpha.zeta
    eta = short_term_alpha.eta
    epsilon = short_term_alpha.epsilon

    
    for i in range(0, Nt - 1, 1):
        
        # Print Counter
        counter(i, Nt - 1)

        # Update S by simulating (10.30) forward
        s_path[:, i+1] = s_path[:, i] + alpha_path[:, i] * dt + sigma * np.random.randn(Nsims)

        # Decide if a Market order arrives
        isMO[:, i] = np.random.rand(Nsims) < np.around((1 - np.exp(-dt * (lambda_p + lambda_m))), decimals=4)

        # Decide if it is a buy or sell market order
        buySellMO[:, i] = 2 * (np.random.rand(Nsims) < lambda_p / (lambda_p + lambda_m)) - 1

        # Update alpha by simulating (10.31) forward
        alpha_path[:, i+1] = np.exp(-zeta * dt) * alpha_path[:, i] + eta * np.sqrt(dt) * np.random.randn(Nsims) + \
                             epsilon * np.multiply(isMO[:, i], buySellMO[:, i])

        tOld = dt * i
        tNow = dt * (i + 1)

        # find the current limit order volume
        idx_t = np.where(t <= tOld)[0][-1]

        # perhaps interpolation is better?
        alpha_path_idx = np.minimum(alpha.shape[0], np.maximum(1, np.floor((alpha_path[:, i] - alpha[0]) / dalpha) + 1))

        # Make decisions of whether to buy or sell
        lp_path[:, i] = [lp[int(q_path[j, i] + Nq), int(alpha_path_idx[j]), int(idx_t)] for j in range(0, Nsims)]
        lm_path[:, i] = [lm[int(q_path[j, i] + Nq), int(alpha_path_idx[j]), int(idx_t)] for j in range(0, Nsims)]

        # Update Inventory
        isfilled_p[:, i] = np.multiply((lp_path[:, i]).astype(int), (isMO[:, i]).astype(int), (buySellMO[:, i] == 1).astype(int))
        isfilled_m[:, i] = np.multiply((lm_path[:, i]).astype(int), (isMO[:, i]).astype(int), (buySellMO[:, i] == -1).astype(int))

        q_path[:, i+1] = q_path[:, i] + isfilled_m[:, i] - isfilled_p[:, i]

        X[:, i + 1] = X[:, i] - np.multiply(s_path[:, i] - 0.5 * Delta, isfilled_m[:, i]) + \
                      np.multiply(s_path[:, i] + 0.5 * Delta, isfilled_p[:, i])

    # account for terminal liquidating trading actions
    X[:, X.shape[1]-1] += np.multiply(s_path[:, s_path.shape[1]-1] +
                                      (0.5 * Delta) * np.sign(q_path[:, q_path.shape[1]-1]) +
                                      varphi * q_path[:, q_path.shape[1]-1],
                                      q_path[:, q_path.shape[1]-1])
    
    q_path[:, q_path.shape[1]-1] = 0
    
    print("done!")
    
    return X, q_path, alpha_path, s_path, lm_path, lp_path, isMO, buySellMO

# %%
# plotting functionality for the simulations
def plot_simulation(x, y, xlabel, ylabel):
    """ This function can be used to generate the right hand side picture of Figure 10.11 on page 271.
    The short-term alpha generated by the simulator vs time. It can also be used to generate inventory size simulated vs time.
    """
    plt.plot(x, y, color="blue", linewidth=1, linestyle="solid")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def plot_simulation_decision(Delta, t, s_path, lm_path, lp_path, isMO, buySellMO, zoom=None):
    """ This function generates the left hand side picture of Figure 10.11 at page 271, which is the sample path of the 
    optimal strategy. Green lines show when and at what price the agent is posted. Solid red circles indicates MOs that arrive 
    and hit/lift the posted bid/offers. Open red circles indicate MOs that arrive and hit/lift the posted bid/offers. 
    Open red circles indicate MOs that arrive but do not hit/lift the agent's posts. Shaded region is the bid-ask spread.
    """
    s_post_buy = s_path - (0.5 * Delta)
    s_post_buy[lm_path == 0] = np.NaN
    
    is_market_buy_order = np.logical_and(isMO == 1, buySellMO == -1)
    # Index for closed circles
    idx_closed_buy = np.where(np.logical_and(is_market_buy_order, nan_to_num(lm_path, nan=0) == 1))
    # Index for open circles
    idx_open_buy = np.where(np.logical_and(is_market_buy_order, nan_to_num(lm_path, nan=1) == 0))

    s_post_sell = s_path + (0.5 * Delta)
    s_post_sell[lp_path == 0] = np.NaN

    is_market_sell_order = np.logical_and(isMO == 1, buySellMO == 1)
    # Index for closed circles
    idx_closed_sell = np.where(np.logical_and(is_market_sell_order, nan_to_num(lp_path, nan=0) == 1))
    # Index for open circles
    idx_open_sell = np.where(np.logical_and(is_market_sell_order, nan_to_num(lp_path, nan=1) == 0))

    
    ax = plt.gca()
    ax.plot(t, s_path, color="black", linewidth=2, linestyle="solid")
    # Plot the buy side of the LOB
    ax.plot(t, s_path - 0.5 * Delta, color="black", linewidth=1, linestyle="solid", zorder=1)
    ax.plot(t, s_post_buy, color="green", linewidth=3, linestyle="solid", zorder=2)
    # Plot the sell side of LOB
    ax.plot(t, s_path + 0.5 * Delta, color="black", linewidth=1, linestyle="solid", zorder=3)
    ax.plot(t, s_post_sell, color="blue", linewidth=3, linestyle="solid", zorder=4)
    # Plot shaded regions
    ax.fill_between(t, s_path + 0.5 * Delta, s_path - 0.5 * Delta, facecolor="blue", alpha=0.2)
    # Add closed circles for buy side
    ax.scatter(t[idx_closed_buy], s_path[idx_closed_buy] - 0.5 * Delta, color='red' , zorder=5)
    # Add open circles for buy side
    ax.scatter(t[idx_open_buy], s_path[idx_open_buy] - 0.5 * Delta, facecolors='none', edgecolors='r', zorder=6)
    # Add closed circles for sell side
    ax.scatter(t[idx_closed_sell], s_path[idx_closed_sell] + 0.5 * Delta, color='red', zorder=7)
    # Add open circles for sell side
    ax.scatter(t[idx_open_sell], s_path[idx_open_sell] + 0.5 * Delta, facecolors='none', edgecolors='r', zorder=8)
    
    if zoom is not None:
        plt.xlim(zoom)
        ylim_min = np.nanmin(s_post_buy[np.logical_and(zoom[0] <= t, t < zoom[1])])
        ylim_max = np.nanmax(s_post_sell[np.logical_and(zoom[0] <= t, t < zoom[1])])
        plt.ylim((ylim_min, ylim_max))

    plt.xlabel(r'$Time$')
    plt.ylabel(r'$Midprice$')
    plt.show()
    
def plot_wealth_hist(t, X, q, S, which_path):
    
    # plot marked to market of wealth sample path
    wealth = X[which_path,:] + q[which_path,:] * S[which_path,:]
    plt.plot(t, wealth)
    plt.xlabel(r'Time ($t$)')
    plt.ylabel(r'Marked-to-Market Value')
    plt.title(r'Wealth Path')
    plt.show()
    
    # plot histogram of terminal wealth
    p = 0.005 # truncate plot to contain 0.005 of data in tails
    q= np.quantile(X[:,-1], np.array([p, 1-p]))
    h_out = plt.hist(X[:,-1],bins=np.linspace(np.round(q[0]),np.round(q[1]),25))
    max_freq = 1.1*np.max(h_out[0])

    q = np.quantile(X[:,-1], np.array([0.01, 0.1, 0.5, 0.9, 0.99]))
    for i in range(len(q)):
        plt.plot(q[i]*np.array([1,1]),np.array([0,max_freq]),'--')
    plt.xlabel(r'Terminal Profit & Loss')
    plt.ylabel(r'Frequency')
    plt.title(r'Terminal Cash Histogram and Quantiles')
    plt.legend(('qtl=0.01','qtl=0.1','qtl=0.5','qtl=0.9','qtl=0.99'))
    plt.ylim((0,max_freq ))
    plt.show()
    
def plot_turnover_hist(q):
    
    Y = np.sum(np.abs(np.diff(q,axis=1)), axis=1)
    
    # plot histogram of terminal wealth
    p = 0.005 # truncate plot to contain 0.005 of data in tails
    q= np.round(np.quantile(Y, np.array([p, 1-p])))
    h_out = plt.hist(Y, bins=np.linspace(0, int(np.round(q[1])), int(np.round(q[1])) + 1))
    max_freq = 1.1*np.max(h_out[0])

    q = np.quantile(Y, np.array([0.01, 0.1, 0.5, 0.9, 0.99]))
    for i in range(len(q)):
        plt.plot(q[i]*np.array([1,1]),np.array([0,max_freq]),'--')
    plt.xlabel(r'$\sum_{t=1}^T|q_t-q_{t-1}|$')
    plt.ylabel(r'Frequency')
    plt.title(r'Total inventory turnover')
    plt.legend(('qtl=0.01','qtl=0.1','qtl=0.5','qtl=0.9','qtl=0.99'))
    plt.ylim((0,max_freq ))
    plt.show()



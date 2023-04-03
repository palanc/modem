import matplotlib.pyplot as plt
from algorithm.helper import Episode
import glob
import pickle
import numpy as np

EPISODE_DIR = '/mnt/nfs_code/robopen_users/plancaster/remodem/modem/episodes/franka-FrankaPickPlaceRandomReal_v2d'
POLICY_DIR = EPISODE_DIR + '/policy'
MPPI_DIR = EPISODE_DIR + '/mppi'

STATE_LABELS = ['qp0','qp1','qp2','qp3','qp4','qp5','qp6','qp7','qp8',
                'qv0','qv1','qv2','qv3','qv4','qv5','qv6','qv7','qv8',
                'eef_x','eef_y','eef_z']
STATE_ENABLED = [1,1,1,1,1,1,1,1,0,
                 0,0,0,0,0,0,0,0,0,
                 1,1,1]
ACT_LABELS = ['act_x','act_y','act_z','act_cos','act_sin','act_grasp','act_term']
ACT_ENABLED = [1,1,1,1,1,1,0]
N_PLOT_COLS = 3

def load_episodes(episode_dir):
    fps = glob.glob(episode_dir + "/*.pt")

    episodes = []
    for fp in fps:
        with open(fp, "rb") as ep_file:
            episodes.append(pickle.load(ep_file)) 
    return episodes

def plot_episode(episode, axs, color):
    assert(episode.state.shape[1] == len(STATE_LABELS))
    assert(episode.action.shape[1] == len(ACT_LABELS))

    plot_row = 0
    plot_col = 0
    x_vals = np.arange(episode.state.shape[0])
    for i in range(episode.state.shape[1]):
        if STATE_ENABLED[i] == 0:
            continue
        axs[plot_row, plot_col].plot(x_vals, episode.state[:,i], color=color,linewidth=0.25)
        #axs[plot_row, plot_col].set_xlabel('Step')
        #axs[plot_row, plot_col].set_ylabel(STATE_LABELS[i])
        axs[plot_row, plot_col].set_title(STATE_LABELS[i])
        plot_col += 1
        if plot_col >= N_PLOT_COLS:
            plot_row += 1
            plot_col = 0

    x_vals = np.arange(episode.action.shape[0])
    for i in range(episode.action.shape[1]):
        if ACT_ENABLED[i] == 0:
            continue
        axs[plot_row, plot_col].plot(x_vals, episode.action[:,i],color=color,linewidth=0.25)
        #axs[plot_row, plot_col].set_xlabel('Step')
        #axs[plot_row, plot_col].set_ylabel(ACT_LABELS[i])
        axs[plot_row, plot_col].set_title(ACT_LABELS[i])
        plot_col += 1
        if plot_col >= N_PLOT_COLS:
            plot_row += 1
            plot_col = 0

def main():
    policy_eps = load_episodes(POLICY_DIR)
    mppi_eps = load_episodes(MPPI_DIR)
    total_rows = (np.sum(np.array(STATE_ENABLED))+np.sum(np.array(ACT_ENABLED)))
    total_rows = (total_rows//N_PLOT_COLS) if (total_rows%N_PLOT_COLS==0) else ((total_rows//N_PLOT_COLS)+1)

    #fig, axs = plt.subplots(total_rows, N_PLOT_COLS, figsize=(6.4, 14.4))
    fig, axs = plt.subplots(total_rows, N_PLOT_COLS)
    fig.tight_layout()

    for ep in mppi_eps:
        plot_episode(ep, axs, 'b')

    #for ep in policy_eps:
    #    plot_episode(ep, axs, 'r')

    plt.savefig(fname='rollouts.pdf', format='pdf')

if __name__ == '__main__':
    main()
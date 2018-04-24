import sys
sys.path.append('/Users/Divye/Documents/cse/599g1/drl_hw1')
from drl_hw1.utils.gym_env import GymEnv
from drl_hw1.policies.gaussian_mlp import MLP
from drl_hw1.baselines.linear_baseline import LinearBaseline
from drl_hw1.baselines.mlp_baseline import MLPBaseline
from drl_hw1.algos.batch_reinforce import BatchREINFORCE
from drl_hw1.algos.npg import NaturalPolicyGradients
from drl_hw1.utils.train_agent import train_agent
import drl_hw1.envs
import time as timer
import itertools
import numpy as np

seeds = [21]
LR = 'learning_rates'
NT = 'num_trajecs'

envs = [(GymEnv('drl_hw1_half_cheetah-v0'), 'cheetah_vpg_exp'), (GymEnv('drl_hw1_swimmer-v0'), 'swimmer_vpg_exp'), (GymEnv('drl_hw1_ant-v0'), 'ant_vpg_exp')]

params = {
    LR : list(np.arange(0.0, 1.0, 0.2)),
    NT:  list(range(2, 20, 4))
}

all_params = itertools.product(params[LR], params[NT])
for seed in seeds:
    for env, job_name in envs:
        for i, (learning_rate, num_traj) in enumerate(all_params):
            policy = MLP(env.spec, hidden_sizes=(64, 64), seed=seed)
            baseline = LinearBaseline(env.spec)
            agent = BatchREINFORCE(env, policy, baseline, learn_rate=learning_rate, seed=seed, save_logs=True)
            train_agent(job_name='%s_lr0%d_traj%d_%d' % (job_name, 10*learning_rate, num_traj, seed),
                        agent=agent,
                        seed=seed,
                        niter=50,
                        gamma=0.995,
                        gae_lambda=0.97,
                        num_cpu=2,
                        sample_mode='trajectories',
                        num_traj=num_traj,
                        save_freq=5,
                        evaluation_rollouts=10)

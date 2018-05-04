import os
from drl_hw1.baselines.mlp_baseline import MLPBaseline
from drl_hw1.algos.npg import NaturalPolicyGradients
from drl_hw1.algos.adaptive_vpg import AdaptiveVPG
from project.settings import *
import drl_hw1.envs

from drl_hw1.policies.gaussian_mlp import MLP
from drl_hw1.baselines.linear_baseline import LinearBaseline
from drl_hw1.algos.batch_reinforce import BatchREINFORCE
from drl_hw1.utils.train_agent import train_agent
import time as timer

import argparse
from drl_hw1.utils.gym_env import GymEnv
import pickle



def main():
    SEED = 500
    
    parser = argparse.ArgumentParser(description='Train agent')
    parser.add_argument('-env', metavar='ENV', type=str,
                        help='The env out of cheetah/ant/swimmer/reacher', default='cheetah')
    parser.add_argument('-bl', metavar='BL', type=str,
                        help='The baseline you want out of linear(lin)/MLP(mlp)', default='mlp')
    parser.add_argument('-a', metavar='A', type=str,
                        help='The agent you want out of batch/adp/npg', default='npg')
    parser.add_argument('-nt', metavar='NT', type=int,
                        help='Number of trajectories', default=10)
    parser.add_argument('-id', metavar='ID', type=str,
                        help='Some special id to add to the result folder', default='')
    parser.add_argument('-seed', metavar='SD', type=int,
                        help='Random seed', default=SEED)
    parser.add_argument('-alpha', metavar='AL', type=float,
                        help='The learning rate', default=0.4)
    parser.add_argument('-delta', metavar='DL', type=float,
                        help='The delta bar', default=1.0)

    args = parser.parse_args()

    SEED = args.seed

    envs = {
        'cheetah': GymEnv('drl_hw1_half_cheetah-v0'),
        'ant' : GymEnv('drl_hw1_ant-v0'),
        'swimmer' : GymEnv('drl_hw1_swimmer-v0'),
        'reacher': GymEnv('drl_hw1_reacher-v0')
    }

    e = envs[args.env]

    policy = MLP(e.spec, hidden_sizes=(32,32), seed=SEED)

    baselines = {
        'lin' : LinearBaseline(e.spec),
        'mlp' : MLPBaseline(e.spec, seed=SEED)
    }

    baseline = baselines[args.bl]

    agents = {
        'batch': BatchREINFORCE(e, policy, baseline, learn_rate=args.alpha, seed=SEED, save_logs=True),
        'adp': AdaptiveVPG(e, policy, baseline, seed=SEED, save_logs=True, kl_desired=args.delta),
        'npg': NaturalPolicyGradients(e, policy, baseline, seed=SEED, save_logs=True, delta=args.delta)
    }

    agent = agents[args.a]

    ts = timer.time()
    job_name = os.path.join(RES_DIR, '%s_traj%d_%d_%s' % (args.env, args.nt, SEED, args.id))
    train_agent(job_name= ensure_dir(job_name),
                agent=agent,
                seed=SEED,
                niter=int(500/args.nt),
                gamma=0.995,
                gae_lambda=0.97,
                num_cpu=5,
                sample_mode='trajectories',
                num_traj=args.nt,
                save_freq=5,
                evaluation_rollouts=10)
    print("time taken = %f" % (timer.time()-ts))

    policy_path = os.path.join(job_name, 'iterations/best_policy.pickle')
    policy = pickle.load(open(policy_path, 'rb'))
    e.visualize_policy(policy, num_episodes=5, horizon=e.horizon, mode='exploration')
    del(e)

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return file_path

if __name__ == '__main__':
    main()

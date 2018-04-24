from drl_hw1.utils.gym_env import GymEnv
import pickle
import drl_hw1.envs

# point mass
# e = GymEnv('drl_hw1_point_mass-v0')
# policy = pickle.load(open('/Users/Divye/Documents/CSE/599G1/drl_hw1/drl_hw1/results/swimmer_traj10_500_it55/iterations/best_policy.pickle', 'rb'))
# e.visualize_policy(policy, num_episodes=10, horizon=e.horizon, mode='evaluation')
# del(e)

# swimmer
e = GymEnv('drl_hw1_swimmer-v0')
# file= '/Users/Divye/Documents/CSE/599G1/drl_hw1/drl_hw1/results/swimmer_traj10_500_it50/iterations/best_policy.pickle'
file = '/Users/Divye/Documents/CSE/599G1/drl_hw1/drl_hw1/results/swimmer_traj10_500_it50_64_64_10_10_01/iterations/best_policy.pickle'
# file = '/Users/Divye/Documents/CSE/599G1/drl_hw1/drl_hw1/results/swimmer_traj10_500_it50_64_64_10_10_desc/iterations/policy_40.pickle'
# file = 'swimmer_pol.pickle'
# file = '/Users/Divye/Documents/CSE/599G1/drl_hw1/drl_hw1/results/swimmer_traj5_500_it100_64_64_10_10_01/iterations/policy_90.pickle'
policy = pickle.load(open(file, 'rb'))
e.visualize_policy(policy, num_episodes=5, horizon=500, mode='exploration')
del(e)
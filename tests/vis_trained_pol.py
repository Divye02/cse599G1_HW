from drl_hw1.utils.gym_env import GymEnv
import pickle
import drl_hw1.envs

# point mass
# e = GymEnv('drl_hw1_point_mass-v0')
# policy = pickle.load(open('/Users/Divye/Documents/CSE/599G1/drl_hw1/drl_hw1/results/swimmer_traj10_500_it55/iterations/best_policy.pickle', 'rb'))
# e.visualize_policy(policy, num_episodes=10, horizon=e.horizon, mode='evaluation')
# del(e)

# swimmer
envs = {
        'cheetah': GymEnv('drl_hw1_half_cheetah-v0'),
        'ant' : GymEnv('drl_hw1_ant-v0'),
        'swimmer' : GymEnv('drl_hw1_swimmer-v0')
    }

file = 'best_policy_che_4.pickle'
e = envs['cheetah']

policy = pickle.load(open(file, 'rb'))
e.visualize_policy(policy, num_episodes=5, horizon=500, mode='exploration')
del(e)
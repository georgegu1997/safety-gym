import gym
import safety_gym
from safety_gym.envs.engine import Engine

from pprint import pprint

config = {
    'robot_base': 'xmls/point.xml',
    'task': 'push',
    'observe_goal_lidar': True,
    'observe_box_lidar': True,
    'observe_hazards': True,
    'observe_vases': True,
    'observation_flatten': False,
    'constrain_hazards': True,
    'lidar_max_dist': 3,
    'lidar_num_bins': 16,
    'hazards_num': 4,
    'vases_num': 4
}

env = Engine(config)

# env = gym.make('Safexp-PointGoal1-v0')

# Run a random policy and visualize the results on the screen
for i in range(10):
    env.reset()
    done = False
    while not done:
        next_observation, reward, done, info = env.step(env.action_space.sample())
        pprint(next_observation)
        print(reward)
        print(done)
        print(info)
        env.render(mode='human')
        input()

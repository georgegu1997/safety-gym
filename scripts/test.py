import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import gym
import safety_gym
from pathlib import Path
from safety_gym.envs.engine import Engine

from pprint import pprint

# from mujoco_py import GlfwContext
# GlfwContext(offscreen=True)  # Create a window to init GLFW.

config = {
    'robot_base': 'xmls/point2.xml',
    'task': 'goal',
    'observe_goal_lidar': True,
    'observe_box_lidar': True,
    'observe_hazards': True,
    'observe_vases': True,
    'constrain_hazards': True,
    'lidar_max_dist': 3,
    'lidar_num_bins': 16,
    'hazards_num': 4,
    'vases_num': 4,
    'observation_flatten': False,
    'render_lidar_markers': False,
    # 'observe_vision': True,
    # "render_labels": True,
    # 'observe_gremlins': False,
}

env = Engine(config)

# env = gym.make('Safexp-PointGoal1-v0')

# Run a random policy and visualize the results on the screen
for ep in range(100):
    env.reset()
    done = False
    t = 0
    ep_folder = Path('./generated_dataset/ep_%06d/' % ep)
    if not os.path.exists(ep_folder):
        os.makedirs(ep_folder)

    while not done:
        next_observation, reward, done, info = env.step(env.action_space.sample())
        pprint(next_observation)
        print(reward)
        print(done)
        print(info)

        # env.render(mode='human')

        # Every 20 steps, save next_observation['vision'] to a file
        # if t % 20 == 0:
        #     img = cv2.cvtColor(next_observation['vision'], cv2.COLOR_BGR2RGB)
        #     img = (img * 255).round().astype(np.uint8)
        #     # vertical flip
        #     img = cv2.flip(img, 0)
        #     cv2.imwrite('./obs/vision_{}.png'.format(t), img)
        #     plt.imshow(img)
        #     plt.show()
        #     pass

        if t % 5 == 0:
            img = env.render(mode='rgb_array', camera_id=1)
            # plt.imshow(img)
            # plt.show()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(str(ep_folder / ('obs_%06d.png'%(t//5))), img)

        t += 1

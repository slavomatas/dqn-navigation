import os
import torch

from unityagents import UnityEnvironment

import sys
sys.path.append("../")

from utils.utils import process_observation
from agent import Agent


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = UnityEnvironment(file_name="../VisualBanana_Linux/Banana.x86_64", no_graphics=False)

# get the default brain# get t
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=False)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space
state = env_info.visual_observations[0]
print('States look like:', state)
print('States have shape:', state.shape)

state = process_observation(state, device)

# load the weights from file
agent = Agent(input_shape=state.shape[1:], action_size=action_size, seed=0)
agent.qnetwork_local.load_state_dict(torch.load('../checkpoints/dueling_checkpoint.pth'))

score = 0  # initialize the score

for i in range(3):
    env_info = env.reset(train_mode=False)[brain_name]

    state = env_info.visual_observations[0]  # get the current state
    state = process_observation(state, device)

    for j in range(2000):
        action = agent.act(state)
        env_info = env.step(action)[brain_name]

        next_state = env_info.visual_observations[0]  # get the next state
        next_state = process_observation(next_state, device)

        reward = env_info.rewards[0]  # get the reward
        done = env_info.local_done[0]  # see if episode has finished
        state = next_state
        score += reward
        print('\rScore: {:.2f}'.format(score), end="")
        if done:
            break

env.close()
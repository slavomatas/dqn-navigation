import os
import torch
import numpy as np

from unityagents import UnityEnvironment

from agent import Agent


def transform_visual_observation(state, device):
    state = np.squeeze(state)

    #state = rgb2gray(state)
    #state = state[:, :, 0] * 0.299 + state[:, :, 1] * 0.587 + state[:, :, 2] * 0.114

    #state = state / 255.0

    #state = state.reshape(state.shape[0], state.shape[1], 1)

    state = state.transpose((2, 0, 1))

    state = torch.from_numpy(state)
    state = state.type(torch.FloatTensor)
    state = state.unsqueeze_(0)
    state = state.to(device)

    return state

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

state = transform_visual_observation(state, device)

# load the weights from file
agent = Agent(input_shape=state.shape[1:], action_size=action_size, seed=0)
agent.qnetwork_local.load_state_dict(torch.load('../checkpoints/dueling_checkpoint.pth'))

score = 0  # initialize the score

for i in range(3):
    env_info = env.reset(train_mode=False)[brain_name]

    state = env_info.visual_observations[0]  # get the current state
    state = transform_visual_observation(state, device)

    for j in range(2000):
        action = agent.act(state)
        env_info = env.step(action)[brain_name]

        next_state = env_info.visual_observations[0]  # get the next state
        next_state = transform_visual_observation(next_state, device)

        reward = env_info.rewards[0]  # get the reward
        done = env_info.local_done[0]  # see if episode has finished
        state = next_state
        score += reward
        print('\rScore: {:.2f}'.format(score), end="")
        if done:
            break

env.close()
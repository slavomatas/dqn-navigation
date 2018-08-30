import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from collections import deque
from skimage.color import rgb2gray

from pixels_dqn_agent import PixelsAgent
from unityagents import UnityEnvironment


# visualize a 3 channel RGB image
def visualize_state(state):
    img = np.squeeze(state.numpy())
    img = img.reshape(img.shape[0], img.shape[1], 1)
    plt.imshow(img)
    plt.show()


def transform_visual_observation(state, device):
    state = np.squeeze(state)

    state = rgb2gray(state)
    #state = state[:, :, 0] * 0.299 + state[:, :, 1] * 0.587 + state[:, :, 2] * 0.114

    #state = state / 255.0

    state = state.reshape(state.shape[0], state.shape[1], 1)

    state = state.transpose((2, 0, 1))

    state = torch.from_numpy(state)
    state = state.type(torch.FloatTensor)
    state = state.unsqueeze_(0)
    state = state.to(device)

    return state

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = UnityEnvironment(file_name="VisualBanana_Linux/Banana.x86_64", no_graphics=False)

# get the default brain# get t
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space
state = env_info.visual_observations[0]
print('States look like:', state)
print('States have shape:', state.shape)

score = 0  # initialize the score

state = transform_visual_observation(state, device)

# Instantiate Agent
agent = PixelsAgent(input_shape=state.shape[1:], action_size=action_size, seed=0)

def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]

        state = env_info.visual_observations[0]  # get the current state
        state = transform_visual_observation(state, device)

        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]

            next_state = env_info.visual_observations[0]  # get the next state
            next_state = transform_visual_observation(next_state, device)

            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            #print('\nEpisode {}\t TimeStep {} \tScore: {:.2f} \tDone {}'.format(i_episode, t, score, done), end="")
            if done:
                break

        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            #visualize_state(state)
        if np.mean(scores_window) >= 13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'pixels_checkpoint.pth')
            break
    return scores


scores = dqn()

# plot the scores
"""
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
"""

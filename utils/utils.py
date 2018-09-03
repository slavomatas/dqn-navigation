import numpy as np
import torch
import torch.nn as nn

EPS = 1e-7


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias, 0.1)


def process_observation(observation, device):
    observation = np.squeeze(observation)

    # Convert the image to greyscale
    observation = observation.mean(axis=2)

    # Add third color dim
    observation = observation.reshape(observation.shape[0], observation.shape[1], 1)

    # Swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W
    observation = observation.transpose((2, 0, 1))

    # Convert to tensor
    observation = torch.from_numpy(observation).type(torch.FloatTensor).unsqueeze(0).to(device)

    return observation


def one_hot(x, n, device):
    assert x.dim() == 2
    one_hot_x = torch.zeros(x.size(0), n).to(device)
    one_hot_x.scatter_(1, x, 1)
    return one_hot_x


import utils
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import deque, namedtuple
from torch.autograd import Variable

from utils.utils import one_hot, EPS

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 5e-4  # learning rate
UPDATE_EVERY = 4  # how often to update the network

Vmax = 15
Vmin = -1
N_ATOMS = 51
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([(0.0 if e.done else 1.0) for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class DistributionalDQN(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(DistributionalDQN, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc = nn.Sequential(
            nn.Linear(state_size, fc1_units),
            nn.ReLU(),
            nn.Linear(fc1_units, fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, action_size* N_ATOMS)
        )

        self.register_buffer("supports", torch.arange(Vmin, Vmax+DELTA_Z, DELTA_Z))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.size()[0]
        fc_out = self.fc(x)
        logits = fc_out.view(batch_size, -1, N_ATOMS)
        probs = nn.functional.softmax(logits, 2)
        return probs


class Agent():
    def __init__(self, state_size, action_size, seed):

        self.num_atoms = N_ATOMS
        self.vmin = float(Vmin)
        self.vmax = float(Vmax)

        self.delta_z = (self.vmax - self.vmin) / (self.num_atoms - 1)

        zpoints = np.linspace(Vmin, Vmax, N_ATOMS).astype(np.float32)
        self.zpoints = Variable(torch.from_numpy(zpoints).unsqueeze(0)).to(device)

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        self.num_actions = action_size

        # Q-Network
        self.online_q_net = DistributionalDQN(state_size, action_size, seed).to(device)
        self.target_q_net = DistributionalDQN(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.online_q_net.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        self.online_q_net.eval()
        self.online_q_net.train()

        # Epsilon-greedy action selection
        return self.get_action(state, eps)

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, non_ends = experiences

        actions = one_hot(actions, self.action_size, device)
        targets = self.compute_targets(rewards, next_states, non_ends, gamma)
        states = Variable(states)
        actions = Variable(actions)
        targets = Variable(targets)
        loss = self.loss(states, actions, targets)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        #return loss.data[0]

        # ------------------- update target network ------------------- #
        self.soft_update(self.online_q_net, self.target_q_net, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def get_action(self, state, eps):
        """Run Greedy-Epsilon for the given state.

        params:
            state: numpy-array [num_frames, w, h]

        return:
            action: int, in range [0, num_actions)
        """
        if np.random.uniform() <= eps:
            action = np.random.randint(0, self.num_actions)
            return action

        q_vals = self.online_q_values(state)
        #utils.assert_eq(q_vals.size(0), 1)
        q_vals = q_vals.view(-1)
        q_vals = q_vals.cpu().numpy()
        action = q_vals.argmax()
        return action

    def _q_values(self, q_net, states):
        """internal function to compute q_value

        params:
            q_net: self.online_q_net or self.target_q_net
            states: Variable [batch, channel, w, h]
        """
        probs = q_net(states) # [batch, num_actions, num_atoms]
        q_vals = (probs * self.zpoints).sum(2)
        return q_vals, probs

    def target_q_values(self, states):
        states = Variable(states, volatile=True)
        q_vals, _ = self._q_values(self.target_q_net, states)
        return q_vals.data

    def online_q_values(self, states):
        states = Variable(states, volatile=True)
        q_vals, _ = self._q_values(self.online_q_net, states)
        return q_vals.data

    def compute_targets(self, rewards, next_states, non_ends, gamma):
        """Compute batch of targets for distributional dqn

        params:
            rewards: Tensor [batch, 1]
            next_states: Tensor [batch, channel, w, h]
            non_ends: Tensor [batch, 1]
            gamma: float
        """
        # get next distribution
        next_states = Variable(next_states, volatile=True)

        # [batch, num_actions], [batch, num_actions, num_atoms]
        next_q_vals, next_probs = self._q_values(self.target_q_net, next_states)
        next_actions = next_q_vals.data.max(1, True)[1] # [batch, 1]
        next_actions = one_hot(next_actions, self.num_actions, device).unsqueeze(2)
        next_greedy_probs = (next_actions * next_probs.data).sum(1)

        # transform the distribution
        rewards = rewards
        non_ends = non_ends
        proj_zpoints = rewards + gamma * non_ends * self.zpoints.data
        proj_zpoints.clamp_(self.vmin, self.vmax)

        # project onto shared support
        b = (proj_zpoints - self.vmin) / self.delta_z
        lower = b.floor()
        upper = b.ceil()
        # handle corner case where b is integer
        eq = (upper == lower).float()
        lower -= eq
        lt0 = (lower < 0).float()
        lower += lt0
        upper += lt0

        # note: it's faster to do the following on cpu
        ml = (next_greedy_probs * (upper - b)).cpu().numpy()
        mu = (next_greedy_probs * (b - lower)).cpu().numpy()

        lower = lower.cpu().numpy().astype(np.int32)
        upper = upper.cpu().numpy().astype(np.int32)

        batch_size = rewards.size(0)
        mass = np.zeros((batch_size, self.num_atoms), dtype=np.float32)
        brange = range(batch_size)
        for i in range(self.num_atoms):
            mass[brange, lower[brange, i]] += ml[brange, i]
            mass[brange, upper[brange, i]] += mu[brange, i]

        return torch.from_numpy(mass).to(device)

    def loss(self, states, actions, targets):
        """
        params:
            states: Variable [batch, channel, w, h]
            actions: Variable [batch, num_actions] one hot encoding
            targets: Variable [batch, num_atoms]
        """
        #utils.assert_eq(actions.size(1), self.num_actions)

        actions = actions.unsqueeze(2)
        probs = self.online_q_net(states) # [batch, num_actions, num_atoms]
        probs = (probs * actions).sum(1) # [batch, num_atoms]
        xent = -(targets * torch.log(probs.clamp(min=EPS))).sum(1)
        xent = xent.mean(0)
        return xent

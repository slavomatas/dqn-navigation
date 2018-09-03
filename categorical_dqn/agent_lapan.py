import random
import traceback

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import namedtuple, deque

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


class DistributionalDQN(nn.Module):
    """Actor (Policy) Model."""

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
        return fc_out.view(batch_size, -1, N_ATOMS)

    def both(self, x):
        cat_out = self(x)
        probs = self.apply_softmax(cat_out)
        weights = probs * self.supports
        res = weights.sum(dim=2)
        return cat_out, res

    def qvals(self, x):
        return self.both(x)[1]

    def apply_softmax(self, t):
        return self.softmax(t.view(-1, N_ATOMS)).view(t.size())


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
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = DistributionalDQN(state_size, action_size, seed).to(device)
        self.qnetwork_target = DistributionalDQN(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

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
        # state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        try:

            states, actions, rewards, next_states, dones = experiences
            batch_size = len(states)

            # next state distribution
            next_distr_v, next_qvals_v = self.qnetwork_target.both(next_states)
            next_actions = next_qvals_v.max(1)[1].data.cpu().numpy()
            next_distr = self.qnetwork_target.apply_softmax(next_distr_v).data.cpu().numpy()

            next_best_distr = next_distr[range(batch_size), next_actions]

            # project our distribution using Bellman update
            proj_distr = self.distr_projection(next_best_distr, rewards, dones, Vmin, Vmax, N_ATOMS, gamma)

            # calculate net output
            distr_v = self.qnetwork_local(states)
            print("distr_v.shape {} actions.shape {}".format(distr_v.shape, actions.shape))
            state_action_values = distr_v[range(batch_size), actions.data]
            state_log_sm_v = F.log_softmax(state_action_values, dim=1)
            proj_distr_v = torch.tensor(proj_distr).to(device)

            """
            if save_prefix is not None:
                pred = F.softmax(state_action_values, dim=1).data.cpu().numpy()
                save_transition_images(batch_size, pred, proj_distr, next_best_distr, dones, rewards, save_prefix)
            """

            loss_v = -state_log_sm_v * proj_distr_v
            loss_v = loss_v.sum(dim=1).mean()

            # Minimize the loss
            self.optimizer.zero_grad()
            loss_v.backward()
            self.optimizer.step()

            # ------------------- update target network ------------------- #
            self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
        except:
            print("====> Exception: Failed to execute learn")
            print(traceback.print_exc())

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        # print("soft_update")

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


    def distr_projection(self, next_distr, rewards, dones, Vmin, Vmax, n_atoms, gamma):
        """
        Perform distribution projection aka Catergorical Algorithm from the
        "A Distributional Perspective on RL" paper
        """
        rewards = rewards.data.cpu().numpy()
        dones = dones.data.cpu().numpy().astype(np.bool)

        batch_size = len(rewards)
        proj_distr = np.zeros((batch_size, n_atoms), dtype=np.float32)
        delta_z = (Vmax - Vmin) / (n_atoms - 1)
        for atom in range(n_atoms):
            tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards + (Vmin + atom * delta_z) * gamma))
            b_j = (tz_j - Vmin) / delta_z
            l = np.floor(b_j).astype(np.int64)
            u = np.ceil(b_j).astype(np.int64)
            eq_mask = u == l
            eq_mask = np.squeeze(eq_mask)
            proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
            ne_mask = u != l
            ne_mask = np.squeeze(ne_mask)
            proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
            proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]
        if dones.any():
            proj_distr[np.squeeze(dones)] = 0.0
            tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards[dones]))
            b_j = (tz_j - Vmin) / delta_z
            l = np.floor(b_j).astype(np.int64)
            u = np.ceil(b_j).astype(np.int64)
            eq_mask = u == l
            eq_dones = dones.copy()
            eq_dones[dones] = eq_mask
            eq_dones = np.squeeze(eq_dones)
            if eq_dones.any():
                proj_distr[eq_dones, l] = 1.0
            ne_mask = u != l
            ne_dones = dones.copy()
            ne_dones[dones] = ne_mask
            ne_dones = np.squeeze(ne_dones)
            if ne_dones.any():
                proj_distr[ne_dones, l] = (u - b_j)[ne_mask]
                proj_distr[ne_dones, u] = (b_j - l)[ne_mask]
        return proj_distr

import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from replay_buffer.replay_buffer import PrioritizedReplayBuffer, LinearSchedule

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 5e-4  # learning rate
UPDATE_EVERY = 4  # how often to update the network

Vmax = 10
Vmin = -10
N_ATOMS = 51
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DistributionalDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DistributionalDQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions * N_ATOMS)
        )

        self.register_buffer("supports", torch.arange(Vmin, Vmax+DELTA_Z, DELTA_Z))
        self.softmax = nn.Softmax(dim=1)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        batch_size = x.size()[0]
        fx = x.float() / 256
        conv_out = self.conv(fx).view(batch_size, -1)
        fc_out = self.fc(conv_out)
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


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, input_shape, action_size, seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = input_shape
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = DistributionalDQN(input_shape, action_size).to(device)
        self.qnetwork_target = DistributionalDQN(input_shape, action_size).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        self.prioritized_replay_alpha = 0.6
        self.prioritized_replay_beta0 = 0.4
        self.prioritized_replay_beta_iters = 100000

        # Replay memory
        self.memory = PrioritizedReplayBuffer(BUFFER_SIZE, alpha=self.prioritized_replay_alpha)
        self.beta_schedule = LinearSchedule(self.prioritized_replay_beta_iters,
                                            initial_p=self.prioritized_replay_beta0,
                                            final_p=1.0)

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
                experiences = self.memory.sample(BATCH_SIZE, beta=self.beta_schedule.value(len(self.memory)))
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
        # print("learn")

        states, actions, rewards, next_states, dones, weights, idxes = experiences
        #states, actions, rewards, dones, next_states = common.unpack_batch(batch)
        batch_size = len(states)

        states_v = torch.tensor(states).to(device)
        actions_v = torch.tensor(actions).to(device)
        next_states_v = torch.tensor(next_states).to(device)

        # next state distribution
        next_distr_v, next_qvals_v = self.qnetwork_target.both(next_states_v)
        next_actions = next_qvals_v.max(1)[1].data.cpu().numpy()
        next_distr = self.qnetwork_target.apply_softmax(next_distr_v).data.cpu().numpy()

        next_best_distr = next_distr[range(batch_size), next_actions]
        dones = dones.astype(np.bool)

        # project our distribution using Bellman update
        proj_distr = self.distr_projection(next_best_distr, rewards, dones, Vmin, Vmax, N_ATOMS, gamma)

        # calculate net output
        distr_v = self.qnetwork_local(states_v)
        state_action_values = distr_v[range(batch_size), actions_v.data]
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


    def distr_projection(next_distr, rewards, dones, Vmin, Vmax, n_atoms, gamma):
        """
        Perform distribution projection aka Catergorical Algorithm from the
        "A Distributional Perspective on RL" paper
        """
        batch_size = len(rewards)
        proj_distr = np.zeros((batch_size, n_atoms), dtype=np.float32)
        delta_z = (Vmax - Vmin) / (n_atoms - 1)
        for atom in range(n_atoms):
            tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards + (Vmin + atom * delta_z) * gamma))
            b_j = (tz_j - Vmin) / delta_z
            l = np.floor(b_j).astype(np.int64)
            u = np.ceil(b_j).astype(np.int64)
            eq_mask = u == l
            proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
            ne_mask = u != l
            proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
            proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]
        if dones.any():
            proj_distr[dones] = 0.0
            tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards[dones]))
            b_j = (tz_j - Vmin) / delta_z
            l = np.floor(b_j).astype(np.int64)
            u = np.ceil(b_j).astype(np.int64)
            eq_mask = u == l
            eq_dones = dones.copy()
            eq_dones[dones] = eq_mask
            if eq_dones.any():
                proj_distr[eq_dones, l] = 1.0
            ne_mask = u != l
            ne_dones = dones.copy()
            ne_dones[dones] = ne_mask
            if ne_dones.any():
                proj_distr[ne_dones, l] = (u - b_j)[ne_mask]
                proj_distr[ne_dones, u] = (b_j - l)[ne_mask]
        return proj_distr

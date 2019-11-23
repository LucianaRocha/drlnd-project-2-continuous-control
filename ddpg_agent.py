from collections import namedtuple, deque

import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim


from model import Actor, Critic

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-2              # for soft update of target parameters
LR_actor = 5e-4         # learning rate for actor
LR_critic = 5e-4        # learning rate for critic
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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

        # Actor-Network
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.soft_update(self.actor_local, self.actor_target, 1)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_actor)

        # Critic-Network
        self.critic_local = Critic(state_size, action_size, seed).to(device)
        self.critic_target = Critic(state_size, action_size, seed).to(device)
        self.soft_update(self.critic_local, self.critic_target, 1)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_critic)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

    def act(self, state):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
        """
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action_values = self.actor_local(state)
        self.actor_local.train()
        return action_values.data.cpu().numpy()

    def step(self, state, action, reward, next_state):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state)
        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) >= BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s') tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states = experiences

        # Update critic
        next_actions = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, next_actions)
        Q_targets = rewards + (gamma * Q_targets_next)
        self.critic_optimizer.zero_grad()
        Q_expected = self.critic_local(states, actions)
        loss = F.mse_loss(Q_expected, Q_targets)
        loss.backward()
        self.critic_optimizer.step()

        # update actor policy
        self.actor_optimizer.zero_grad()
        actions_predict = self.actor_local(states)
        loss = -self.critic_local(states, actions_predict).mean()
        loss.backward()
        self.actor_optimizer.step()

        #soft-update
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

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
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


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
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
  
        return (states, actions, rewards, next_states)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

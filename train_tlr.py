import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import pandas as pd

# Hyperparameters
GAMMA = 0.99
LR = 1e-4
BUFFER_SIZE = 100000
BATCH_SIZE = 64
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
TARGET_UPDATE = 10

# Modified Replay Buffer with Prioritization
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.alpha = alpha

    def add(self, transition, td_error=1.0):
        max_priority = max(self.priorities, default=1.0)
        self.buffer.append(transition)
        self.priorities.append(max_priority if td_error == 0 else td_error)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)
            self.priorities.pop(0)

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            return [], []

        priorities = np.array(self.priorities) ** self.alpha
        probs = priorities / priorities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()

        transitions = [self.buffer[idx] for idx in indices]
        return transitions, weights, indices

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-5  # Small value to prevent zero priority

    def __len__(self):
        return len(self.buffer)

# Shared Replay Buffer
class SharedReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        transitions = [self.buffer[idx] for idx in indices]
        return zip(*transitions)

# Modified DQNetwork to include Dueling DQN option
class DQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, dueling=False):
        super(DQNetwork, self).__init__()
        self.dueling = dueling

        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)

        if dueling:
            self.value_stream = nn.Linear(128, 1)
            self.advantage_stream = nn.Linear(128, action_dim)
        else:
            self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        if self.dueling:
            value = self.value_stream(x)
            advantage = self.advantage_stream(x)
            return value + (advantage - advantage.mean())
        else:
            return self.fc3(x)

# Modified DQNAgent for Prioritized Replay
class DQNAgent:
    def __init__(self, state_dim, action_dim, dueling=False, double=False, prioritized=False, shared_buffer=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dueling = dueling
        self.double = double
        self.prioritized = prioritized

        self.q_network = DQNetwork(state_dim, action_dim, dueling)
        self.target_network = DQNetwork(state_dim, action_dim, dueling)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LR)

        if prioritized:
            self.buffer = PrioritizedReplayBuffer(BUFFER_SIZE)
        else:
            self.buffer = ReplayBuffer(BUFFER_SIZE)

        self.shared_buffer = shared_buffer  # Shared replay buffer
        self.epsilon = 1.0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()

    def train(self, beta=0.4):
        if len(self.buffer) < BATCH_SIZE:
            return

        if self.prioritized:
            transitions, weights, indices = self.buffer.sample(BATCH_SIZE, beta)
            states, actions, rewards, next_states, dones = zip(*transitions)
        else:
            states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_values = self.q_network(states).gather(1, actions).squeeze(1)
        if self.double:
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)
            next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1)
        else:
            next_q_values = self.target_network(next_states).max(1)[0]

        targets = rewards + (1 - dones) * GAMMA * next_q_values

        td_errors = targets.detach() - q_values
        loss = (td_errors ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.prioritized:
            self.buffer.update_priorities(indices, td_errors.abs().detach().numpy())

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

# Multi-Environment Training Manager with Shared Buffer
class MultiEnvManager:
    def __init__(self, envs, dqn_types, prioritized=False):
        self.envs = [gym.make(env) for env in envs]
        self.shared_buffer = SharedReplayBuffer(BUFFER_SIZE) if prioritized else None
        self.agents = [
            DQNAgent(
                env.observation_space.shape[0],
                env.action_space.n,
                dueling=(dqn_type == "dueling"),
                double=(dqn_type == "double"),
                prioritized=prioritized,
                shared_buffer=self.shared_buffer
            ) for env, dqn_type in zip(self.envs, dqn_types)
        ]
        self.episode_logs = []

    def train(self, episodes, log_file="training_log.csv"):
        for episode in range(episodes):
            total_rewards = [0] * len(self.envs)
            states = [env.reset() for env in self.envs]
            dones = [False] * len(self.envs)

            while not all(dones):
                for i, (env, agent) in enumerate(zip(self.envs, self.agents)):
                    if not dones[i]:
                        action = agent.select_action(states[i])
                        next_state, reward, done, _ = env.step(action)
                        agent.buffer.add((states[i], action, reward, next_state, done))
                        if self.shared_buffer:
                            self.shared_buffer.add((states[i], action, reward, next_state, done))
                        agent.train()
                        total_rewards[i] += reward
                        states[i] = next_state
                        dones[i] = done

            for agent in self.agents:
                agent.update_target_network()

            # Logging
            avg_rewards = [total / episodes for total in total_rewards]
            self.episode_logs.append([episode] + total_rewards)

            # Epsilon decay
            for agent in self.agents:
                agent.epsilon = max(MIN_EPSILON, agent.epsilon * EPSILON_DECAY)

            # Terminal Output
            print(f"Episode {episode + 1}/{episodes} - Rewards: {total_rewards}")

        # Save logs to CSV
        log_df = pd.DataFrame(self.episode_logs, columns=["Episode"] + [f"Env_{i+1}_Reward" for i in range(len(self.envs))])
        log_df.to_csv(log_file, index=False)
        print(f"Training log saved to {log_file}")

# Initialize and train
if __name__ == "__main__":
    envs = ["CartPole-v1", "LunarLander-v2", "SpaceInvaders-ram-v0"]
    dqn_types = ["standard", "double", "dueling"]  # DQN variants for each environment
    manager = MultiEnvManager(envs, dqn_types, prioritized=True)
    manager.train(episodes=25000)

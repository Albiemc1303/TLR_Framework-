import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import pandas as pd

from AI_project import PPOAgent

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

# Standard Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        transitions = [self.buffer[idx] for idx in indices]
        return transitions

    def __len__(self):
        return len(self.buffer)

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
            transitions = self.buffer.sample(BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*transitions)

            states = torch.FloatTensor(np.array(states))
            actions = torch.LongTensor(actions).unsqueeze(1)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(np.array(next_states))
            dones = torch.FloatTensor(dones)
    
            if self.prioritized:
                self.buffer.update_priorities(indices, td_errors.abs().detach().numpy())
    
            q_values = self.q_network(states).gather(1, actions).squeeze()
            next_q_values = self.target_network(next_states).max(1)[0].detach()
    
            targets = rewards + (1 - dones) * GAMMA * next_q_values
    
            td_errors = targets - q_values
            loss = (td_errors ** 2).mean()
        
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        if self.prioritized:
            self.buffer.update_priorities(indices, td_errors.abs().detach().numpy())
        
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=LR):
            self.state_dim = state_dim
            self.action_dim = action_dim
    
            # Actor and Critic Networks
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(),
                nn.Linear(128, action_dim), nn.Softmax(dim=-1)
            )
            self.critic = nn.Sequential(
                nn.Linear(state_dim, 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(),
                nn.Linear(128, 1)
            )
    
            self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
            self.buffer = []
    
    def select_action(self, state):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            probs = self.actor(state_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            return action.item(), dist.log_prob(action)
    
    def store_transition(self, transition):
            self.buffer.append(transition)
    
    def train(self, gamma=0.99, epsilon=0.2):
            states, actions, rewards, log_probs, dones = zip(*self.buffer)
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            log_probs = torch.stack(log_probs)
    
            # Compute returns
            returns = []
            discounted_sum = 0
            for reward, done in zip(reversed(rewards), reversed(dones)):
                discounted_sum = reward + gamma * discounted_sum * (1 - done)
                returns.insert(0, discounted_sum)
            returns = torch.FloatTensor(returns)
    
            # Calculate advantages
            values = self.critic(states).squeeze()
            advantages = returns - values.detach()
    
            # Update Actor (Clipped PPO Loss)
            new_probs = self.actor(states)
            dist = torch.distributions.Categorical(new_probs)
            new_log_probs = dist.log_prob(actions)
    
            ratio = (new_log_probs - log_probs).exp()
            clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
            actor_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    
            # Update Critic (Value Loss)
            critic_loss = nn.MSELoss()(values, returns)
    
            # Backpropagation
            loss = actor_loss + 0.5 * critic_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Clear buffer
            self.buffer = []

class ICM(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ICM, self).__init__()
        # Forward Model: Predict next state
        self.forward_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, state_dim)
        )
        # Inverse Model: Predict action from state transition
        self.inverse_model = nn.Sequential(
            nn.Linear(2 * state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, state, next_state, action):
        # Predict action (inverse model)
        action_pred = self.inverse_model(torch.cat([state, next_state], dim=1))
        # Predict next state (forward model)
        state_action = torch.cat([state, action], dim=1)
        next_state_pred = self.forward_model(state_action)
        return action_pred, next_state_pred    

class ICM_Agent(DQNAgent):
    def __init__(self, state_dim, action_dim, icm=True, **kwargs):
        super().__init__(state_dim, action_dim, **kwargs)
        self.icm = ICM(state_dim, action_dim) if icm else None
        self.icm_optimizer = optim.Adam(self.icm.parameters(), lr=LR) if icm else None

    def compute_intrinsic_reward(self, state, next_state, action):
        if self.icm is None:
            return 0.0
        state = torch.FloatTensor(state).unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        action = torch.FloatTensor(np.eye(self.action_dim)[action]).unsqueeze(0)

        with torch.no_grad():
            _, next_state_pred = self.icm(state, next_state, action)
        intrinsic_reward = torch.norm(next_state - next_state_pred, p=2).item()
        return intrinsic_reward

    def train_with_icm(self, beta=0.2):
        if len(self.buffer) < BATCH_SIZE:
            return
        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)
        intrinsic_rewards = [self.compute_intrinsic_reward(s, ns, a) for s, ns, a in zip(states, next_states, actions)]
        total_rewards = rewards + beta * np.array(intrinsic_rewards)

        # Standard DQN Training
        self.train_dqn(states, actions, total_rewards, next_states, dones)

        # Train ICM (Forward Model Loss)
        if self.icm:
            states = torch.FloatTensor(states)
            next_states = torch.FloatTensor(next_states)
            actions_onehot = torch.FloatTensor(np.eye(self.action_dim)[actions])
            _, next_state_pred = self.icm(states, next_states, actions_onehot)
            forward_loss = nn.MSELoss()(next_state_pred, next_states)

            self.icm_optimizer.zero_grad()
            forward_loss.backward()
            self.icm_optimizer.step()           

class MultiEnvManager:
    def __init__(self, envs, agent_types):
        self.envs = [gym.make(env) for env in envs]
        self.agents = [
            PPOAgent(env.observation_space.shape[0], env.action_space.n) if agent_type == "ppo" else
            DQNAgent(env.observation_space.shape[0], env.action_space.n)
            for env, agent_type in zip(self.envs, agent_types)
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
                        if isinstance(agent, PPOAgent):
                            action, log_prob = agent.select_action(states[i])
                            next_state, reward, done, _ = env.step(action)
                            agent.store_transition((states[i], action, reward, log_prob, done))
                        else:
                            action = agent.select_action(states[i])
                            next_state, reward, done, _ = env.step(action)
                            agent.buffer.add((states[i], action, reward, next_state, done))
                            agent.train()
                        total_rewards[i] += reward
                        states[i] = next_state
                        dones[i] = done

            for agent in self.agents:
                if isinstance(agent, PPOAgent):
                    agent.train()
                else:
                    agent.update_target_network()
            # Logging
            self.episode_logs.append([episode] + total_rewards)
            self.episode_logs.append([episode] + total_rewards)

            # Epsilon decay (only for DQN agents)
            for agent in self.agents:
                if isinstance(agent, DQNAgent):
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
    agent_types = ["dqn", "ppo", "dqn"]  # DQN for Cart Pole and Space Invader, PPO for Lunar Lander
    manager = MultiEnvManager(envs, agent_types)
    manager.train(episodes=1000)

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
